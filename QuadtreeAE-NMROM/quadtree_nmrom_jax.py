"""
quadtree_nmrom_jax.py
---------------------
JAX/Flax port of CompressedBottleneckAE + NM-ROM GN solver
on 2D adaptive quadtree meshes.

PDE:  -∇²u = f  on [0,1]²,  u = 0 on ∂Ω
f(x,y; k1,k2) = sin(k1 π x) sin(k2 π y)
u(x,y; k1,k2) = sin(k1 π x) sin(k2 π y) / ((k1²+k2²)π²)

Quadtree refined on |∇u| (solution gradient).

Key design:
  - CompressedBottleneckAE (D_BOTTLE=3, emb_dim=6) -> 510-dim latent z
  - z = concat[E0(6), E1(24), E2(96), E3(384)]
  - NM-ROM: GN in 510-dim space, depth-scaled diagonal preconditioner
  - FVM cell-centred Laplacian on adaptive quadtree
  - Differentiable spmv -> JAX autodiff computes J^T automatically
"""
import os, sys, time, pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import jax.scipy.sparse.linalg as jspla
import flax.linen as nn
import optax
from functools import partial
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print(f"JAX {jax.__version__}  |  devices: {jax.devices()}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
MAX_LEVEL   = 7
MIN_LEVEL   = 6
D_BOTTLE    = 3
EMB_DIM     = 6
HIDDEN      = 128
POS_FREQS   = 6
POS_DIM     = 3 + 6 * POS_FREQS   # 39

DEPTH_SIZES = [4**d for d in range(MAX_LEVEL + 1)]
# Depths 0-3 always fully refined (MIN_LEVEL=6 > D_BOTTLE=3),
# so latent dim is always fixed.
LATENT_DIM  = sum(DEPTH_SIZES[:D_BOTTLE + 1]) * EMB_DIM   # (1+4+16+64)*6 = 510

# Depth-scale preconditioner: coarser depths get larger scale
_depth_scales = []
for _d in range(D_BOTTLE + 1):
    _depth_scales.extend([float(2**(D_BOTTLE - _d))] * (DEPTH_SIZES[_d] * EMB_DIM))
DEPTH_SCALE_VEC = jnp.array(_depth_scales, dtype=jnp.float32)  # (510,)

OUT_DIR = Path('plots_quad_nmrom')
OUT_DIR.mkdir(exist_ok=True)

print(f"LATENT_DIM={LATENT_DIM}  |  DEPTH_SIZES(0-3)={DEPTH_SIZES[:D_BOTTLE+1]}")
print(f"Preconditioner scale range: [{float(DEPTH_SCALE_VEC.min()):.1f}, "
      f"{float(DEPTH_SCALE_VEC.max()):.1f}]")

# ═══════════════════════════════════════════════════════════════════════════
# 2. MORTON CODE UTILITIES  (NumPy, host-side)
# ═══════════════════════════════════════════════════════════════════════════
def _interleave(x):
    x = np.asarray(x, np.int64) & 0x0000FFFF
    x = (x | (x <<  8)) & 0x00FF00FF
    x = (x | (x <<  4)) & 0x0F0F0F0F
    x = (x | (x <<  2)) & 0x33333333
    x = (x | (x <<  1)) & 0x55555555
    return x

def _deinterleave(x):
    x = np.asarray(x, np.int64) & 0x55555555
    x = (x | (x >>  1)) & 0x33333333
    x = (x | (x >>  2)) & 0x0F0F0F0F
    x = (x | (x >>  4)) & 0x00FF00FF
    x = (x | (x >>  8)) & 0x0000FFFF
    return x

def xy2key(x, y):
    return (_interleave(x) | (_interleave(y) << 1)).astype(np.int64)

def key2xy(k):
    k = np.asarray(k, np.int64)
    return _deinterleave(k), _deinterleave(k >> 1)

# ═══════════════════════════════════════════════════════════════════════════
# 3. PDE — ANALYTICAL SOLUTION & FORCING
# ═══════════════════════════════════════════════════════════════════════════
def poisson_u(x, y, k1, k2):
    return np.sin(k1*np.pi*x) * np.sin(k2*np.pi*y) / ((k1**2 + k2**2) * np.pi**2)

def poisson_f(x, y, k1, k2):
    return np.sin(k1*np.pi*x) * np.sin(k2*np.pi*y)

def grad_mag_u(x, y, k1, k2):
    c   = 1.0 / ((k1**2 + k2**2) * np.pi**2)
    gx  = c * k1 * np.pi * np.cos(k1*np.pi*x) * np.sin(k2*np.pi*y)
    gy  = c * k2 * np.pi * np.sin(k1*np.pi*x) * np.cos(k2*np.pi*y)
    return np.sqrt(gx**2 + gy**2)

# ═══════════════════════════════════════════════════════════════════════════
# 4. QUADTREE BUILDER  (NumPy, host-side)
# ═══════════════════════════════════════════════════════════════════════════
class _QuadNode:
    __slots__ = ('x','y','size','level','children','val')
    def __init__(self, x, y, size, level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.children = []
        self.val = None

    def subdivide(self, k1, k2, grad_thresh=0.08):
        if self.level >= MAX_LEVEL:
            return
        cx, cy = self.x + self.size/2, self.y + self.size/2
        force   = self.level < MIN_LEVEL
        refine  = force or (grad_mag_u(cx, cy, k1, k2) > grad_thresh)
        if refine:
            h = self.size / 2
            self.children = [
                _QuadNode(self.x,   self.y,   h, self.level+1),
                _QuadNode(self.x+h, self.y,   h, self.level+1),
                _QuadNode(self.x,   self.y+h, h, self.level+1),
                _QuadNode(self.x+h, self.y+h, h, self.level+1),
            ]
            for c in self.children:
                c.subdivide(k1, k2, grad_thresh)

    def collect_leaves(self, out, k1, k2):
        if not self.children:
            cx, cy   = self.x + self.size/2, self.y + self.size/2
            self.val = poisson_u(cx, cy, k1, k2)
            out.append(self)
        else:
            for c in self.children:
                c.collect_leaves(out, k1, k2)


def _build_neighs(keys, depth):
    N = len(keys)
    if N == 0:
        return np.full((0, 9), -1, np.int64)
    if depth == 0:
        m = np.full((1, 9), -1, np.int64); m[0,4] = 0; return m
    xi, yi = key2xy(keys)
    off    = np.array([[-1,-1],[-1,0],[-1,1],
                       [ 0,-1],[ 0,0],[ 0,1],
                       [ 1,-1],[ 1,0],[ 1,1]], np.int64)
    coords = np.stack([xi, yi], 1)[:,None,:] + off[None]   # (N,9,2)
    res    = 1 << depth
    nx, ny = coords[...,0], coords[...,1]
    valid  = (nx>=0) & (nx<res) & (ny>=0) & (ny<res)
    nk     = np.full((N,9), -1, np.int64)
    if valid.any():
        nk[valid] = xy2key(nx[valid], ny[valid])
    idx    = np.searchsorted(keys, np.where(nk >= 0, nk, 0)).clip(0, N-1)
    found  = valid & (keys[idx] == nk)
    return np.where(found, idx, -1).astype(np.int64)


def build_quadtree(k1, k2):
    root   = _QuadNode(0.0, 0.0, 1.0, 0)
    root.subdivide(k1, k2)
    leaves = []
    root.collect_leaves(leaves, k1, k2)

    # Bucket leaves by depth
    lk_by_d = [[] for _ in range(MAX_LEVEL+1)]
    lv_by_d = [[] for _ in range(MAX_LEVEL+1)]
    for nd in leaves:
        d   = nd.level
        res = 1 << d
        ix  = min(int(nd.x * res), res-1)
        iy  = min(int(nd.y * res), res-1)
        k   = int(xy2key(np.array([ix]), np.array([iy]))[0])
        lk_by_d[d].append(k)
        lv_by_d[d].append(nd.val)

    # Build topology
    keys = [None]*(MAX_LEVEL+1)
    for d in range(MAX_LEVEL+1):
        arr     = np.array(lk_by_d[d], np.int64)
        keys[d] = np.unique(arr) if len(arr) > 0 else np.array([], np.int64)
    if len(keys[0]) == 0:
        keys[0] = np.array([0], np.int64)

    # Ancestor closure: every leaf's ancestor must exist
    for d in range(MAX_LEVEL, 0, -1):
        if len(keys[d]) == 0: continue
        parents   = np.unique(keys[d] >> 2)
        keys[d-1] = np.unique(np.concatenate([keys[d-1], parents]))

    # Children indices (sentinel = DEPTH_SIZES[d+1] = "no child")
    children = []
    for d in range(MAX_LEVEL):
        kd, kn = keys[d], keys[d+1]
        S       = DEPTH_SIZES[d+1]
        if len(kd) == 0:
            children.append(np.full((0,4), S, np.int64)); continue
        ck  = (kd[:,None] << 2) + np.arange(4, dtype=np.int64)
        if len(kn) == 0:
            children.append(np.full((len(kd),4), S, np.int64)); continue
        idx = np.searchsorted(kn, ck).clip(0, len(kn)-1)
        ch  = np.where(kn[idx] == ck, idx, S)
        children.append(ch.astype(np.int64))

    # Parent indices
    parents_idx = [None]*(MAX_LEVEL+1)
    parents_idx[0] = np.array([], np.int64)
    for d in range(1, MAX_LEVEL+1):
        kd, kp  = keys[d], keys[d-1]
        if len(kd) == 0:
            parents_idx[d] = np.array([], np.int64); continue
        parents_idx[d] = np.searchsorted(kp, kd>>2).clip(0, len(kp)-1).astype(np.int64)

    # Neighbours
    neighs = [_build_neighs(keys[d], d) for d in range(MAX_LEVEL+1)]

    # Leaf mask + split GT
    leaf_mask = [None]*(MAX_LEVEL+1)
    split_gt  = [None]*MAX_LEVEL
    for d in range(MAX_LEVEL+1):
        Nd = len(keys[d])
        if d == MAX_LEVEL:
            leaf_mask[d] = np.ones(Nd, bool)
        else:
            ch           = children[d]
            is_leaf      = (ch == DEPTH_SIZES[d+1]).all(1) if Nd > 0 else np.array([], bool)
            leaf_mask[d] = is_leaf
            split_gt[d]  = (~is_leaf).astype(np.float32) if Nd > 0 else np.array([], np.float32)

    # GT values per node (leaf-averaged)
    values_gt = [np.zeros((len(keys[d]),1), np.float32) for d in range(MAX_LEVEL+1)]
    for d in range(MAX_LEVEL+1):
        kd     = keys[d]
        arr_k  = np.array(lk_by_d[d], np.int64)
        arr_v  = np.array(lv_by_d[d], np.float32)
        if len(arr_k) == 0: continue
        uk, inv = np.unique(arr_k, return_inverse=True)
        pooled  = np.zeros(len(uk), np.float32)
        cnt     = np.zeros(len(uk), np.float32)
        np.add.at(pooled, inv, arr_v)
        np.add.at(cnt,    inv, 1.0)
        pooled /= cnt.clip(1)
        idx2   = np.searchsorted(kd, uk).clip(0, len(kd)-1)
        found2 = kd[idx2] == uk
        values_gt[d][idx2[found2], 0] = pooled[found2]

    # Bottleneck ancestor index: for each node at depth d,
    # store its ancestor's index in keys[D_BOTTLE]
    bottle_anc = [None]*(MAX_LEVEL+1)
    for d in range(MAX_LEVEL+1):
        Nd = len(keys[d])
        Sb = DEPTH_SIZES[D_BOTTLE]
        if d <= D_BOTTLE:
            # Self-reference into padded E[d] (first DEPTH_SIZES[d] entries)
            bottle_anc[d] = np.arange(DEPTH_SIZES[d], dtype=np.int32)
        else:
            shift     = 2*(d - D_BOTTLE)
            anc_keys  = keys[d] >> shift
            kb        = keys[D_BOTTLE]
            if len(kb) == 0:
                bottle_anc[d] = np.zeros(Nd, np.int32)
            else:
                idx = np.searchsorted(kb, anc_keys).clip(0, len(kb)-1).astype(np.int32)
                bottle_anc[d] = idx

    return {
        'keys': keys, 'children': children, 'parents': parents_idx,
        'neighs': neighs, 'leaf_mask': leaf_mask, 'split_gt': split_gt,
        'values_gt': values_gt, 'bottle_anc': bottle_anc,
        'k1': k1, 'k2': k2, 'leaves': leaves,
    }

# ═══════════════════════════════════════════════════════════════════════════
# 5. POSITIONAL ENCODING  (NumPy)
# ═══════════════════════════════════════════════════════════════════════════
def _node_pos_enc(keys, depth):
    if len(keys) == 0:
        return np.zeros((0, POS_DIM), np.float32)
    xi, yi = key2xy(keys)
    h      = 1.0 / (1 << depth)
    xc     = (xi.astype(np.float32) + 0.5) * h
    yc     = (yi.astype(np.float32) + 0.5) * h
    dn     = np.full(len(keys), float(depth)/float(MAX_LEVEL), np.float32)
    pos3   = np.stack([xc, yc, dn], 1)  # (N, 3)
    freqs  = 2.0**np.arange(POS_FREQS, dtype=np.float32)
    x_sin  = pos3[:,:,None] * np.pi * 2.0 * freqs[None,None,:]
    enc    = np.concatenate([np.sin(x_sin), np.cos(x_sin)], -1).reshape(len(keys), -1)
    return np.concatenate([pos3, enc], 1).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# 6. PADDED ARRAY CONVERTER  (quadtree -> fixed-shape JAX arrays)
# ═══════════════════════════════════════════════════════════════════════════
def topology_to_padded(qt):
    pd = {k: [] for k in ('feats','pos','neigh','children','parents',
                           'valid','leaf','split_gt','values_gt','bottle_anc')}
    for d in range(MAX_LEVEL+1):
        S  = DEPTH_SIZES[d]
        Nd = len(qt['keys'][d])

        vm = np.zeros(S, bool);  vm[:Nd] = True
        pd['valid'].append(jnp.array(vm))

        f = np.zeros((S,1), np.float32);  f[:Nd] = qt['values_gt'][d]
        pd['feats'].append(jnp.array(f))

        pos_np = np.zeros((S, POS_DIM), np.float32)
        pos_np[:Nd] = _node_pos_enc(qt['keys'][d], d)
        pd['pos'].append(jnp.array(pos_np))

        # neighbour: invalid -> S (sentinel -> zero-padded row)
        nh_raw = qt['neighs'][d]
        nh_np  = np.full((S,9), S, np.int32)
        if Nd > 0:
            nh_np[:Nd] = np.where(nh_raw >= 0, nh_raw, S)
        pd['neigh'].append(jnp.array(nh_np))

        lm = np.zeros(S, bool);  lm[:Nd] = qt['leaf_mask'][d]
        pd['leaf'].append(jnp.array(lm))

        vg = np.zeros((S,1), np.float32);  vg[:Nd] = qt['values_gt'][d]
        pd['values_gt'].append(jnp.array(vg))

        sg = np.zeros(S, np.float32)
        if d < MAX_LEVEL and qt['split_gt'][d] is not None and len(qt['split_gt'][d]) > 0:
            sg[:Nd] = qt['split_gt'][d]
        pd['split_gt'].append(jnp.array(sg))

        ban = np.zeros(S, np.int32)
        if Nd > 0: ban[:Nd] = qt['bottle_anc'][d]
        pd['bottle_anc'].append(jnp.array(ban))

        if d < MAX_LEVEL:
            S_next = DEPTH_SIZES[d+1]
            ch_raw = qt['children'][d]
            ch_np  = np.full((S,4), S_next, np.int32)
            if Nd > 0: ch_np[:Nd] = ch_raw.clip(0, S_next)
            pd['children'].append(jnp.array(ch_np))
        else:
            pd['children'].append(None)

        if d > 0:
            pr_np  = np.zeros(S, np.int32)
            pr_raw = qt['parents'][d]
            if len(pr_raw) > 0: pr_np[:Nd] = pr_raw[:Nd]
            pd['parents'].append(jnp.array(pr_np))
        else:
            pd['parents'].append(None)

    return pd

# ═══════════════════════════════════════════════════════════════════════════
# 7. FLAX MODEL
# ═══════════════════════════════════════════════════════════════════════════
class CompressedEncoder(nn.Module):
    hidden:  int = HIDDEN
    emb_dim: int = EMB_DIM

    @nn.compact
    def __call__(self, pd):
        # Input projection shared across depths
        in_proj   = nn.Dense(self.hidden, name='in_proj')
        enc_convs = [nn.Dense(self.hidden, name=f'enc_conv_{d}') for d in range(MAX_LEVEL)]
        emb_projs = [nn.Dense(self.emb_dim, name=f'emb_proj_{d}') for d in range(D_BOTTLE+1)]
        emb_norms = [nn.LayerNorm(name=f'emb_norm_{d}') for d in range(D_BOTTLE+1)]

        h = []
        for d in range(MAX_LEVEL+1):
            x    = jnp.concatenate([pd['feats'][d], pd['pos'][d]], axis=1)
            h_d  = jax.nn.relu(in_proj(x))
            h_d  = h_d * pd['valid'][d][:,None]
            h.append(h_d)

        # Bottom-up pooling
        for d in range(MAX_LEVEL, 0, -1):
            S_p    = DEPTH_SIZES[d-1]
            v      = pd['valid'][d].astype(jnp.float32)
            pooled = jnp.zeros((S_p, self.hidden)).at[pd['parents'][d]].add(h[d] * v[:,None])
            cnt    = jnp.zeros((S_p, 1)).at[pd['parents'][d]].add(v[:,None])
            h[d-1] = h[d-1] + pooled / cnt.clip(1.0)
            # QuadConv at parent depth
            S      = DEPTH_SIZES[d-1]
            h_pad  = jnp.concatenate([h[d-1], jnp.zeros((1, self.hidden))], 0)
            flat   = h_pad[pd['neigh'][d-1]].reshape(S, -1)
            h[d-1] = jax.nn.relu(enc_convs[d-1](flat)) * pd['valid'][d-1][:,None]

        E = []
        for d in range(D_BOTTLE+1):
            emb = emb_projs[d](h[d])
            emb = emb_norms[d](emb)
            E.append(emb * pd['valid'][d][:,None])
        return E


class CompressedDecoder(nn.Module):
    hidden:  int = HIDDEN
    emb_dim: int = EMB_DIM
    out_c:   int = 1

    @nn.compact
    def __call__(self, E, pd):
        root_token  = self.param('root_token', nn.initializers.normal(0.01), (1, self.hidden))
        skip_norm   = nn.LayerNorm(name='skip_norm')
        fuse1       = nn.Dense(self.hidden, name='fuse1')
        fuse2       = nn.Dense(self.hidden, name='fuse2')
        mix_convs   = [nn.Dense(self.hidden, name=f'mix_{d}') for d in range(MAX_LEVEL+1)]
        child_head1 = nn.Dense(self.hidden, name='child_h1')
        child_head2 = nn.Dense(4*self.hidden, name='child_h2')
        split_h     = nn.Dense(self.hidden, name='split_h')
        split_out   = nn.Dense(1, name='split_out')
        val_h       = nn.Dense(self.hidden, name='val_h')
        val_out     = nn.Dense(self.out_c, name='val_out')

        def get_skip(d):
            if d <= D_BOTTLE:
                return skip_norm(E[d])
            else:
                return skip_norm(E[D_BOTTLE][pd['bottle_anc'][d]])

        S0    = DEPTH_SIZES[0]
        h_cur = [jnp.broadcast_to(root_token, (S0, self.hidden))]
        for d in range(1, MAX_LEVEL+1):
            h_cur.append(jnp.zeros((DEPTH_SIZES[d], self.hidden)))

        split_logits = [None]*MAX_LEVEL
        val_pred     = [None]*(MAX_LEVEL+1)

        for d in range(MAX_LEVEL+1):
            h    = h_cur[d]
            skip = get_skip(d)
            x    = jnp.concatenate([h, skip, pd['pos'][d]], axis=1)
            h    = jax.nn.relu(fuse1(x))
            h    = fuse2(h)

            # QuadConv spatial mixing
            S      = DEPTH_SIZES[d]
            h_pad  = jnp.concatenate([h, jnp.zeros((1, self.hidden))], 0)
            flat   = h_pad[pd['neigh'][d]].reshape(S, -1)
            h      = jax.nn.relu(mix_convs[d](flat))
            h      = h * pd['valid'][d][:,None]

            val_pred[d] = val_out(jax.nn.relu(val_h(h)))

            if d == MAX_LEVEL:
                break

            split_logits[d] = split_out(jax.nn.relu(split_h(h))).squeeze(-1)

            # Scatter child states
            S_next     = DEPTH_SIZES[d+1]
            cf         = child_head2(jax.nn.relu(child_head1(h)))
            cf         = cf.reshape(DEPTH_SIZES[d], 4, self.hidden)
            ch         = pd['children'][d]   # (S, 4), sentinel=S_next
            h_next     = jnp.zeros((S_next, self.hidden))
            for c in range(4):
                valid_c = ch[:,c] < S_next
                safe_i  = jnp.where(valid_c, ch[:,c], 0)
                h_next  = h_next.at[safe_i].add(cf[:,c] * valid_c[:,None])
            h_cur[d+1] = h_next

        return split_logits, val_pred



class CompressedBottleneckAE(nn.Module):
    hidden:  int = HIDDEN
    emb_dim: int = EMB_DIM

    def setup(self):
        self._enc = CompressedEncoder(self.hidden, self.emb_dim)
        self._dec = CompressedDecoder(self.hidden, self.emb_dim)

    def __call__(self, pd):
        E = self._enc(pd)
        split_logits, val_pred = self._dec(E, pd)
        return split_logits, val_pred, E

    def encode(self, pd):
        return self._enc(pd)

    def decode(self, E, pd):
        return self._dec(E, pd)

# ═══════════════════════════════════════════════════════════════════════════
# 8. LATENT VECTOR <-> E LIST
# ═══════════════════════════════════════════════════════════════════════════
def z_to_E(z):
    E, idx = [], 0
    for d in range(D_BOTTLE+1):
        n = DEPTH_SIZES[d] * EMB_DIM
        E.append(z[idx:idx+n].reshape(DEPTH_SIZES[d], EMB_DIM))
        idx += n
    return E

def E_to_z(E):
    return jnp.concatenate([E[d].reshape(-1) for d in range(D_BOTTLE+1)])

# ═══════════════════════════════════════════════════════════════════════════
# 9. FVM OPERATOR  K
#
# Cell-centred Laplacian on adaptive quadtree.
# For each leaf pair sharing a face, conductance = face_length / face_distance.
# Handles same-level, coarser, and finer neighbours.
# K_op(u) is differentiable via JAX scatter-add.
# ═══════════════════════════════════════════════════════════════════════════
def build_fvm_operator(qt):
    # Global index: (depth, morton_key) -> leaf index
    global_idx = {}
    leaf_list  = []
    gidx       = 0
    for d in range(MAX_LEVEL+1):
        kd = qt['keys'][d]
        lm = qt['leaf_mask'][d]
        for nd_i, is_leaf in enumerate(lm):
            if is_leaf:
                k        = int(kd[nd_i])
                xi, yi   = key2xy(np.array([k]))
                xi, yi   = int(xi[0]), int(yi[0])
                h        = 1.0 / (1 << d)
                xc, yc   = (xi + 0.5)*h, (yi + 0.5)*h
                global_idx[(d, k)] = gidx
                leaf_list.append((xc, yc, h, d, k, xi, yi))
                gidx += 1

    N = gidx
    rows_l, cols_l, vals_l = [], [], []

    for gi, (xc, yc, h, d, k, xi, yi) in enumerate(leaf_list):
        diag = 0.0
        res  = 1 << d
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx_, ny_ = xi+dx, yi+dy

            # Dirichlet boundary
            if nx_ < 0 or nx_ >= res or ny_ < 0 or ny_ >= res:
                cond  = h / (h/2)   # face_len/face_dist
                diag += cond
                continue

            nk_ = int(xy2key(np.array([nx_]), np.array([ny_]))[0])

            # Same-level neighbour
            if (d, nk_) in global_idx:
                gj    = global_idx[(d, nk_)]
                cond  = h / h
                rows_l.append(gi); cols_l.append(gj); vals_l.append(cond)
                diag += cond
                continue

            # Coarser neighbour (one level up)
            found = False
            if d > 0:
                nx_p, ny_p = nx_//2, ny_//2
                res_p      = 1 << (d-1)
                if 0 <= nx_p < res_p and 0 <= ny_p < res_p:
                    nk_p = int(xy2key(np.array([nx_p]), np.array([ny_p]))[0])
                    if (d-1, nk_p) in global_idx:
                        gj        = global_idx[(d-1, nk_p)]
                        face_dist = h/2 + h   # fine_half + coarse_half
                        cond      = h / face_dist
                        rows_l.append(gi); cols_l.append(gj); vals_l.append(cond)
                        diag += cond
                        found = True
            if found:
                continue

            # Finer neighbours (one level down, 2 cells share this face)
            if d < MAX_LEVEL:
                hc    = h / 2
                res_c = 1 << (d+1)
                if dx != 0:
                    pairs = [(nx_*2 + (1 if dx>0 else 0), ny_*2),
                             (nx_*2 + (1 if dx>0 else 0), ny_*2+1)]
                else:
                    pairs = [(nx_*2,   ny_*2 + (1 if dy>0 else 0)),
                             (nx_*2+1, ny_*2 + (1 if dy>0 else 0))]
                for nx_c, ny_c in pairs:
                    if 0 <= nx_c < res_c and 0 <= ny_c < res_c:
                        nk_c = int(xy2key(np.array([nx_c]), np.array([ny_c]))[0])
                        if (d+1, nk_c) in global_idx:
                            gj        = global_idx[(d+1, nk_c)]
                            face_dist = h/2 + hc/2
                            cond      = hc / face_dist
                            rows_l.append(gi); cols_l.append(gj); vals_l.append(cond)
                            diag += cond

        # Diagonal entry
        rows_l.append(gi); cols_l.append(gi); vals_l.append(-diag)

    rows  = jnp.array(rows_l, dtype=jnp.int32)
    cols  = jnp.array(cols_l, dtype=jnp.int32)
    vals  = jnp.array(vals_l, dtype=jnp.float32)

    # Forcing and reference
    k1_, k2_  = qt['k1'], qt['k2']
    xc_arr    = np.array([lf[0] for lf in leaf_list], np.float32)
    yc_arr    = np.array([lf[1] for lf in leaf_list], np.float32)
    f_vec     = jnp.array(poisson_f(xc_arr, yc_arr, k1_, k2_), jnp.float32)
    u_ref     = jnp.array(poisson_u(xc_arr, yc_arr, k1_, k2_), jnp.float32)

    @jit
    def K_op(u):
        return jnp.zeros(N).at[rows].add(vals * u[cols])

    return K_op, f_vec, u_ref, N, global_idx, leaf_list

# ═══════════════════════════════════════════════════════════════════════════
# 10. EXTRACT LEAF VALUES FROM DECODER OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
def build_leaf_gather(qt, global_idx, pd):
    """Precompute, for each depth, the (padded_idx, global_idx) pairs of leaves.
    Returns lists of jnp arrays so the inner loop in `extract_leaf_values`
    becomes a handful of vectorized gathers/scatters instead of thousands of
    scalar .at[].set() ops."""
    N = len(global_idx)
    pad_idx_per_d = []
    glob_idx_per_d = []
    for d in range(MAX_LEVEL + 1):
        kd = qt['keys'][d]
        lm = qt['leaf_mask'][d]
        pad_l, glob_l = [], []
        for nd_i, is_leaf in enumerate(lm):
            if is_leaf:
                k  = int(kd[nd_i])
                gi = global_idx.get((d, k))
                if gi is not None:
                    pad_l.append(nd_i); glob_l.append(gi)
        pad_idx_per_d.append(jnp.asarray(pad_l, dtype=jnp.int32))
        glob_idx_per_d.append(jnp.asarray(glob_l, dtype=jnp.int32))
    return N, pad_idx_per_d, glob_idx_per_d


def extract_leaf_values(val_pred, leaf_gather):
    N, pad_idx_per_d, glob_idx_per_d = leaf_gather
    u = jnp.zeros(N)
    for d in range(MAX_LEVEL + 1):
        if pad_idx_per_d[d].size == 0:
            continue
        vals = val_pred[d][pad_idx_per_d[d], 0]
        u    = u.at[glob_idx_per_d[d]].set(vals)
    return u

# ═══════════════════════════════════════════════════════════════════════════
# 11. NM-ROM SOLVER  (Gauss-Newton + depth-scaled preconditioner)
# ═══════════════════════════════════════════════════════════════════════════
def make_nmrom_solver(model, params, pd, qt, K_op, f_vec, global_idx,
                      max_outer=8, max_inner=6, tol=1e-5):
    """
    GN loop in 510-dim latent space.

    Preconditioned normal equations (depth-scaled diagonal P):
        P^{-1} J^T K^T K J P^{-1} dz~ = -P^{-1} J^T K^T r
    where dz~ = P dz, so dz = P^{-1} dz~.

    Inner CG uses JVP/VJP of the differentiable pipeline:
        z -> E_list -> decoder -> leaf_values -> K_op -> residual
    """
    P_inv       = 1.0 / DEPTH_SCALE_VEC   # (510,)
    leaf_gather = build_leaf_gather(qt, global_idx, pd)

    def constrained_decode(z):
        E = z_to_E(z)
        _, vpred = model.apply({'params': params}, E, pd, method=model.decode)
        return extract_leaf_values(vpred, leaf_gather)

    # JIT only the decode forward (heavy but compiled once and reused
    # by both vjp and jvp paths during the eager GN loop).
    constrained_decode_jit = jit(constrained_decode)

    def gn_step(z):
        # NM-ROM Petrov-Galerkin: solve (J^T K J) dz = -J^T r,
        # where r = K u(z) - f. Matches quadtree_nmrom.py's working solver.
        u_pred, vjp_fn = jax.vjp(constrained_decode, z)
        r              = K_op(u_pred) - f_vec
        g_red          = vjp_fn(r)[0]                       # (LATENT_DIM,)

        def GN_op_tilde(dz_tilde):
            dz       = P_inv * dz_tilde
            _, Jdz   = jax.jvp(constrained_decode, (z,), (dz,))
            KJdz     = K_op(Jdz)
            JtKJdz   = vjp_fn(KJdz)[0]
            return P_inv * JtKJdz

        rhs         = -(P_inv * g_red)
        dz_tilde, _ = jspla.cg(GN_op_tilde, rhs, tol=1e-3, maxiter=30)
        dz          = P_inv * dz_tilde
        return z + dz, jnp.linalg.norm(r), jnp.linalg.norm(g_red)

    def solve(z_init=None, verbose=False):
        z       = jnp.zeros(LATENT_DIM) if z_init is None else z_init
        history = []
        total_iters = max_outer * max_inner
        for it in range(total_iters):
            z, rn, gn = gn_step(z)
            rn_f = float(rn); gn_f = float(gn)
            history.append(rn_f)
            if verbose:
                print(f"    GN iter {it:3d}  ||r||={rn_f:.4e}  ||J^T r||={gn_f:.4e}",
                      flush=True)
            if rn_f < tol or gn_f < tol:
                break
        u_pred = constrained_decode_jit(z)
        return z, u_pred, history

    return solve, constrained_decode

# ═══════════════════════════════════════════════════════════════════════════
# 12. TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train(model, n_steps_p1=4000, n_steps_p2=2000, lr_p1=1e-3, lr_p2=1e-4, seed=42):
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    # Init on a dummy sample
    qt0    = build_quadtree(1.0, 1.0)
    pd0    = topology_to_padded(qt0)
    key, sk = jax.random.split(key)
    params  = model.init(sk, pd0)['params']
    n_p     = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model params: {n_p:,}")

    def loss_fn(params, pd):
        split_logits, val_pred, E = model.apply({'params': params}, pd)

        L_val = jnp.array(0.0); n_val = 0
        for d in range(MAX_LEVEL+1):
            if val_pred[d] is None: continue
            mask   = pd['leaf'][d] & pd['valid'][d]
            n_ok   = jnp.sum(mask).clip(1)
            diff2  = jnp.where(mask[:,None], (val_pred[d] - pd['values_gt'][d])**2, 0.0)
            L_val += jnp.sum(diff2) / n_ok
            n_val += 1
        if n_val: L_val /= n_val

        L_split = jnp.array(0.0); n_split = 0
        for d in range(MAX_LEVEL):
            if split_logits[d] is None: continue
            sl   = split_logits[d]
            sg   = pd['split_gt'][d]
            v    = pd['valid'][d]
            bce  = jax.nn.softplus(-sl)*sg + jax.nn.softplus(sl)*(1-sg)
            n_ok = jnp.sum(v).clip(1)
            L_split += jnp.sum(jnp.where(v, bce, 0.0)) / n_ok
            n_split += 1
        if n_split: L_split /= n_split

        return L_val + 0.5*L_split, (L_val, L_split)

    loss_and_grad = jit(jax.value_and_grad(loss_fn, has_aux=True))

    def run_phase(params, n_steps, lr, name):
        warmup    = min(200, max(1, n_steps // 5))
        schedule  = optax.warmup_cosine_decay_schedule(0., lr, warmup, n_steps, 1e-6)
        tx        = optax.adamw(schedule, weight_decay=1e-4)
        opt_state = tx.init(params)

        @jit
        def update_step(params, opt_state, pd_):
            (loss, (lv, ls)), grads = loss_and_grad(params, pd_)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, lv, ls

        best_loss, best_params = float('inf'), params
        hist = []
        print(f"\n{'='*55}\n  {name}  ({n_steps} steps)\n{'='*55}", flush=True)
        t0 = time.perf_counter()
        for step in range(n_steps):
            k1 = float(np.random.uniform(1.0, 5.0))
            k2 = float(np.random.uniform(1.0, 5.0))
            qt_ = build_quadtree(k1, k2)
            pd_ = topology_to_padded(qt_)
            params, opt_state, loss, lv, ls = update_step(params, opt_state, pd_)
            lf = float(loss)
            hist.append(lf)
            if lf < best_loss:
                best_loss, best_params = lf, params
            if step % 200 == 0 or step == n_steps-1:
                print(f"  step {step:4d}/{n_steps}  loss={lf:.5f}  "
                      f"val={float(lv):.5f}  split={float(ls):.5f}  "
                      f"k=({k1:.1f},{k2:.1f})  best={best_loss:.5f}  "
                      f"t={time.perf_counter()-t0:.0f}s")
        return best_params, hist

    params, h1 = run_phase(params, n_steps_p1, lr_p1, "PHASE 1: Teacher Forcing")
    params, h2 = run_phase(params, n_steps_p2, lr_p2, "PHASE 2: Fine-tuning")
    return params, h1+h2

# ═══════════════════════════════════════════════════════════════════════════
# 13. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
def plot_solution(leaf_list, u_pred, u_true, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap  = plt.get_cmap('viridis')
    u_min, u_max = float(np.min(u_true)), float(np.max(u_true))
    e_max = float(np.max(np.abs(np.array(u_pred) - np.array(u_true))))

    for ax, vals, ttl, vmin, vmax in zip(
        axes,
        [u_true, u_pred, np.abs(np.array(u_pred)-np.array(u_true))],
        ['FOM (analytical)', 'NM-ROM', '|Error|'],
        [u_min, u_min, 0.0],
        [u_max, u_max, e_max+1e-10],
    ):
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(ttl, fontsize=11)
        for i, (xc, yc, h, d, k, xi, yi) in enumerate(leaf_list):
            v     = float(vals[i])
            norm  = (v - vmin) / (vmax - vmin)
            color = cmap(np.clip(norm, 0, 1))
            ax.add_patch(patches.Rectangle(
                (xi*h, yi*h), h, h,
                linewidth=0.2, edgecolor='k', facecolor=color))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        sm.set_array([]); plt.colorbar(sm, ax=ax, fraction=0.046)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_residuals(history, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(history, 'b-o', markersize=3)
    ax.set_xlabel('GN Iteration'); ax.set_ylabel('||K ũ - f||')
    ax.set_title('GN Residual History')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training_loss(history, save_path):
    split = 4000   # phase 1 steps
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history, linewidth=0.8)
    if len(history) > split:
        ax.axvline(split, color='r', linestyle='--', label='Phase 1→2')
        ax.legend()
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 14. MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    import sys
    CKPT        = Path('quadtree_nmrom_params.pkl')
    SKIP_TRAIN  = '--use-checkpoint' in sys.argv
    FORCE_TRAIN = '--retrain' in sys.argv

    model = CompressedBottleneckAE(hidden=HIDDEN, emb_dim=EMB_DIM)

    if (CKPT.exists() and not FORCE_TRAIN) or (CKPT.exists() and SKIP_TRAIN):
        print(f"Loading checkpoint: {CKPT}")
        with open(CKPT, 'rb') as f:
            params = pickle.load(f)
    else:
        print("Training from scratch...")
        params, hist = train(model, n_steps_p1=4000, n_steps_p2=2000)
        with open(CKPT, 'wb') as f:
            pickle.dump(params, f)
        print(f"Checkpoint saved: {CKPT}")
        plot_training_loss(hist, OUT_DIR / 'training_loss.png')

    # ── Test cases ────────────────────────────────────────────────────────
    TEST_CASES = [
        (1.5, 2.3),
        (2.1, 1.4),
        (3.0, 2.0),
        (1.2, 3.5),
    ]

    print(f"\n{'='*60}")
    print(f"  NM-ROM BENCHMARK  ({len(TEST_CASES)} test cases)")
    print(f"{'='*60}")
    rel_errors = []

    for ci, (k1, k2) in enumerate(TEST_CASES):
        print(f"\nCase {ci+1}: k1={k1}  k2={k2}")
        qt  = build_quadtree(k1, k2)
        pd  = topology_to_padded(qt)
        K_op, f_vec, u_ref, N, global_idx, leaf_list = build_fvm_operator(qt)
        print(f"  Leaf nodes: {N}")

        solver, _ = make_nmrom_solver(
            model, params, pd, qt, K_op, f_vec, global_idx,
            max_outer=8, max_inner=6, tol=1e-5,
        )

        # JIT warm-up
        _ = solver(verbose=False)

        t0 = time.perf_counter()
        z_star, u_pred, history = solver(verbose=True)
        t_rom = time.perf_counter() - t0

        u_t   = np.array(u_ref)
        u_p   = np.array(u_pred)
        rel_l2 = float(np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t))
        rel_errors.append(rel_l2)
        print(f"  time={t_rom:.3f}s  |  rel-L2={rel_l2:.4e}  |  GN iters={len(history)}")

        plot_solution(
            leaf_list, u_p, u_t,
            title=f"Quadtree NM-ROM  k1={k1} k2={k2}  RelL2={rel_l2:.3e}",
            save_path=OUT_DIR / f'case_{ci+1}.png',
        )
        plot_residuals(history, OUT_DIR / f'residuals_case_{ci+1}.png')

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for (k1,k2), err in zip(TEST_CASES, rel_errors):
        print(f"  k=({k1:.1f},{k2:.1f})  rel-L2 = {err:.4e}")
    print(f"  Mean rel-L2 = {np.mean(rel_errors):.4e}")
    print(f"\nAll outputs: {OUT_DIR}/")
    print("=== DONE ===")


if __name__ == '__main__':
    main()
