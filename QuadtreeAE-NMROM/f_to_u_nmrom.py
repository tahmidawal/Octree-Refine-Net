"""
f_to_u_nmrom.py
---------------
End-to-end pipeline for "given forcing F, recover solution u" via:

  F  ──► encoder ──► z_F  ──► MLP ──► z_u₀ ──► decoder ──► u₀
                                                  │
                                                  ▼
                                              NM-ROM GN
                                                  │
                                                  ▼
                                                z_u*  ──► decoder ──► u*

PDE:   -∇²u = f   on [0,1]², u=0 on ∂Ω
       f(x,y; k1, k2) = sin(k1·π·x) · sin(k2·π·y)
       u_exact = f / ((k1² + k2²)·π²)

Stages (each invokable as a CLI subcommand):
  data       — generate paired (F, u) quadtree samples, cache to disk
  train_ae   — train shared AE on union of F and u samples
  train_mlp  — train MLP z_F → z_u from encoded pairs
  bench      — benchmark NM-ROM with cold-start vs MLP-warm-start vs FOM
  all        — run all of the above in sequence

Built on top of quadtree_nmrom_jax.py's machinery (Morton, FVM, decoder, GN solver),
but independent so the existing scripts stay untouched.
"""
import argparse, os, pickle, sys, time
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy.sparse.linalg as jspla
import flax.linen as nn
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG (matches quadtree_nmrom_jax.py for compatibility)
# ═══════════════════════════════════════════════════════════════════════════
MAX_LEVEL   = 7
MIN_LEVEL   = 6
D_BOTTLE    = 3
EMB_DIM     = 6
HIDDEN      = 128
POS_FREQS   = 6
POS_DIM     = 3 + 6 * POS_FREQS    # 39

DEPTH_SIZES = [4**d for d in range(MAX_LEVEL + 1)]
LATENT_DIM  = sum(DEPTH_SIZES[:D_BOTTLE + 1]) * EMB_DIM   # 510

# u has natural amplitude ~ 1/(π²(k1²+k2²)). For k ∈ [1, 3]² the scaling factor
# π²(k1²+k2²) ranges roughly 20–180 → multiplying u by U_SCALE puts both F and
# u-views into the same dynamic range so MSE loss is balanced.
U_SCALE = 50.0

_depth_scales = []
for _d in range(D_BOTTLE + 1):
    _depth_scales.extend([float(2**(D_BOTTLE - _d))] * (DEPTH_SIZES[_d] * EMB_DIM))
DEPTH_SCALE_VEC = jnp.array(_depth_scales, dtype=jnp.float32)


# ═══════════════════════════════════════════════════════════════════════════
# MORTON
# ═══════════════════════════════════════════════════════════════════════════
def _interleave(x):
    x = np.asarray(x, np.int64) & 0x0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x

def _deinterleave(x):
    x = np.asarray(x, np.int64) & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF
    return x

def xy2key(x, y):
    return (_interleave(x) | (_interleave(y) << 1)).astype(np.int64)

def key2xy(k):
    return _deinterleave(k), _deinterleave(k >> 1)


# ═══════════════════════════════════════════════════════════════════════════
# PDE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def poisson_u(x, y, k1, k2):
    return np.sin(k1*np.pi*x) * np.sin(k2*np.pi*y) / ((k1**2 + k2**2) * np.pi**2)

def poisson_f(x, y, k1, k2):
    return np.sin(k1*np.pi*x) * np.sin(k2*np.pi*y)

def grad_mag_f_normalized(x, y, k1, k2):
    """|∇F| normalized to [0,1] by its theoretical maximum.
    Refines based on F (the only thing we have at solve time)."""
    gx  = k1 * np.pi * np.cos(k1*np.pi*x) * np.sin(k2*np.pi*y)
    gy  = k2 * np.pi * np.sin(k1*np.pi*x) * np.cos(k2*np.pi*y)
    g   = np.sqrt(gx*gx + gy*gy)
    g_max = np.pi * np.sqrt(k1**2 + k2**2)
    return g / g_max


# ═══════════════════════════════════════════════════════════════════════════
# QUADTREE BUILDER  — refines on |∇F|, stores BOTH F and u values
# ═══════════════════════════════════════════════════════════════════════════
class _QuadNode:
    __slots__ = ('x','y','size','level','children','val_f','val_u')
    def __init__(self, x, y, size, level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.children = []
        self.val_f = None
        self.val_u = None

    def subdivide(self, k1, k2, grad_thresh=0.5):
        if self.level >= MAX_LEVEL:
            return
        cx, cy = self.x + self.size/2, self.y + self.size/2
        force  = self.level < MIN_LEVEL
        refine = force or (grad_mag_f_normalized(cx, cy, k1, k2) > grad_thresh)
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
            cx, cy     = self.x + self.size/2, self.y + self.size/2
            self.val_f = poisson_f(cx, cy, k1, k2)
            self.val_u = poisson_u(cx, cy, k1, k2)
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
    coords = np.stack([xi, yi], 1)[:,None,:] + off[None]
    res    = 1 << depth
    nx, ny = coords[...,0], coords[...,1]
    valid  = (nx>=0) & (nx<res) & (ny>=0) & (ny<res)
    nk     = np.full((N,9), -1, np.int64)
    if valid.any():
        nk[valid] = xy2key(nx[valid], ny[valid])
    idx    = np.searchsorted(keys, np.where(nk >= 0, nk, 0)).clip(0, N-1)
    found  = valid & (keys[idx] == nk)
    return np.where(found, idx, -1).astype(np.int64)


def build_quadtree(k1, k2, grad_thresh=0.5):
    """Build adaptive quadtree refined on |∇F|. Returns dict with keys per depth,
    leaf masks, child/parent/neighbour indices, and BOTH F-values and u-values
    at each leaf node (interpolated to internal nodes via mean-pooling)."""
    root   = _QuadNode(0.0, 0.0, 1.0, 0)
    root.subdivide(k1, k2, grad_thresh)
    leaves = []
    root.collect_leaves(leaves, k1, k2)

    lk_by_d = [[] for _ in range(MAX_LEVEL+1)]
    lvf_by_d = [[] for _ in range(MAX_LEVEL+1)]
    lvu_by_d = [[] for _ in range(MAX_LEVEL+1)]
    for nd in leaves:
        d   = nd.level
        res = 1 << d
        ix  = min(int(nd.x * res), res-1)
        iy  = min(int(nd.y * res), res-1)
        k   = int(xy2key(np.array([ix]), np.array([iy]))[0])
        lk_by_d[d].append(k)
        lvf_by_d[d].append(nd.val_f)
        lvu_by_d[d].append(nd.val_u)

    keys = [None]*(MAX_LEVEL+1)
    for d in range(MAX_LEVEL+1):
        arr = np.array(lk_by_d[d], np.int64)
        keys[d] = np.unique(arr) if len(arr) > 0 else np.array([], np.int64)
    if len(keys[0]) == 0:
        keys[0] = np.array([0], np.int64)

    for d in range(MAX_LEVEL, 0, -1):
        if len(keys[d]) == 0: continue
        parents   = np.unique(keys[d] >> 2)
        keys[d-1] = np.unique(np.concatenate([keys[d-1], parents]))

    children = []
    for d in range(MAX_LEVEL):
        kd, kn = keys[d], keys[d+1]
        S      = DEPTH_SIZES[d+1]
        if len(kd) == 0:
            children.append(np.full((0,4), S, np.int64)); continue
        ck  = (kd[:,None] << 2) + np.arange(4, dtype=np.int64)
        if len(kn) == 0:
            children.append(np.full((len(kd),4), S, np.int64)); continue
        idx = np.searchsorted(kn, ck).clip(0, len(kn)-1)
        ch  = np.where(kn[idx] == ck, idx, S)
        children.append(ch.astype(np.int64))

    parents_idx = [None]*(MAX_LEVEL+1)
    parents_idx[0] = np.array([], np.int64)
    for d in range(1, MAX_LEVEL+1):
        kd, kp = keys[d], keys[d-1]
        if len(kd) == 0:
            parents_idx[d] = np.array([], np.int64); continue
        parents_idx[d] = np.searchsorted(kp, kd>>2).clip(0, len(kp)-1).astype(np.int64)

    neighs = [_build_neighs(keys[d], d) for d in range(MAX_LEVEL+1)]

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

    def _pool_to_nodes(lk_d, lv_d, kd):
        out = np.zeros((len(kd), 1), np.float32)
        if len(lk_d) == 0:
            return out
        arr_k  = np.array(lk_d, np.int64)
        arr_v  = np.array(lv_d, np.float32)
        uk, inv = np.unique(arr_k, return_inverse=True)
        pooled  = np.zeros(len(uk), np.float32)
        cnt     = np.zeros(len(uk), np.float32)
        np.add.at(pooled, inv, arr_v)
        np.add.at(cnt,    inv, 1.0)
        pooled /= cnt.clip(1)
        idx2   = np.searchsorted(kd, uk).clip(0, len(kd)-1)
        found2 = kd[idx2] == uk
        out[idx2[found2], 0] = pooled[found2]
        return out

    values_f = [_pool_to_nodes(lk_by_d[d], lvf_by_d[d], keys[d]) for d in range(MAX_LEVEL+1)]
    values_u = [_pool_to_nodes(lk_by_d[d], lvu_by_d[d], keys[d]) for d in range(MAX_LEVEL+1)]

    bottle_anc = [None]*(MAX_LEVEL+1)
    for d in range(MAX_LEVEL+1):
        Nd = len(keys[d])
        if d <= D_BOTTLE:
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
        'values_f': values_f, 'values_u': values_u,
        'bottle_anc': bottle_anc,
        'k1': float(k1), 'k2': float(k2), 'leaves': leaves,
    }


# ═══════════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING  (NumPy)
# ═══════════════════════════════════════════════════════════════════════════
def _node_pos_enc(keys, depth):
    if len(keys) == 0:
        return np.zeros((0, POS_DIM), np.float32)
    xi, yi = key2xy(keys)
    h      = 1.0 / (1 << depth)
    xc     = (xi.astype(np.float32) + 0.5) * h
    yc     = (yi.astype(np.float32) + 0.5) * h
    dn     = np.full(len(keys), float(depth)/float(MAX_LEVEL), np.float32)
    pos3   = np.stack([xc, yc, dn], 1)
    freqs  = 2.0**np.arange(POS_FREQS, dtype=np.float32)
    x_sin  = pos3[:,:,None] * np.pi * 2.0 * freqs[None,None,:]
    enc    = np.concatenate([np.sin(x_sin), np.cos(x_sin)], -1).reshape(len(keys), -1)
    return np.concatenate([pos3, enc], 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# PADDED ARRAYS  — produces TWO pd dicts (for F and for u) from one tree
# ═══════════════════════════════════════════════════════════════════════════
def topology_to_padded(qt, which='u'):
    """Return padded dict for either F-values (which='f') or u-values (which='u').
    Topology fields are identical across both — only feats/values_gt differ.
    For 'u', values are multiplied by U_SCALE so the AE sees comparable amplitudes
    to F. Decoded u must be divided by U_SCALE to recover the actual solution."""
    if which == 'f':
        vals = qt['values_f']
    else:
        vals = [v * U_SCALE for v in qt['values_u']]
    pd = {k: [] for k in ('feats','pos','neigh','children','parents',
                          'valid','leaf','split_gt','values_gt','bottle_anc')}
    for d in range(MAX_LEVEL+1):
        S  = DEPTH_SIZES[d]
        Nd = len(qt['keys'][d])

        vm = np.zeros(S, bool); vm[:Nd] = True
        pd['valid'].append(jnp.array(vm))

        f = np.zeros((S,1), np.float32); f[:Nd] = vals[d]
        pd['feats'].append(jnp.array(f))

        pos_np = np.zeros((S, POS_DIM), np.float32)
        pos_np[:Nd] = _node_pos_enc(qt['keys'][d], d)
        pd['pos'].append(jnp.array(pos_np))

        nh_raw = qt['neighs'][d]
        nh_np  = np.full((S,9), S, np.int32)
        if Nd > 0:
            nh_np[:Nd] = np.where(nh_raw >= 0, nh_raw, S)
        pd['neigh'].append(jnp.array(nh_np))

        lm = np.zeros(S, bool); lm[:Nd] = qt['leaf_mask'][d]
        pd['leaf'].append(jnp.array(lm))

        vg = np.zeros((S,1), np.float32); vg[:Nd] = vals[d]
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
            pr_np = np.zeros(S, np.int32)
            pr_raw = qt['parents'][d]
            if len(pr_raw) > 0: pr_np[:Nd] = pr_raw[:Nd]
            pd['parents'].append(jnp.array(pr_np))
        else:
            pd['parents'].append(None)
    return pd


# ═══════════════════════════════════════════════════════════════════════════
# FLAX MODEL  (CompressedBottleneckAE — same as quadtree_nmrom_jax.py)
# ═══════════════════════════════════════════════════════════════════════════
class CompressedEncoder(nn.Module):
    hidden:  int = HIDDEN
    emb_dim: int = EMB_DIM

    @nn.compact
    def __call__(self, pd):
        in_proj   = nn.Dense(self.hidden, name='in_proj')
        enc_convs = [nn.Dense(self.hidden, name=f'enc_conv_{d}') for d in range(MAX_LEVEL)]
        emb_projs = [nn.Dense(self.emb_dim, name=f'emb_proj_{d}') for d in range(D_BOTTLE+1)]
        emb_norms = [nn.LayerNorm(name=f'emb_norm_{d}') for d in range(D_BOTTLE+1)]

        h = []
        for d in range(MAX_LEVEL+1):
            x = jnp.concatenate([pd['feats'][d], pd['pos'][d]], axis=1)
            h_d = jax.nn.relu(in_proj(x))
            h_d = h_d * pd['valid'][d][:,None]
            h.append(h_d)

        for d in range(MAX_LEVEL, 0, -1):
            S_p    = DEPTH_SIZES[d-1]
            v      = pd['valid'][d].astype(jnp.float32)
            pooled = jnp.zeros((S_p, self.hidden)).at[pd['parents'][d]].add(h[d] * v[:,None])
            cnt    = jnp.zeros((S_p, 1)).at[pd['parents'][d]].add(v[:,None])
            h[d-1] = h[d-1] + pooled / cnt.clip(1.0)
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

            S      = DEPTH_SIZES[d]
            h_pad  = jnp.concatenate([h, jnp.zeros((1, self.hidden))], 0)
            flat   = h_pad[pd['neigh'][d]].reshape(S, -1)
            h      = jax.nn.relu(mix_convs[d](flat))
            h      = h * pd['valid'][d][:,None]

            val_pred[d] = val_out(jax.nn.relu(val_h(h)))

            if d == MAX_LEVEL:
                break

            split_logits[d] = split_out(jax.nn.relu(split_h(h))).squeeze(-1)
            S_next  = DEPTH_SIZES[d+1]
            cf      = child_head2(jax.nn.relu(child_head1(h)))
            cf      = cf.reshape(DEPTH_SIZES[d], 4, self.hidden)
            ch      = pd['children'][d]
            h_next  = jnp.zeros((S_next, self.hidden))
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
# z <-> E
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
# FVM OPERATOR  (cell-centred Laplacian on adaptive quadtree)
# ═══════════════════════════════════════════════════════════════════════════
def build_fvm_operator(qt):
    global_idx = {}
    leaf_list  = []
    gidx       = 0
    for d in range(MAX_LEVEL+1):
        kd = qt['keys'][d]
        lm = qt['leaf_mask'][d]
        for nd_i, is_leaf in enumerate(lm):
            if is_leaf:
                k       = int(kd[nd_i])
                xi, yi  = key2xy(np.array([k]))
                xi, yi  = int(xi[0]), int(yi[0])
                h       = 1.0 / (1 << d)
                xc, yc  = (xi + 0.5)*h, (yi + 0.5)*h
                global_idx[(d, k)] = gidx
                leaf_list.append((xc, yc, h, d, k, xi, yi))
                gidx += 1

    N = gidx
    # Symmetric FV assembly of  -∇² u = f  in cell-integrated form:
    #   K_ii =  +Σ_face cond     (positive diagonal)
    #   K_ij =  -cond_ij         (negative off-diagonal, symmetric: cond_ij = cond_ji)
    #   rhs  =   f · |V_c|       (volume-scaled, applied per cell below)
    # cond_face = face_length / face_distance.
    # Earlier we divided K by |V_c| to keep RHS as plain f, but on adaptive meshes
    # |V_c| differs across cells so K[c, f] ≠ K[f, c] -> non-symmetric -> CG fails.
    rows_l, cols_l, vals_l = [], [], []
    vol_per_leaf = np.zeros(N, np.float32)
    for gi, (xc, yc, h, d, k, xi, yi) in enumerate(leaf_list):
        vol_per_leaf[gi] = h * h
        diag = 0.0
        res  = 1 << d
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx_, ny_ = xi+dx, yi+dy
            if nx_ < 0 or nx_ >= res or ny_ < 0 or ny_ >= res:
                cond  = h / (h/2)
                diag += cond
                continue
            nk_ = int(xy2key(np.array([nx_]), np.array([ny_]))[0])
            if (d, nk_) in global_idx:
                gj    = global_idx[(d, nk_)]
                cond  = h / h
                rows_l.append(gi); cols_l.append(gj); vals_l.append(-cond)
                diag += cond
                continue
            found = False
            if d > 0:
                nx_p, ny_p = nx_//2, ny_//2
                res_p      = 1 << (d-1)
                if 0 <= nx_p < res_p and 0 <= ny_p < res_p:
                    nk_p = int(xy2key(np.array([nx_p]), np.array([ny_p]))[0])
                    if (d-1, nk_p) in global_idx:
                        gj        = global_idx[(d-1, nk_p)]
                        face_dist = h/2 + h
                        cond      = h / face_dist
                        rows_l.append(gi); cols_l.append(gj); vals_l.append(-cond)
                        diag += cond
                        found = True
            if found:
                continue
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
                            rows_l.append(gi); cols_l.append(gj); vals_l.append(-cond)
                            diag += cond
        rows_l.append(gi); cols_l.append(gi); vals_l.append(+diag)

    rows = jnp.array(rows_l, dtype=jnp.int32)
    cols = jnp.array(cols_l, dtype=jnp.int32)
    vals = jnp.array(vals_l, dtype=jnp.float32)
    vols = jnp.array(vol_per_leaf, jnp.float32)

    k1_, k2_  = qt['k1'], qt['k2']
    xc_arr    = np.array([lf[0] for lf in leaf_list], np.float32)
    yc_arr    = np.array([lf[1] for lf in leaf_list], np.float32)
    # Volume-scaled RHS — paired with the symmetric K above.
    f_vec     = jnp.array(poisson_f(xc_arr, yc_arr, k1_, k2_), jnp.float32) * vols
    u_ref     = jnp.array(poisson_u(xc_arr, yc_arr, k1_, k2_), jnp.float32)

    @jit
    def K_op(u):
        return jnp.zeros(N).at[rows].add(vals * u[cols])

    return K_op, f_vec, u_ref, N, global_idx, leaf_list


def build_leaf_gather(qt, global_idx):
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
    return len(global_idx), pad_idx_per_d, glob_idx_per_d


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
# NM-ROM SOLVER  (GN, Petrov-Galerkin form, depth-scaled preconditioner)
# ═══════════════════════════════════════════════════════════════════════════
def make_nmrom_solver(model, params, pd, qt, K_op, f_vec, global_idx,
                      max_outer=8, max_inner=6, tol=1e-5,
                      residual_weights=None, lam_reg=0.0, z_anchor=None):
    """
    residual_weights: optional (N,) array of per-leaf weights for the residual.
        If None, uniform 1.0 (standard NM-ROM). Setting weights to 0 on coarse
        cells and 1 on fine cells restricts the PDE constraint to fine cells.
        With a non-zero mask but a strong residual gradient at fine cells
        only, the latent z becomes under-determined and GN can drift wildly
        — pair with `lam_reg > 0` to keep z anchored.
    lam_reg: Tikhonov penalty weight on ‖z − z_anchor‖². Adds λ·I to the GN
        Hessian and λ(z − z_anchor) to the gradient. Critical when residual
        is masked.
    z_anchor: the latent to regularize toward (e.g. the MLP-warm-start).
        If None and lam_reg > 0, anchors at z_init in the solve() call.
    """
    P_inv       = 1.0 / DEPTH_SCALE_VEC
    leaf_gather = build_leaf_gather(qt, global_idx)

    if residual_weights is None:
        w = jnp.ones(f_vec.shape[0], jnp.float32)
    else:
        w = jnp.asarray(residual_weights, jnp.float32)
    lam = float(lam_reg)

    def constrained_decode(z):
        # Decoder produces u in U_SCALE-rescaled space; divide to get physical u.
        E = z_to_E(z)
        _, vpred = model.apply({'params': params}, E, pd, method=model.decode)
        return extract_leaf_values(vpred, leaf_gather) / U_SCALE

    constrained_decode_jit = jit(constrained_decode)

    @jit
    def gn_step(z, z_a):
        u_pred, vjp_fn = jax.vjp(constrained_decode, z)
        r              = (K_op(u_pred) - f_vec) * w
        g_red          = vjp_fn(r)[0] + lam * (z - z_a)   # Tikhonov gradient

        def GN_op_tilde(dz_tilde):
            dz       = P_inv * dz_tilde
            _, Jdz   = jax.jvp(constrained_decode, (z,), (dz,))
            KJdz     = K_op(Jdz) * w
            JtKJdz   = vjp_fn(KJdz)[0]
            # Add Tikhonov damping. P_inv·λ·I·P_inv·dz_tilde = λ·P_inv²·dz_tilde
            return P_inv * JtKJdz + lam * P_inv * P_inv * dz_tilde

        rhs         = -(P_inv * g_red)
        dz_tilde, _ = jspla.cg(GN_op_tilde, rhs, tol=1e-4, maxiter=100)
        dz          = P_inv * dz_tilde
        return z + dz, jnp.linalg.norm(r), jnp.linalg.norm(g_red)

    def solve(z_init=None, verbose=False):
        z = jnp.zeros(LATENT_DIM) if z_init is None else z_init
        z_a = z if z_anchor is None else z_anchor
        history = []
        total_iters = max_outer * max_inner
        for it in range(total_iters):
            z, rn, gn = gn_step(z, z_a)
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
# MLP  z_F → z_u
# ═══════════════════════════════════════════════════════════════════════════
class LatentMLP(nn.Module):
    in_dim:    int = LATENT_DIM
    out_dim:   int = LATENT_DIM
    hidden:    int = 512
    n_layers:  int = 3

    @nn.compact
    def __call__(self, z_F):
        h = z_F
        for _ in range(self.n_layers):
            h = nn.Dense(self.hidden)(h)
            h = nn.gelu(h)
        return nn.Dense(self.out_dim)(h)


# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION  — paired (F_pd, u_pd) samples, cached to disk
# ═══════════════════════════════════════════════════════════════════════════
def generate_paired_dataset(n_train, n_val, k_lo=1, k_hi=5, seed=0,
                            grad_thresh=0.5, save_path=None,
                            target='analytical'):
    """k_lo, k_hi are INTEGER bounds. We sample integer (k1, k2) pairs.
    Why integer: poisson_u(x,y; k1,k2) = sin(k1·π·x)sin(k2·π·y)/((k1²+k2²)π²)
    only satisfies u=0 on [0,1]² boundary when k1, k2 are integers.

    target='analytical': store u = closed-form analytical solution at cell centers.
    target='fom':        store u = FVM-discrete solution (one CG solve per sample).
                          This makes the AE's manifold align with what NM-ROM solves for.
    """
    rng = np.random.RandomState(seed)
    train_freqs = rng.randint(k_lo, k_hi + 1, size=(n_train, 2)).astype(np.float32)
    val_freqs   = rng.randint(k_lo, k_hi + 1, size=(n_val,   2)).astype(np.float32)

    def _replace_u_with_fom(qt):
        """Solve K·u = rhs for the FVM-discrete u and overwrite qt['values_u']
        with the FOM solution interpolated to all node centers."""
        K_op, f_vec, _u_ref, N, global_idx, leaf_list = build_fvm_operator(qt)
        sol, _ = jspla.cg(K_op, f_vec, x0=jnp.zeros(N), tol=1e-10, maxiter=20000)
        sol = np.array(sol.block_until_ready())   # (N,) leaf values in global order

        # Build a (depth, morton_key) -> global_index mapping back into per-depth
        # node arrays so we can plug FOM values into qt['values_u'][d][i, 0].
        leaf_at_depth = [{} for _ in range(MAX_LEVEL + 1)]
        for (d, mk), gi in global_idx.items():
            leaf_at_depth[d][int(mk)] = gi

        new_values_u = [np.zeros((len(qt['keys'][d]), 1), np.float32) for d in range(MAX_LEVEL+1)]
        for d in range(MAX_LEVEL + 1):
            kd = qt['keys'][d]; lm = qt['leaf_mask'][d]
            for i, is_leaf in enumerate(lm):
                if is_leaf:
                    gi = leaf_at_depth[d].get(int(kd[i]))
                    if gi is not None:
                        new_values_u[d][i, 0] = sol[gi]
        # Internal-node values: pool from children (mean) so encoder can use them.
        for d in range(MAX_LEVEL - 1, -1, -1):
            ch = qt['children'][d]; kd = qt['keys'][d]
            S_next = DEPTH_SIZES[d + 1]
            for i in range(len(kd)):
                if not qt['leaf_mask'][d][i]:
                    sm = 0.0; cnt = 0
                    for c in range(4):
                        ci = int(ch[i, c])
                        if ci < S_next and ci < len(qt['keys'][d + 1]):
                            sm += float(new_values_u[d + 1][ci, 0]); cnt += 1
                    if cnt > 0:
                        new_values_u[d][i, 0] = sm / cnt
        qt['values_u'] = new_values_u
        return qt

    def build(k_pairs, label):
        out = []
        t0 = time.perf_counter()
        for i, (k1, k2) in enumerate(k_pairs):
            qt = build_quadtree(float(k1), float(k2), grad_thresh)
            if target == 'fom':
                qt = _replace_u_with_fom(qt)
            n_leaves = sum(int(qt['leaf_mask'][d].sum()) for d in range(MAX_LEVEL+1))
            out.append({
                'k1': float(k1), 'k2': float(k2),
                'qt': qt, 'n_leaves': n_leaves,
            })
            if (i+1) % 25 == 0 or (i+1) == len(k_pairs):
                el = time.perf_counter() - t0
                print(f"  [{label}] {i+1}/{len(k_pairs)} | last n_leaves={n_leaves} | {el:.1f}s",
                      flush=True)
        return out

    print(f"Building {n_train} train + {n_val} val paired (F,u) trees... target={target}")
    train_set = build(train_freqs, 'train')
    val_set   = build(val_freqs,   'val')

    if save_path is not None:
        with open(save_path, 'wb') as fh:
            pickle.dump({
                'train': train_set, 'val': val_set,
                'train_freqs': train_freqs, 'val_freqs': val_freqs,
                'config': dict(n_train=n_train, n_val=n_val, k_lo=k_lo, k_hi=k_hi,
                               seed=seed, grad_thresh=grad_thresh,
                               MAX_LEVEL=MAX_LEVEL, MIN_LEVEL=MIN_LEVEL,
                               D_BOTTLE=D_BOTTLE, EMB_DIM=EMB_DIM, HIDDEN=HIDDEN),
            }, fh)
        print(f"  Saved {save_path}  ({os.path.getsize(save_path)/1e6:.1f} MB)")
    return train_set, val_set


# ═══════════════════════════════════════════════════════════════════════════
# AE TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train_shared_ae(train_set, val_set, n_steps=8000, lr=1e-3, seed=42,
                    out_dir=None, save_path=None, eval_every=500,
                    hidden=HIDDEN, emb_dim=EMB_DIM):
    """Train one shared AE on the union of F-views and u-views of the data.
    At each step, sample (i, view) where view ∈ {f, u} and view becomes the
    AE input/target."""
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    sample0 = train_set[0]
    pd0     = topology_to_padded(sample0['qt'], which='u')

    model = CompressedBottleneckAE(hidden=hidden, emb_dim=emb_dim)
    key, sk = jax.random.split(key)
    params  = model.init(sk, pd0)['params']
    n_p     = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  AE params: {n_p:,}  | latent={LATENT_DIM}  | n_steps={n_steps}")

    warmup   = min(500, n_steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0., peak_value=lr, warmup_steps=warmup,
        decay_steps=n_steps, end_value=lr*1e-3,
    )
    tx        = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = tx.init(params)

    def loss_fn(params, pd):
        # MSE over leaf nodes (per depth, averaged across depths). With U_SCALE
        # applied to u, F and u views have comparable amplitudes so MSE works.
        _, val_pred, _ = model.apply({'params': params}, pd)
        L = 0.0; n = 0
        for d in range(MAX_LEVEL+1):
            mask = pd['leaf'][d] & pd['valid'][d]
            n_ok = jnp.sum(mask).clip(1)
            diff2 = jnp.where(mask[:,None], (val_pred[d] - pd['values_gt'][d])**2, 0.0)
            L += jnp.sum(diff2) / n_ok
            n += 1
        return L / n

    @jax.jit
    def train_step(params, opt_state, pd):
        loss, grads = jax.value_and_grad(loss_fn)(params, pd)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def eval_loss(params, dataset, which):
        losses = []
        for sample in dataset:
            pd = topology_to_padded(sample['qt'], which=which)
            l = loss_fn(params, pd)
            losses.append(float(l))
        return float(np.mean(losses))

    def eval_rel_l2(params, dataset, which):
        rels = []
        for sample in dataset:
            pd = topology_to_padded(sample['qt'], which=which)
            _, val_pred, _ = model.apply({'params': params}, pd)
            num, den = 0.0, 0.0
            for d in range(MAX_LEVEL+1):
                mask = pd['leaf'][d] & pd['valid'][d]
                gt   = pd['values_gt'][d][:,0]
                pr   = val_pred[d][:,0]
                num += float(jnp.sum(jnp.where(mask, (pr - gt)**2, 0.0)))
                den += float(jnp.sum(jnp.where(mask, gt**2, 0.0)))
            rels.append(np.sqrt(num / max(den, 1e-30)))
        return float(np.mean(rels))

    best_val = float('inf')
    best_params = params
    history = []
    t0 = time.perf_counter()
    for step in range(n_steps):
        i      = int(rng.integers(0, len(train_set)))
        which  = 'f' if rng.random() < 0.5 else 'u'
        pd     = topology_to_padded(train_set[i]['qt'], which=which)
        params, opt_state, loss = train_step(params, opt_state, pd)

        if (step+1) % eval_every == 0 or step == n_steps - 1:
            vl_f = eval_loss(params, val_set[:20], 'f')
            vl_u = eval_loss(params, val_set[:20], 'u')
            rl_f = eval_rel_l2(params, val_set[:20], 'f')
            rl_u = eval_rel_l2(params, val_set[:20], 'u')
            vl   = 0.5*(vl_f + vl_u)
            print(f"  step {step+1:5d}/{n_steps} | tr-loss {float(loss):.4e} | "
                  f"val-loss F {vl_f:.4e} U {vl_u:.4e} | rel-L2 F {rl_f:.4e} U {rl_u:.4e} | "
                  f"{time.perf_counter()-t0:.0f}s",
                  flush=True)
            history.append({'step': step+1, 'tr_loss': float(loss),
                            'vl_f': vl_f, 'vl_u': vl_u,
                            'rl_f': rl_f, 'rl_u': rl_u})
            if vl < best_val:
                best_val = vl
                best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump({'params': best_params, 'history': history}, f)
        print(f"  Saved AE to {save_path}")

    if out_dir is not None:
        _plot_ae_history(history, out_dir / 'ae_loss.png')

    return model, best_params, history


def _plot_ae_history(history, save_path):
    if not history: return
    steps = [h['step'] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].semilogy(steps, [h['vl_f'] for h in history], '-', label='val MSE F')
    ax[0].semilogy(steps, [h['vl_u'] for h in history], '-', label='val MSE u')
    ax[0].set_xlabel('step'); ax[0].set_ylabel('MSE'); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].semilogy(steps, [h['rl_f'] for h in history], '-', label='val rel-L2 F')
    ax[1].semilogy(steps, [h['rl_u'] for h in history], '-', label='val rel-L2 u')
    ax[1].set_xlabel('step'); ax[1].set_ylabel('rel-L2'); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.suptitle('Shared AE training')
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches='tight'); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MLP TRAINING  z_F -> z_u
# ═══════════════════════════════════════════════════════════════════════════
def encode_pairs(model, params, dataset):
    """Returns (Z_F, Z_u): two stacked (N, LATENT_DIM) arrays."""
    @jax.jit
    def encode_one(pd):
        E = model.apply({'params': params}, pd, method=model.encode)
        return E_to_z(E)

    z_fs, z_us = [], []
    for sample in dataset:
        pd_f = topology_to_padded(sample['qt'], which='f')
        pd_u = topology_to_padded(sample['qt'], which='u')
        z_fs.append(np.array(encode_one(pd_f)))
        z_us.append(np.array(encode_one(pd_u)))
    return np.stack(z_fs), np.stack(z_us)


def train_mlp(Z_F_train, Z_u_train, Z_F_val, Z_u_val,
              hidden=512, n_layers=3, n_steps=5000, batch_size=64,
              lr=1e-3, seed=0, save_path=None):
    key = jax.random.PRNGKey(seed)
    mlp = LatentMLP(hidden=hidden, n_layers=n_layers)
    key, sk = jax.random.split(key)
    params  = mlp.init(sk, jnp.zeros((1, LATENT_DIM)))['params']
    n_p     = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  MLP params: {n_p:,} | hidden={hidden} | layers={n_layers}")

    schedule  = optax.warmup_cosine_decay_schedule(0., lr, 200, n_steps, lr*1e-3)
    tx        = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params, zf, zu):
        pred = mlp.apply({'params': params}, zf)
        return jnp.mean((pred - zu)**2)

    @jax.jit
    def train_step(params, opt_state, zf, zu):
        loss, grads = jax.value_and_grad(loss_fn)(params, zf, zu)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def latent_rel(params, zf, zu):
        pred = mlp.apply({'params': params}, zf)
        num = jnp.linalg.norm(pred - zu, axis=1)
        den = jnp.linalg.norm(zu, axis=1) + 1e-12
        return jnp.mean(num / den)

    rng = np.random.default_rng(seed+1)
    Z_F_train_j = jnp.asarray(Z_F_train)
    Z_u_train_j = jnp.asarray(Z_u_train)
    Z_F_val_j   = jnp.asarray(Z_F_val)
    Z_u_val_j   = jnp.asarray(Z_u_val)

    best_val = float('inf')
    best_params = params
    history = []
    t0 = time.perf_counter()
    for step in range(n_steps):
        idx = rng.integers(0, len(Z_F_train), size=batch_size)
        zf = Z_F_train_j[idx]; zu = Z_u_train_j[idx]
        params, opt_state, loss = train_step(params, opt_state, zf, zu)

        if (step+1) % 250 == 0 or step == n_steps - 1:
            vl   = float(loss_fn(params, Z_F_val_j, Z_u_val_j))
            rel  = float(latent_rel(params, Z_F_val_j, Z_u_val_j))
            print(f"  mlp step {step+1:5d}/{n_steps} | tr-loss {float(loss):.4e} | "
                  f"val MSE {vl:.4e} | val latent-rel {rel:.4e} | "
                  f"{time.perf_counter()-t0:.0f}s",
                  flush=True)
            history.append({'step': step+1, 'tr_loss': float(loss),
                            'val_mse': vl, 'val_rel': rel})
            if vl < best_val:
                best_val = vl
                best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump({'params': best_params, 'history': history,
                         'hidden': hidden, 'n_layers': n_layers}, f)
        print(f"  Saved MLP to {save_path}")

    return mlp, best_params, history


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK  —  cold-start vs MLP-warm-start vs FOM
# ═══════════════════════════════════════════════════════════════════════════
def benchmark(ae_model, ae_params, mlp_model, mlp_params, val_set,
              n_test=6, max_outer=8, max_inner=6, tol=1e-5,
              out_dir=None, log_path=None):
    test_set = val_set[:n_test]

    @jax.jit
    def encode_F(pd_f):
        E = ae_model.apply({'params': ae_params}, pd_f, method=ae_model.encode)
        return E_to_z(E)

    @jax.jit
    def mlp_predict(z_F):
        return mlp_model.apply({'params': mlp_params}, z_F)

    rows = []
    for i, sample in enumerate(test_set):
        k1, k2 = sample['k1'], sample['k2']
        qt = sample['qt']
        pd_f = topology_to_padded(qt, which='f')
        pd_u = topology_to_padded(qt, which='u')
        K_op, f_vec, u_ref, N, global_idx, leaf_list = build_fvm_operator(qt)

        # Per-leaf depth + fine-cell mask. Fine cells = leaves at the deepest
        # level present in this tree (depth = max_depth_present, here MAX_LEVEL
        # for adaptive trees with any depth-MAX_LEVEL refinement).
        leaf_depths = np.array([lf[3] for lf in leaf_list], np.int32)
        max_d_present = int(leaf_depths.max())
        fine_mask  = (leaf_depths == max_d_present).astype(np.float32)
        coarse_mask = 1.0 - fine_mask
        n_fine   = int(fine_mask.sum()); n_coarse = N - n_fine
        print(f"\nCase {i+1}: k=({k1:.3f}, {k2:.3f})  N_leaves={N}  "
              f"fine={n_fine}  coarse={n_coarse}", flush=True)

        # Cold-start solver (full residual)
        solve_cold, _ = make_nmrom_solver(ae_model, ae_params, pd_f, qt, K_op, f_vec,
                                          global_idx, max_outer=max_outer,
                                          max_inner=max_inner, tol=tol)
        _ = solve_cold(z_init=jnp.zeros(LATENT_DIM))
        t0 = time.perf_counter()
        z_cold, u_cold, hist_cold = solve_cold(z_init=jnp.zeros(LATENT_DIM), verbose=False)
        u_cold.block_until_ready()
        t_cold = time.perf_counter() - t0

        # MLP warm-start
        z_F0  = encode_F(pd_f)
        z_u0  = mlp_predict(z_F0)
        leaf_gather = build_leaf_gather(qt, global_idx)
        @jax.jit
        def decode_z(z):
            E = z_to_E(z)
            _, vpred = ae_model.apply({'params': ae_params}, E, pd_f, method=ae_model.decode)
            return extract_leaf_values(vpred, leaf_gather) / U_SCALE
        u_mlp_only = decode_z(z_u0)

        # WARM (full residual) and FINE (residual restricted to fine leaves)
        solve_warm, _ = make_nmrom_solver(ae_model, ae_params, pd_f, qt, K_op, f_vec,
                                          global_idx, max_outer=max_outer,
                                          max_inner=max_inner, tol=tol)
        _ = solve_warm(z_init=z_u0)
        t0 = time.perf_counter()
        z_warm, u_warm, hist_warm = solve_warm(z_init=z_u0, verbose=False)
        u_warm.block_until_ready()
        t_warm = time.perf_counter() - t0

        # Fine-cell solver: residual masked to fine leaves, with Tikhonov
        # regularization toward the MLP-warm-start latent so the under-determined
        # masked problem can't drift z far from the MLP guess.
        solve_fine, _ = make_nmrom_solver(ae_model, ae_params, pd_f, qt, K_op, f_vec,
                                          global_idx, max_outer=max_outer,
                                          max_inner=max_inner, tol=tol,
                                          residual_weights=fine_mask,
                                          lam_reg=1e-3, z_anchor=z_u0)
        _ = solve_fine(z_init=z_u0)
        t0 = time.perf_counter()
        z_fine, u_fine, hist_fine = solve_fine(z_init=z_u0, verbose=False)
        u_fine.block_until_ready()
        t_fine = time.perf_counter() - t0

        # FOM (CG on the FVM operator)
        @jit
        def cg_solve(b):
            sol, _ = jspla.cg(K_op, b, x0=jnp.zeros_like(b), tol=1e-8, maxiter=2000)
            return sol
        _ = cg_solve(f_vec)
        t0 = time.perf_counter()
        u_fom = cg_solve(f_vec)
        u_fom.block_until_ready()
        t_fom = time.perf_counter() - t0

        # Metrics: rel-L2 against analytical, against FOM-discrete, plus fine-only
        # rel-L2 (the metric this iteration is targeting).
        u_ref_np  = np.array(u_ref)
        u_fom_np  = np.array(u_fom)
        fmask_np  = fine_mask
        cmask_np  = coarse_mask

        def rel(u_arr, ref, mask=None):
            ua = np.array(u_arr)
            if mask is None:
                num = np.linalg.norm(ua - ref); den = np.linalg.norm(ref)
            else:
                w_ = mask
                num = np.sqrt(np.sum(w_ * (ua - ref)**2))
                den = np.sqrt(np.sum(w_ * ref**2))
            return float(num / max(den, 1e-30))

        # vs analytical (full / fine / coarse)
        rel_cold     = rel(u_cold,     u_ref_np)
        rel_warm     = rel(u_warm,     u_ref_np)
        rel_fine     = rel(u_fine,     u_ref_np)
        rel_mlp_only = rel(u_mlp_only, u_ref_np)
        rel_fom      = rel(u_fom,      u_ref_np)

        rel_warm_fineonly = rel(u_warm,     u_ref_np, fmask_np)
        rel_fine_fineonly = rel(u_fine,     u_ref_np, fmask_np)
        rel_mlp_fineonly  = rel(u_mlp_only, u_ref_np, fmask_np)
        rel_fom_fineonly  = rel(u_fom,      u_ref_np, fmask_np)

        rel_warm_coarse = rel(u_warm,     u_ref_np, cmask_np) if n_coarse else 0.0
        rel_fine_coarse = rel(u_fine,     u_ref_np, cmask_np) if n_coarse else 0.0
        rel_mlp_coarse  = rel(u_mlp_only, u_ref_np, cmask_np) if n_coarse else 0.0

        # vs FOM-discrete
        rel_cold_fom = rel(u_cold,     u_fom_np)
        rel_warm_fom = rel(u_warm,     u_fom_np)
        rel_fine_fom = rel(u_fine,     u_fom_np)
        rel_mlp_fom  = rel(u_mlp_only, u_fom_np)

        row = dict(
            i=i+1, k1=k1, k2=k2, N=N, n_fine=n_fine, n_coarse=n_coarse,
            rel_cold=rel_cold, rel_warm=rel_warm, rel_fine=rel_fine,
            rel_mlp=rel_mlp_only, rel_fom=rel_fom,
            rel_warm_fineonly=rel_warm_fineonly, rel_fine_fineonly=rel_fine_fineonly,
            rel_mlp_fineonly=rel_mlp_fineonly,   rel_fom_fineonly=rel_fom_fineonly,
            rel_warm_coarse=rel_warm_coarse, rel_fine_coarse=rel_fine_coarse,
            rel_mlp_coarse=rel_mlp_coarse,
            rel_cold_fom=rel_cold_fom, rel_warm_fom=rel_warm_fom,
            rel_fine_fom=rel_fine_fom, rel_mlp_fom=rel_mlp_fom,
            t_cold=t_cold, t_warm=t_warm, t_fine=t_fine, t_fom=t_fom,
            iters_cold=len(hist_cold), iters_warm=len(hist_warm),
            iters_fine=len(hist_fine),
        )
        rows.append(row)
        print(f"  COLD       : rel-L2 vs ana={rel_cold:.4e}  vs FOM={rel_cold_fom:.4e}  iters={len(hist_cold):2d}  time={t_cold:.2f}s")
        print(f"  WARM full  : rel-L2 vs ana={rel_warm:.4e}  vs FOM={rel_warm_fom:.4e}  iters={len(hist_warm):2d}  time={t_warm:.2f}s")
        print(f"  WARM fine  : rel-L2 vs ana={rel_fine:.4e}  vs FOM={rel_fine_fom:.4e}  iters={len(hist_fine):2d}  time={t_fine:.2f}s")
        print(f"  MLP only   : rel-L2 vs ana={rel_mlp_only:.4e}  vs FOM={rel_mlp_fom:.4e}")
        print(f"  FOM        : rel-L2 vs ana={rel_fom:.4e}  time={t_fom:.3f}s")
        print(f"  fine cells (where the FINE solver puts effort):")
        print(f"    rel-L2 (fine-only)  WARM={rel_warm_fineonly:.4e}  FINE={rel_fine_fineonly:.4e}  "
              f"MLP={rel_mlp_fineonly:.4e}  FOM={rel_fom_fineonly:.4e}")
        print(f"  coarse cells (left to MLP):")
        print(f"    rel-L2 (coarse-only) WARM={rel_warm_coarse:.4e}  FINE={rel_fine_coarse:.4e}  "
              f"MLP={rel_mlp_coarse:.4e}")

        if out_dir is not None:
            _plot_case_v2(leaf_list, np.array(u_ref), np.array(u_warm),
                          np.array(u_fine), np.array(u_mlp_only), fmask_np,
                          title=(f"k=({k1:.0f},{k2:.0f})  N={N}  fine={n_fine}  "
                                 f"rel-L2: warm={rel_warm:.2e} fine={rel_fine:.2e} "
                                 f"mlp={rel_mlp_only:.2e}\n"
                                 f"fine-only rel-L2: warm={rel_warm_fineonly:.2e} "
                                 f"fine={rel_fine_fineonly:.2e} mlp={rel_mlp_fineonly:.2e}"),
                          save_path=out_dir / f'case_{i+1:02d}.png')

    keys_avg = ('rel_cold','rel_warm','rel_fine','rel_mlp','rel_fom',
                'rel_warm_fineonly','rel_fine_fineonly','rel_mlp_fineonly','rel_fom_fineonly',
                'rel_warm_coarse','rel_fine_coarse','rel_mlp_coarse',
                'rel_cold_fom','rel_warm_fom','rel_fine_fom','rel_mlp_fom',
                't_cold','t_warm','t_fine','t_fom',
                'iters_cold','iters_warm','iters_fine')
    avg = {f'avg_{k}': float(np.mean([r[k] for r in rows])) for k in keys_avg}
    print(f"\n=== SUMMARY ({n_test} cases) ===")
    print(f"  Full-mesh rel-L2 vs analytical:")
    print(f"    COLD={avg['avg_rel_cold']:.4e}  WARM={avg['avg_rel_warm']:.4e}  "
          f"FINE={avg['avg_rel_fine']:.4e}  MLP={avg['avg_rel_mlp']:.4e}  "
          f"FOM={avg['avg_rel_fom']:.4e}")
    print(f"  Fine-cells-only rel-L2 vs analytical (the metric that matters):")
    print(f"    WARM={avg['avg_rel_warm_fineonly']:.4e}  FINE={avg['avg_rel_fine_fineonly']:.4e}  "
          f"MLP={avg['avg_rel_mlp_fineonly']:.4e}  FOM={avg['avg_rel_fom_fineonly']:.4e}")
    print(f"  Coarse-cells-only rel-L2 vs analytical:")
    print(f"    WARM={avg['avg_rel_warm_coarse']:.4e}  FINE={avg['avg_rel_fine_coarse']:.4e}  "
          f"MLP={avg['avg_rel_mlp_coarse']:.4e}")
    print(f"  Time:  COLD={avg['avg_t_cold']:.2f}s  WARM={avg['avg_t_warm']:.2f}s  "
          f"FINE={avg['avg_t_fine']:.2f}s  FOM={avg['avg_t_fom']:.3f}s")
    print(f"  Iters: COLD={avg['avg_iters_cold']:.1f}  WARM={avg['avg_iters_warm']:.1f}  "
          f"FINE={avg['avg_iters_fine']:.1f}")

    if log_path is not None:
        cols = ('i','k1','k2','N','n_fine','n_coarse',
                'rel_cold','rel_warm','rel_fine','rel_mlp','rel_fom',
                'rel_warm_fineonly','rel_fine_fineonly','rel_mlp_fineonly','rel_fom_fineonly',
                'rel_warm_coarse','rel_fine_coarse','rel_mlp_coarse',
                'rel_cold_fom','rel_warm_fom','rel_fine_fom','rel_mlp_fom',
                't_cold','t_warm','t_fine','t_fom',
                'iters_cold','iters_warm','iters_fine')
        with open(log_path, 'a') as f:
            for r in rows:
                f.write('\t'.join(str(r[k]) for k in cols) + '\n')
            f.write('AVG\t-\t-\t-\t-\t-\t' +
                    '\t'.join(f"{avg[f'avg_{k}']:.4e}" for k in cols[6:]) + '\n\n')

    return rows, avg


def _plot_case(qt, leaf_list, u_ref, u_warm, u_cold, u_mlp, title, save_path):
    # 4-panel: u_ref | u_warm | u_cold | u_mlp_only
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [(u_ref, 'u_ref'), (u_warm, 'NM-ROM warm'), (u_cold, 'NM-ROM cold'),
              (u_mlp, 'MLP only')]
    vmin = min(u_ref.min(), u_warm.min())
    vmax = max(u_ref.max(), u_warm.max())
    for ax, (u, name) in zip(axes, panels):
        for li, (xc, yc, h, _, _, _, _) in enumerate(leaf_list):
            rect = patches.Rectangle((xc - h/2, yc - h/2), h, h,
                                     linewidth=0, facecolor=plt.cm.viridis(
                                         (u[li] - vmin) / (vmax - vmin + 1e-30)))
            ax.add_patch(rect)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_title(name)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight'); plt.close()


def _plot_case_v2(leaf_list, u_ref, u_warm, u_fine, u_mlp, fine_mask,
                  title, save_path):
    """6-panel: u_ref, MLP, NM-ROM warm (full residual), NM-ROM fine (fine-only),
    error of fine vs analytical, and the mesh visualisation (fine in white,
    coarse darker) so the structure is visible at a glance."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    vmin = float(u_ref.min()); vmax = float(u_ref.max())
    err  = np.abs(u_fine - u_ref); e_max = float(err.max())

    def draw(ax, vals, name, lo, hi, cmap='RdBu_r'):
        cmap_obj = plt.get_cmap(cmap)
        for li, (xc, yc, h, *_) in enumerate(leaf_list):
            t = (vals[li] - lo) / max(hi - lo, 1e-30)
            ax.add_patch(patches.Rectangle((xc - h/2, yc - h/2), h, h,
                                            facecolor=cmap_obj(t), edgecolor='none'))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    draw(axes[0,0], u_ref, 'analytical u', vmin, vmax)
    draw(axes[0,1], u_mlp, 'MLP only', vmin, vmax)
    draw(axes[0,2], u_warm, 'NM-ROM (full residual)', vmin, vmax)

    draw(axes[1,0], u_fine, 'NM-ROM (fine-cells-only)', vmin, vmax)
    draw(axes[1,1], err,    f'|fine-NMROM - analytical|  max={e_max:.2e}',
         0, e_max, cmap='hot')

    # Mesh structure: fine = light, coarse = dark. Just to remind which
    # cells are getting the extra refinement.
    ax = axes[1,2]
    for li, (xc, yc, h, *_) in enumerate(leaf_list):
        c = '#fdfd9b' if fine_mask[li] > 0.5 else '#444466'
        ax.add_patch(patches.Rectangle((xc - h/2, yc - h/2), h, h,
                                        facecolor=c, edgecolor='black',
                                        linewidth=0.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('mesh: yellow = fine cells (NM-ROM target)\nblue = coarse cells (MLP)', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight'); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['data','train_ae','train_mlp','bench','all'])
    p.add_argument('--out',           type=str,   default='f_to_u_runs/exp1')
    p.add_argument('--n-train',       type=int,   default=200)
    p.add_argument('--n-val',         type=int,   default=40)
    p.add_argument('--k-lo',          type=int,   default=1)
    p.add_argument('--k-hi',          type=int,   default=5)
    p.add_argument('--target',        type=str,   default='analytical',
                   choices=['analytical', 'fom'],
                   help='What to use as the u-side training target: closed-form '
                        'analytical solution, or one-time FVM CG solve per sample.')
    p.add_argument('--grad-thresh',   type=float, default=0.5)
    p.add_argument('--seed',          type=int,   default=0)
    p.add_argument('--ae-steps',      type=int,   default=8000)
    p.add_argument('--ae-lr',         type=float, default=1e-3)
    p.add_argument('--hidden',        type=int,   default=HIDDEN,
                   help='AE hidden width (encoder/decoder feature dim)')
    p.add_argument('--emb-dim',       type=int,   default=EMB_DIM,
                   help='per-node bottleneck embedding dim')
    p.add_argument('--mlp-steps',     type=int,   default=5000)
    p.add_argument('--mlp-hidden',    type=int,   default=512)
    p.add_argument('--mlp-layers',    type=int,   default=3)
    p.add_argument('--mlp-lr',        type=float, default=1e-3)
    p.add_argument('--max-outer',     type=int,   default=8)
    p.add_argument('--max-inner',     type=int,   default=6)
    p.add_argument('--gn-tol',        type=float, default=1e-5)
    p.add_argument('--n-test',        type=int,   default=6)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== f_to_u_nmrom: {args.command} | out={out_dir} ===")
    print(f"JAX devices: {jax.devices()}")

    data_path = out_dir / 'paired_dataset.pkl'
    ae_path   = out_dir / 'shared_ae.pkl'
    mlp_path  = out_dir / 'mlp_z_f_to_z_u.pkl'
    bench_log = out_dir / 'benchmark_results.tsv'

    if args.command in ('data', 'all'):
        if data_path.exists():
            print(f"Found existing dataset at {data_path} — skipping data gen.")
        else:
            generate_paired_dataset(args.n_train, args.n_val,
                                    args.k_lo, args.k_hi, args.seed,
                                    args.grad_thresh, save_path=data_path,
                                    target=args.target)

    if args.command in ('train_ae', 'all'):
        with open(data_path, 'rb') as f:
            d = pickle.load(f)
        train_set, val_set = d['train'], d['val']
        train_shared_ae(train_set, val_set, n_steps=args.ae_steps, lr=args.ae_lr,
                        seed=args.seed, out_dir=out_dir, save_path=ae_path,
                        hidden=args.hidden, emb_dim=args.emb_dim)

    if args.command in ('train_mlp', 'all'):
        with open(data_path, 'rb') as f:
            d = pickle.load(f)
        train_set, val_set = d['train'], d['val']
        with open(ae_path, 'rb') as f:
            ae_ck = pickle.load(f)
        ae_model = CompressedBottleneckAE(hidden=HIDDEN, emb_dim=EMB_DIM)
        ae_params = ae_ck['params']
        print('Encoding train pairs...')
        Z_F_tr, Z_u_tr = encode_pairs(ae_model, ae_params, train_set)
        print('Encoding val pairs...')
        Z_F_va, Z_u_va = encode_pairs(ae_model, ae_params, val_set)
        # Quick latent stats
        print(f"  ‖Z_F_tr‖ mean={np.linalg.norm(Z_F_tr,axis=1).mean():.3e} "
              f"‖Z_u_tr‖ mean={np.linalg.norm(Z_u_tr,axis=1).mean():.3e}")
        train_mlp(Z_F_tr, Z_u_tr, Z_F_va, Z_u_va,
                  hidden=args.mlp_hidden, n_layers=args.mlp_layers,
                  n_steps=args.mlp_steps, lr=args.mlp_lr,
                  seed=args.seed, save_path=mlp_path)

    if args.command in ('bench', 'all'):
        with open(data_path, 'rb') as f:
            d = pickle.load(f)
        with open(ae_path, 'rb') as f:
            ae_ck = pickle.load(f)
        with open(mlp_path, 'rb') as f:
            mlp_ck = pickle.load(f)
        ae_model  = CompressedBottleneckAE(hidden=HIDDEN, emb_dim=EMB_DIM)
        ae_params = ae_ck['params']
        mlp_model = LatentMLP(hidden=mlp_ck.get('hidden', 512),
                              n_layers=mlp_ck.get('n_layers', 3))
        mlp_params = mlp_ck['params']
        bench_dir = out_dir / 'plots_bench'; bench_dir.mkdir(exist_ok=True)
        if not bench_log.exists():
            with open(bench_log, 'w') as f:
                f.write('i\tk1\tk2\tN\tn_fine\tn_coarse\t'
                        'rel_cold\trel_warm\trel_fine\trel_mlp\trel_fom\t'
                        'rel_warm_fineonly\trel_fine_fineonly\trel_mlp_fineonly\trel_fom_fineonly\t'
                        'rel_warm_coarse\trel_fine_coarse\trel_mlp_coarse\t'
                        'rel_cold_fom\trel_warm_fom\trel_fine_fom\trel_mlp_fom\t'
                        't_cold\tt_warm\tt_fine\tt_fom\t'
                        'iters_cold\titers_warm\titers_fine\n')
        benchmark(ae_model, ae_params, mlp_model, mlp_params, d['val'],
                  n_test=args.n_test, max_outer=args.max_outer,
                  max_inner=args.max_inner, tol=args.gn_tol,
                  out_dir=bench_dir, log_path=bench_log)


if __name__ == '__main__':
    main()
