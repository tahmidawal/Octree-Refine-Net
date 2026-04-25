"""
quadtree_nmrom.py  (v2 — padded arrays for JIT)
-------------------------------------------------
Quadtree Autoencoder NM-ROM for 2D Poisson.

Key design: all arrays padded to fixed max sizes per depth so JAX compiles
the training step ONCE. valid_mask tracks real vs padding entries.

Architecture:
  - Morton2D utilities (JAX)
  - QuadConv (9-neighbor), QuadPool (child->parent)
  - CompressedEncoder: bottom-up, outputs E[0..D_BOTTLE]
  - CompressedDecoder: top-down with ancestor-indexed skip connections
  - 2D Poisson FD operator on adaptive quadtree
  - NM-ROM Gauss-Newton solver in latent space

PDE: -nabla^2 u = f,  f(x,y) = sin(2*pi*k1*x) * sin(2*pi*k2*y)
     u_exact = f / (4*pi^2*(k1^2 + k2^2))  with u=0 on boundary
"""

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import flax.linen as nn
import optax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import pickle
from pathlib import Path
from functools import partial

# ==========================================
# 0. Logging + Config
# ==========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
(SCRIPT_DIR / 'plots').mkdir(parents=True, exist_ok=True)

LOG_FILE = SCRIPT_DIR / 'quadtree_nmrom_training.log'

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(LOG_FILE)

# Hyperparameters
MAX_DEPTH    = 7
MIN_LEVEL    = 6
D_BOTTLE     = 3
EMB_DIM      = 6
HIDDEN       = 128
POS_FREQS    = 6
POS_DIM      = 3 + 2 * POS_FREQS * 3  # 39

# Fixed max sizes per depth (complete quadtree)
MAX_NODES = tuple(4**d for d in range(MAX_DEPTH + 1))
# (1, 4, 16, 64, 256, 1024, 4096, 16384)

# Training
PHASE1_STEPS = 6000
PHASE2_STEPS = 4000
LR_PHASE1    = 1e-3
LR_PHASE2    = 3e-4
SPLIT_WEIGHT = 0.5
PATIENCE     = 500

# NM-ROM
K_DIM        = EMB_DIM * sum(4**d for d in range(D_BOTTLE + 1))  # 510
GN_MAX_ITERS = 12
GN_TOL       = 1e-6

print(f"=" * 70)
print(f"Quadtree NM-ROM for 2D Poisson")
print(f"D_BOTTLE={D_BOTTLE} | EMB_DIM={EMB_DIM} | HIDDEN={HIDDEN}")
print(f"Latent dim: {K_DIM} floats")
print(f"JAX devices: {jax.devices()}")
print(f"=" * 70)


# ==========================================
# 1. Morton Code Utilities
# ==========================================

def _interleave_np(x):
    x = np.asarray(x, dtype=np.int32)
    x = x & 0x0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x

def _deinterleave_np(x):
    x = np.asarray(x, dtype=np.int32)
    x = x & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF
    return x

def morton_xy2key_np(x, y):
    return (_interleave_np(x) | (_interleave_np(y) << 1)).astype(np.int32)

def morton_key2xy_np(key):
    return _deinterleave_np(key), _deinterleave_np(key >> 1)

def _interleave_jax(x):
    x = x.astype(jnp.int32) & 0x0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x

def _deinterleave_jax(x):
    x = x.astype(jnp.int32) & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF
    return x

def morton_key2xy_jax(key):
    return _deinterleave_jax(key), _deinterleave_jax(key >> 1)


# ==========================================
# 2. Positional Encoding (JAX)
# ==========================================

def node_centers_jax(keys, depth, max_depth=MAX_DEPTH):
    x, y = morton_key2xy_jax(keys)
    res = float(1 << depth)
    xf = (x.astype(jnp.float32) + 0.5) / res
    yf = (y.astype(jnp.float32) + 0.5) / res
    dnorm = jnp.full_like(xf, float(depth) / float(max_depth))
    return jnp.stack([xf, yf, dnorm], axis=-1)

def fourier_encode(pos, num_freqs=POS_FREQS):
    freqs = (2.0 ** jnp.arange(num_freqs, dtype=jnp.float32)).reshape(1, 1, -1)
    x = pos[..., None] * jnp.pi * 2.0 * freqs
    enc = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1).reshape(pos.shape[0], -1)
    return jnp.concatenate([pos, enc], axis=-1)


# ==========================================
# 3. Data Generation (NumPy, offline)
# ==========================================

def analytical_solution(x, y, k1, k2):
    return np.sin(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y) / (4*np.pi**2*(k1**2 + k2**2))

def forcing_gradient_magnitude(x, y, k1, k2):
    dfdx = 2*np.pi*k1 * np.cos(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y)
    dfdy = 2*np.pi*k2 * np.sin(2*np.pi*k1*x) * np.cos(2*np.pi*k2*y)
    return np.sqrt(dfdx**2 + dfdy**2)


class QuadNode:
    __slots__ = ['x', 'y', 'size', 'level', 'max_level', 'min_level', 'children', 'val']
    def __init__(self, x, y, size, level, max_level, min_level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.max_level, self.min_level = max_level, min_level
        self.children = []
        self.val = None

    def subdivide(self, k1, k2, grad_threshold=0.3):
        if self.level >= self.max_level:
            return
        cx, cy = self.x + self.size/2, self.y + self.size/2
        force_split = self.level < self.min_level
        grad_mag = forcing_gradient_magnitude(cx, cy, k1, k2)
        max_grad = 2*np.pi*np.sqrt(k1**2 + k2**2)
        normalized = grad_mag / max_grad
        depth_factor = 1.0 + 0.1 * self.level
        if force_split or (normalized > grad_threshold * depth_factor):
            half = self.size / 2
            self.children = [
                QuadNode(self.x,        self.y,        half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y,        half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x,        self.y + half, half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y + half, half, self.level+1, self.max_level, self.min_level),
            ]
            for c in self.children:
                c.subdivide(k1, k2, grad_threshold)

    def collect_leaves(self, leaves_list, k1, k2):
        if not self.children:
            cx, cy = self.x + self.size/2, self.y + self.size/2
            self.val = analytical_solution(cx, cy, k1, k2)
            leaves_list.append(self)
        else:
            for c in self.children:
                c.collect_leaves(leaves_list, k1, k2)


def generate_sample(k1, k2, noise_std=0.0):
    """Generate one quadtree sample. Returns leaf keys/vals by depth."""
    root = QuadNode(0, 0, 1.0, 0, MAX_DEPTH, MIN_LEVEL)
    root.subdivide(k1, k2)
    leaves = []
    root.collect_leaves(leaves, k1, k2)

    leaf_keys = [[] for _ in range(MAX_DEPTH + 1)]
    leaf_vals = [[] for _ in range(MAX_DEPTH + 1)]
    for node in leaves:
        d = node.level
        res = 1 << d
        ix = max(0, min(res-1, int(node.x * res)))
        iy = max(0, min(res-1, int(node.y * res)))
        leaf_keys[d].append(morton_xy2key_np(ix, iy))
        leaf_vals[d].append(float(node.val))

    keys_out, vals_out = [], []
    for d in range(MAX_DEPTH + 1):
        if leaf_keys[d]:
            keys_out.append(np.array(leaf_keys[d], dtype=np.int32))
            v = np.array(leaf_vals[d], dtype=np.float32)
            if noise_std > 0:
                v = v + noise_std * np.random.randn(len(v)).astype(np.float32)
            vals_out.append(v)
        else:
            keys_out.append(np.empty(0, dtype=np.int32))
            vals_out.append(np.empty(0, dtype=np.float32))

    return keys_out, vals_out, leaves, k1, k2


# ==========================================
# 4. Build PADDED Quadtree Arrays
# ==========================================

def build_padded_tree(leaf_keys_by_depth, leaf_vals_by_depth):
    """
    Build quadtree and pad all arrays to fixed MAX_NODES[d] per depth.
    This ensures JAX JIT compiles once and reuses for all samples.

    Returns dict with:
      keys[d]: (MAX_NODES[d],) int32, padded with -1
      features[d]: (MAX_NODES[d], 1) float32
      neighs[d]: (MAX_NODES[d], 9) int32, padded with -1
      children_idx[d]: (MAX_NODES[d], 4) int32, padded with -1
      valid[d]: (MAX_NODES[d],) bool
      leaf_mask[d]: (MAX_NODES[d],) bool
      split_gt[d]: (MAX_NODES[d],) float32
      values_gt[d]: (MAX_NODES[d], 1) float32
    """
    max_depth = MAX_DEPTH

    # Step 1: build unpadded tree structure (numpy)
    keys_raw = [None] * (max_depth + 1)
    for d in range(max_depth + 1):
        lk = leaf_keys_by_depth[d]
        if len(lk) > 0:
            keys_raw[d] = np.unique(lk)
        else:
            keys_raw[d] = np.empty(0, dtype=np.int32)

    # Ensure root
    if len(keys_raw[0]) == 0:
        keys_raw[0] = np.array([0], dtype=np.int32)

    # Ancestor closure
    for d in range(max_depth, 0, -1):
        if len(keys_raw[d]) == 0:
            continue
        parents = np.unique(keys_raw[d] >> 2)
        keys_raw[d-1] = np.unique(np.concatenate([keys_raw[d-1], parents]))

    # Features from leaf values
    features_raw = [np.zeros((len(keys_raw[d]), 1), dtype=np.float32) for d in range(max_depth+1)]
    for d in range(max_depth + 1):
        lk = leaf_keys_by_depth[d]
        lv = leaf_vals_by_depth[d]
        if len(lk) > 0:
            kd = keys_raw[d]
            for i in range(len(lk)):
                idx = np.searchsorted(kd, lk[i])
                if idx < len(kd) and kd[idx] == lk[i]:
                    features_raw[d][idx, 0] = lv[i]

    # Children indices
    children_raw = [None] * (max_depth + 1)
    for d in range(max_depth):
        kd = keys_raw[d]
        kn = keys_raw[d+1]
        if len(kd) == 0 or len(kn) == 0:
            children_raw[d] = np.full((len(kd), 4), -1, dtype=np.int32)
            continue
        ck = (kd[:, None].astype(np.int64) << 2) + np.arange(4)[None, :]
        ck = ck.astype(np.int32)
        idx = np.searchsorted(kn, ck.ravel()).reshape(ck.shape).clip(0, max(len(kn)-1, 0))
        children_raw[d] = np.where(kn[idx] == ck, idx, -1).astype(np.int32)
    children_raw[max_depth] = np.full((len(keys_raw[max_depth]), 4), -1, dtype=np.int32)

    # Neighbors (9-point stencil)
    offsets = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]], dtype=np.int32)
    neighs_raw = [None] * (max_depth + 1)
    for d in range(max_depth + 1):
        kd = keys_raw[d]
        N = len(kd)
        if N == 0:
            neighs_raw[d] = np.empty((0, 9), dtype=np.int32)
            continue
        if d == 0:
            n = np.full((N, 9), -1, dtype=np.int32)
            n[:, 4] = 0
            neighs_raw[d] = n
            continue
        x, y = morton_key2xy_np(kd)
        res = 1 << d
        nx = x[:, None] + offsets[None, :, 0]
        ny = y[:, None] + offsets[None, :, 1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
        nk = np.where(valid, morton_xy2key_np(np.clip(nx, 0, res-1), np.clip(ny, 0, res-1)), -1)
        idx = np.searchsorted(kd, np.maximum(nk, 0).ravel()).reshape(N, 9).clip(0, max(N-1, 0))
        found = valid & (kd[idx] == nk)
        neighs_raw[d] = np.where(found, idx, -1).astype(np.int32)

    # Leaf mask and split GT
    leaf_mask_raw = [None] * (max_depth + 1)
    split_gt_raw = [None] * (max_depth + 1)
    values_gt_raw = [None] * (max_depth + 1)

    for d in range(max_depth + 1):
        kd = keys_raw[d]
        N = len(kd)

        # Values GT
        vgt = np.zeros((N, 1), dtype=np.float32)
        lk = leaf_keys_by_depth[d]
        lv = leaf_vals_by_depth[d]
        if len(lk) > 0:
            for i in range(len(lk)):
                idx = np.searchsorted(kd, lk[i])
                if idx < len(kd) and kd[idx] == lk[i]:
                    vgt[idx, 0] = lv[i]
        values_gt_raw[d] = vgt

        if d == max_depth:
            leaf_mask_raw[d] = np.ones(N, dtype=bool)
            split_gt_raw[d] = np.zeros(N, dtype=np.float32)
        else:
            ch = children_raw[d]
            is_leaf = np.all(ch == -1, axis=1)
            leaf_mask_raw[d] = is_leaf
            split_gt_raw[d] = (~is_leaf).astype(np.float32)

    # Step 2: PAD to fixed sizes
    result = {
        'keys': [], 'features': [], 'neighs': [], 'children_idx': [],
        'valid': [], 'leaf_mask': [], 'split_gt': [], 'values_gt': [],
        'n_real': [],  # actual count per depth
    }

    for d in range(max_depth + 1):
        M = MAX_NODES[d]
        N = len(keys_raw[d])

        # Keys: pad with 0 (doesn't matter, masked out)
        k_pad = np.zeros(M, dtype=np.int32)
        k_pad[:N] = keys_raw[d]
        result['keys'].append(k_pad)

        # Features
        f_pad = np.zeros((M, 1), dtype=np.float32)
        f_pad[:N] = features_raw[d]
        result['features'].append(f_pad)

        # Neighbors: pad with -1
        n_pad = np.full((M, 9), -1, dtype=np.int32)
        if N > 0:
            n_pad[:N] = neighs_raw[d]
        result['neighs'].append(n_pad)

        # Children: pad with -1
        c_pad = np.full((M, 4), -1, dtype=np.int32)
        if N > 0:
            c_pad[:N] = children_raw[d]
        result['children_idx'].append(c_pad)

        # Valid mask
        v_pad = np.zeros(M, dtype=bool)
        v_pad[:N] = True
        result['valid'].append(v_pad)

        # Leaf mask
        lm_pad = np.zeros(M, dtype=bool)
        if N > 0:
            lm_pad[:N] = leaf_mask_raw[d]
        result['leaf_mask'].append(lm_pad)

        # Split GT
        sg_pad = np.zeros(M, dtype=np.float32)
        if N > 0:
            sg_pad[:N] = split_gt_raw[d]
        result['split_gt'].append(sg_pad)

        # Values GT
        vg_pad = np.zeros((M, 1), dtype=np.float32)
        if N > 0:
            vg_pad[:N] = values_gt_raw[d]
        result['values_gt'].append(vg_pad)

        result['n_real'].append(N)

    return result


def tree_to_jax(tree_dict):
    """Convert padded numpy tree to JAX arrays."""
    return {
        'keys': [jnp.array(k) for k in tree_dict['keys']],
        'features': [jnp.array(f) for f in tree_dict['features']],
        'neighs': [jnp.array(n) for n in tree_dict['neighs']],
        'children_idx': [jnp.array(c) for c in tree_dict['children_idx']],
        'valid': [jnp.array(v) for v in tree_dict['valid']],
        'leaf_mask': [jnp.array(m) for m in tree_dict['leaf_mask']],
        'split_gt': [jnp.array(s) for s in tree_dict['split_gt']],
        'values_gt': [jnp.array(v) for v in tree_dict['values_gt']],
    }


def tree_to_jax_tuple(tree_dict):
    """Convert to a flat tuple structure for JAX JIT compatibility."""
    # Pack into a single dict of stacked arrays where possible
    # For lists of arrays with different sizes, use tuple of arrays
    return (
        tuple(jnp.array(k) for k in tree_dict['keys']),
        tuple(jnp.array(f) for f in tree_dict['features']),
        tuple(jnp.array(n) for n in tree_dict['neighs']),
        tuple(jnp.array(c) for c in tree_dict['children_idx']),
        tuple(jnp.array(v) for v in tree_dict['valid']),
        tuple(jnp.array(m) for m in tree_dict['leaf_mask']),
        tuple(jnp.array(s) for s in tree_dict['split_gt']),
        tuple(jnp.array(v) for v in tree_dict['values_gt']),
    )


# ==========================================
# 5. Neural Network Modules (Flax)
# ==========================================

class QuadConv(nn.Module):
    """9-neighbor convolution. Handles padded arrays via valid masking."""
    out_channels: int

    @nn.compact
    def __call__(self, features, neigh_idx, valid_mask):
        """
        features: (M, C)  — padded
        neigh_idx: (M, 9) — -1 for missing/padding
        valid_mask: (M,)  — True for real nodes
        """
        M, C = features.shape
        pad_vec = jnp.zeros((1, C))
        feat_padded = jnp.concatenate([features, pad_vec], axis=0)
        safe_idx = jnp.where(neigh_idx == -1, M, neigh_idx)
        gathered = feat_padded[safe_idx]  # (M, 9, C)
        flat = gathered.reshape(M, 9 * C)
        out = nn.Dense(self.out_channels)(flat)
        return out * valid_mask[:, None]  # zero out padding


class QuadPool(nn.Module):
    """Pool children to parent."""
    @nn.compact
    def __call__(self, child_feat, children_idx, valid_parent):
        """
        child_feat: (M_child, C)
        children_idx: (M_parent, 4) — indices into child_feat, -1 if missing
        valid_parent: (M_parent,)
        """
        M_child, C = child_feat.shape
        pad = jnp.zeros((1, C))
        feat_pad = jnp.concatenate([child_feat, pad], axis=0)
        safe_idx = jnp.where(children_idx == -1, M_child, children_idx)
        gathered = feat_pad[safe_idx]  # (M_parent, 4, C)
        valid_ch = (children_idx != -1).astype(jnp.float32)
        cnt = valid_ch.sum(axis=1, keepdims=True).clip(min=1)
        pooled = (gathered * valid_ch[..., None]).sum(axis=1) / cnt
        return pooled * valid_parent[:, None]


class CompressedEncoder(nn.Module):
    hidden: int = HIDDEN
    emb_dim: int = EMB_DIM
    max_depth: int = MAX_DEPTH
    d_bottle: int = D_BOTTLE

    @nn.compact
    def __call__(self, features, keys, neighs, children_idx, valid):
        """All inputs are tuples of padded arrays, one per depth."""
        hidden = self.hidden
        in_proj = nn.Dense(hidden, name='in_proj')
        pool = QuadPool(name='pool')

        h = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            pos = fourier_encode(node_centers_jax(keys[d], d))
            h[d] = in_proj(jnp.concatenate([features[d], pos], axis=-1))
            h[d] = h[d] * valid[d][:, None]

        # Bottom-up
        convs = [QuadConv(hidden, name=f'conv_{d}') for d in range(self.max_depth + 1)]
        for d in range(self.max_depth, 0, -1):
            pooled = pool(h[d], children_idx[d-1], valid[d-1])
            h[d-1] = h[d-1] + pooled
            if d-1 >= 1:
                h[d-1] = nn.relu(convs[d-1](h[d-1], neighs[d-1], valid[d-1]))

        # Extract bottleneck embeddings
        E = []
        for d in range(self.d_bottle + 1):
            proj = nn.Dense(self.emb_dim, name=f'to_emb_{d}')
            ln = nn.LayerNorm(name=f'emb_norm_{d}')
            gain = self.param(f'depth_gain_{d}', nn.initializers.ones, ())
            z = gain * ln(proj(h[d]))
            z = z * valid[d][:, None]
            E.append(z)
        return tuple(E)


class CompressedDecoder(nn.Module):
    hidden: int = HIDDEN
    emb_dim: int = EMB_DIM
    out_c: int = 1
    max_depth: int = MAX_DEPTH
    d_bottle: int = D_BOTTLE

    @nn.compact
    def __call__(self, emb_list, keys, neighs, children_idx, valid, leaf_mask):
        """
        Teacher-forced decode with padded arrays.
        Returns split_logits (tuple of (M_d,)), val_pred (tuple of (M_d, 1)).
        """
        hidden = self.hidden
        root_token = self.param('root_token', nn.initializers.zeros, (1, hidden))
        skip_ln = nn.LayerNorm(name='skip_norm')

        fuse1 = nn.Dense(hidden, name='fuse_1')
        fuse2 = nn.Dense(hidden, name='fuse_2')
        mix_convs = [QuadConv(hidden, name=f'mix_{d}') for d in range(self.max_depth + 1)]
        split_h1 = nn.Dense(hidden, name='split_h1')
        split_h2 = nn.Dense(1, name='split_h2')
        child_h1 = nn.Dense(hidden, name='child_h1')
        child_h2 = nn.Dense(4 * hidden, name='child_h2')
        val_h1 = nn.Dense(hidden, name='val_h1')
        val_h2 = nn.Dense(self.out_c, name='val_h2')

        h_list = [None] * (self.max_depth + 1)
        M0 = MAX_NODES[0]
        h_list[0] = jnp.broadcast_to(root_token, (M0, hidden))

        split_logits = []
        val_pred = []

        for d in range(self.max_depth + 1):
            h = h_list[d]
            kd = keys[d]
            vd = valid[d]

            pos = fourier_encode(node_centers_jax(kd, d))
            skip = self._ancestor_skip(kd, emb_list, keys, d, vd, skip_ln)
            h = nn.relu(fuse2(nn.relu(fuse1(jnp.concatenate([h, skip, pos], axis=-1)))))
            h = h * vd[:, None]

            if d >= 1:
                h = nn.relu(mix_convs[d](h, neighs[d], vd))

            vp = val_h2(nn.relu(val_h1(h)))
            vp = vp * vd[:, None]
            val_pred.append(vp)

            if d == self.max_depth:
                break

            sl = split_h2(nn.relu(split_h1(h))).squeeze(-1)
            sl = sl * vd.astype(jnp.float32)
            split_logits.append(sl)

            # Scatter to children
            cf = child_h2(nn.relu(child_h1(h))).reshape(-1, 4, hidden)
            M_next = MAX_NODES[d+1]
            h_next = jnp.zeros((M_next, hidden))
            ch = children_idx[d]
            for c in range(4):
                idx_c = ch[:, c]
                valid_c = (idx_c != -1) & vd
                safe_c = jnp.where(valid_c, idx_c, 0)
                contrib = jnp.where(valid_c[:, None], cf[:, c, :], 0.0)
                h_next = h_next.at[safe_c].add(contrib)
            h_list[d+1] = h_next

        return tuple(split_logits), tuple(val_pred)

    def _ancestor_skip(self, kd, emb_list, keys, d, valid_d, skip_ln):
        Nd = kd.shape[0]
        if d <= self.d_bottle:
            if len(emb_list) > d:
                return skip_ln(emb_list[d])
            return jnp.zeros((Nd, self.emb_dim))

        # Ancestor at d_bottle
        shift = 2 * (d - self.d_bottle)
        ancestor_keys = (kd >> shift).astype(jnp.int32)
        bottle_keys = keys[self.d_bottle]
        bottle_emb = emb_list[self.d_bottle]
        M_bottle = bottle_keys.shape[0]
        idx = jnp.searchsorted(bottle_keys, ancestor_keys).clip(0, M_bottle - 1)
        found = bottle_keys[idx] == ancestor_keys
        skip = jnp.where(found[:, None], skip_ln(bottle_emb[idx]), 0.0)

        # Multi-scale: d_bottle-1
        if self.d_bottle >= 1 and len(emb_list) > self.d_bottle - 1:
            cd = self.d_bottle - 1
            shift2 = 2 * (d - cd)
            ak2 = (kd >> shift2).astype(jnp.int32)
            ck = keys[cd]
            ce = emb_list[cd]
            M_c = ck.shape[0]
            idx2 = jnp.searchsorted(ck, ak2).clip(0, M_c - 1)
            found2 = ck[idx2] == ak2
            skip = skip + jnp.where(found2[:, None], skip_ln(ce[idx2]), 0.0)

        return skip * valid_d[:, None]

    def decode_from_z_flat(self, z_flat, keys, neighs, children_idx, valid, leaf_mask):
        """Decode from flat z (510,) -> leaf values."""
        emb_list = []
        offset = 0
        for d in range(self.d_bottle + 1):
            n_d = MAX_NODES[d]
            emb_list.append(z_flat[offset:offset + n_d * self.emb_dim].reshape(n_d, self.emb_dim))
            offset += n_d * self.emb_dim
        emb_list = tuple(emb_list)

        _, val_pred = self.__call__(emb_list, keys, neighs, children_idx, valid, leaf_mask)

        # Collect leaf values — all depths, masked
        all_vals = []
        for d in range(self.max_depth + 1):
            v = val_pred[d].squeeze(-1) * leaf_mask[d].astype(jnp.float32)
            all_vals.append(v)
        return tuple(all_vals)


class QuadtreeAE(nn.Module):
    hidden: int = HIDDEN
    emb_dim: int = EMB_DIM
    max_depth: int = MAX_DEPTH
    d_bottle: int = D_BOTTLE

    def setup(self):
        self.encoder = CompressedEncoder(
            hidden=self.hidden, emb_dim=self.emb_dim,
            max_depth=self.max_depth, d_bottle=self.d_bottle)
        self.decoder = CompressedDecoder(
            hidden=self.hidden, emb_dim=self.emb_dim, out_c=1,
            max_depth=self.max_depth, d_bottle=self.d_bottle)

    def __call__(self, features, keys, neighs, children_idx, valid, leaf_mask):
        E = self.encoder(features, keys, neighs, children_idx, valid)
        split_logits, val_pred = self.decoder(E, keys, neighs, children_idx, valid, leaf_mask)
        return split_logits, val_pred, E

    def encode_flat(self, features, keys, neighs, children_idx, valid):
        E = self.encoder(features, keys, neighs, children_idx, valid)
        return jnp.concatenate([e.ravel() for e in E], axis=0)

    def decode_flat(self, z_flat, keys, neighs, children_idx, valid, leaf_mask):
        return self.decoder.decode_from_z_flat(z_flat, keys, neighs, children_idx, valid, leaf_mask)


# ==========================================
# 6. Loss Function
# ==========================================

def compute_loss(params, model_apply, tree_tuple):
    """Compute value + split loss. JIT-compatible with fixed-shape inputs."""
    keys, features, neighs, children_idx, valid, leaf_mask, split_gt, values_gt = tree_tuple

    split_logits, val_pred, E = model_apply(
        {'params': params},
        features, keys, neighs, children_idx, valid, leaf_mask
    )

    # Value loss on leaf nodes (MSE, masked)
    val_loss = 0.0
    n_val = 0
    for d in range(MAX_DEPTH + 1):
        lm = leaf_mask[d]
        n_leaves = lm.sum()
        # Only count if there are actual leaves
        mask_f = lm.astype(jnp.float32)
        err = ((val_pred[d].squeeze(-1) - values_gt[d].squeeze(-1)) ** 2) * mask_f
        val_loss = val_loss + err.sum() / jnp.maximum(n_leaves, 1)
        n_val += 1
    val_loss = val_loss / max(n_val, 1)

    # Split loss (BCE, masked by valid)
    split_loss = 0.0
    n_split = 0
    for d in range(MAX_DEPTH):
        vd = valid[d].astype(jnp.float32)
        n_valid = vd.sum()
        bce = optax.sigmoid_binary_cross_entropy(split_logits[d], split_gt[d]) * vd
        split_loss = split_loss + bce.sum() / jnp.maximum(n_valid, 1)
        n_split += 1
    split_loss = split_loss / max(n_split, 1)

    total = val_loss + SPLIT_WEIGHT * split_loss
    return total, (val_loss, split_loss)


# ==========================================
# 7. Training
# ==========================================

def train_ae():
    print("\n" + "=" * 60)
    print("Training Quadtree AE on Poisson solutions")
    print("=" * 60)

    model = QuadtreeAE()

    # Init with a dummy sample
    lk, lv, _, _, _ = generate_sample(1, 1)
    tree_np = build_padded_tree(lk, lv)
    tree_tuple = tree_to_jax_tuple(tree_np)

    rng = jax.random.PRNGKey(42)
    params = model.init(rng,
        tree_tuple[1], tree_tuple[0], tree_tuple[2], tree_tuple[3],
        tree_tuple[4], tree_tuple[5]
    )['params']

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,}")

    # JIT compile training step
    @jax.jit
    def train_step(params, opt_state, tree_tuple):
        (loss, (vl, sl)), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            params, model.apply, tree_tuple
        )
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, vl, sl

    # Phase 1
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR_PHASE1,
        warmup_steps=200, decay_steps=PHASE1_STEPS, end_value=1e-5
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = tx.init(params)

    rng_np = np.random.RandomState(0)
    best_loss = float('inf')
    best_params = None
    patience_ctr = 0
    losses = {'val': [], 'split': [], 'total': []}

    t0 = time.perf_counter()
    print(f"\nPhase 1: {PHASE1_STEPS} steps, lr={LR_PHASE1}")

    for step in range(PHASE1_STEPS):
        k1 = rng_np.randint(1, 6)
        k2 = rng_np.randint(1, 6)
        lk, lv, leaves, _, _ = generate_sample(k1, k2, noise_std=0.02)
        tree_np = build_padded_tree(lk, lv)
        tt = tree_to_jax_tuple(tree_np)

        params, opt_state, loss, vl, sl = train_step(params, opt_state, tt)

        loss_f = float(loss)
        losses['val'].append(float(vl))
        losses['split'].append(float(sl))
        losses['total'].append(loss_f)

        if loss_f < best_loss:
            best_loss = loss_f
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            patience_ctr = 0
        else:
            patience_ctr += 1

        if step % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[P1] Step {step:5d} | val={float(vl):.6f} split={float(sl):.6f} "
                  f"total={loss_f:.6f} | k=({k1},{k2}) leaves={len(leaves)} | "
                  f"best={best_loss:.6f} pat={patience_ctr}/{PATIENCE} | {elapsed:.0f}s")

        if patience_ctr >= PATIENCE:
            print(f"Phase 1 early stop at step {step}")
            break

    print(f"Phase 1 done. Best: {best_loss:.6f}")
    with open(SCRIPT_DIR / 'quadtree_ae_phase1.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    # Phase 2: fine-tune with lower LR
    print(f"\nPhase 2: {PHASE2_STEPS} steps, lr={LR_PHASE2}")
    params = best_params
    schedule2 = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR_PHASE2,
        warmup_steps=100, decay_steps=PHASE2_STEPS, end_value=1e-6
    )
    tx2 = optax.adamw(learning_rate=schedule2, weight_decay=1e-4)
    opt_state2 = tx2.init(params)

    @jax.jit
    def train_step2(params, opt_state, tree_tuple):
        (loss, (vl, sl)), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            params, model.apply, tree_tuple
        )
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, new_opt_state = tx2.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, vl, sl

    best_loss2 = float('inf')
    patience_ctr2 = 0

    for step in range(PHASE2_STEPS):
        k1 = rng_np.randint(1, 6)
        k2 = rng_np.randint(1, 6)
        lk, lv, leaves, _, _ = generate_sample(k1, k2, noise_std=0.01)
        tree_np = build_padded_tree(lk, lv)
        tt = tree_to_jax_tuple(tree_np)

        params, opt_state2, loss, vl, sl = train_step2(params, opt_state2, tt)

        loss_f = float(loss)
        losses['val'].append(float(vl))
        losses['split'].append(float(sl))
        losses['total'].append(loss_f)

        if loss_f < best_loss2:
            best_loss2 = loss_f
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            patience_ctr2 = 0
        else:
            patience_ctr2 += 1

        gs = PHASE1_STEPS + step
        if step % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[P2] Step {gs:5d} | val={float(vl):.6f} split={float(sl):.6f} "
                  f"total={loss_f:.6f} | k=({k1},{k2}) leaves={len(leaves)} | "
                  f"best={best_loss2:.6f} pat={patience_ctr2}/{PATIENCE} | {elapsed:.0f}s")

        if patience_ctr2 >= PATIENCE:
            print(f"Phase 2 early stop at step {gs}")
            break

    print(f"Phase 2 done. Best: {best_loss2:.6f}")
    with open(SCRIPT_DIR / 'quadtree_ae_final.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    # Plot losses
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(losses['val'], lw=0.5, alpha=0.7)
    axes[0].set_title('Value Loss'); axes[0].set_yscale('log'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(losses['split'], lw=0.5, alpha=0.7, color='coral')
    axes[1].set_title('Split Loss'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(losses['total'], lw=0.5, alpha=0.7, color='navy')
    axes[2].set_title('Total Loss'); axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)
    plt.suptitle('Quadtree AE Training', fontsize=14)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'plots' / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Loss plot saved.")

    return model, best_params, losses


# ==========================================
# 8. AE Reconstruction Check
# ==========================================

def check_ae_quality(model, params):
    print("\n" + "=" * 60)
    print("AE Reconstruction Check")
    print("=" * 60)

    for k1, k2 in [(1,1), (2,3), (3,2), (4,1), (1,5), (5,5)]:
        lk, lv, leaves, _, _ = generate_sample(k1, k2)
        tree_np = build_padded_tree(lk, lv)
        tt = tree_to_jax_tuple(tree_np)
        keys, features, neighs, children_idx, valid, leaf_mask, split_gt, values_gt = tt

        _, val_pred, _ = model.apply(
            {'params': params},
            features, keys, neighs, children_idx, valid, leaf_mask
        )

        total_err, total_norm, n_leaves = 0.0, 0.0, 0
        for d in range(MAX_DEPTH + 1):
            lm = leaf_mask[d]
            if lm.sum() > 0:
                pred = val_pred[d].squeeze(-1) * lm
                gt = values_gt[d].squeeze(-1) * lm
                total_err += float(jnp.sum((pred - gt)**2))
                total_norm += float(jnp.sum(gt**2))
                n_leaves += int(lm.sum())

        rel_l2 = np.sqrt(total_err / max(total_norm, 1e-12))
        print(f"  k=({k1},{k2}): {n_leaves} leaves, rel_L2 = {rel_l2:.6e}")


# ==========================================
# 9. NM-ROM Gauss-Newton Solver
# ==========================================

def assemble_poisson_on_leaves(tree_np):
    """
    Build Poisson operator for leaf cells of the quadtree.

    Returns:
        K_op: function (n_leaves,) -> (n_leaves,)
        leaf_coords: (n_leaves, 2) cell centers
        leaf_sizes: (n_leaves,) cell widths
        n_leaves: int
        boundary_mask: (n_leaves,) bool
        leaf_depth_idx: list of (depth, padded_index) for each leaf
    """
    # Collect leaf info from padded arrays
    leaf_info = []  # (depth, padded_idx, morton_key)
    for d in range(MAX_DEPTH + 1):
        valid = tree_np['valid'][d]
        lm = tree_np['leaf_mask'][d]
        kd = tree_np['keys'][d]
        for i in range(len(kd)):
            if valid[i] and lm[i]:
                leaf_info.append((d, i, int(kd[i])))

    n_leaves = len(leaf_info)

    # Compute centers and sizes
    coords = np.zeros((n_leaves, 2), dtype=np.float32)
    sizes = np.zeros(n_leaves, dtype=np.float32)
    for li, (d, pi, mk) in enumerate(leaf_info):
        x, y = morton_key2xy_np(np.array([mk]))[0][0], morton_key2xy_np(np.array([mk]))[1][0]
        res = 1 << d
        h = 1.0 / res
        coords[li, 0] = (x + 0.5) * h
        coords[li, 1] = (y + 0.5) * h
        sizes[li] = h

    # Build neighbor lookup among leaves
    # For each leaf, find 4 FD neighbors (N, S, E, W)
    offsets_xy = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float32)
    neighbor_idx = np.full((n_leaves, 4), -1, dtype=np.int32)

    # Build spatial lookup: cell -> leaf index
    # Use a dict keyed by (approximate x, y)
    cell_lookup = {}
    for li in range(n_leaves):
        # Key by center coords rounded to avoid floating point issues
        cx, cy = round(coords[li, 0], 8), round(coords[li, 1], 8)
        cell_lookup[(cx, cy)] = li

    for li in range(n_leaves):
        cx, cy = coords[li, 0], coords[li, 1]
        h = sizes[li]
        for nbr in range(4):
            nx = cx + offsets_xy[nbr, 0] * h
            ny = cy + offsets_xy[nbr, 1] * h
            if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                continue
            # Find leaf containing (nx, ny)
            best = -1
            best_h = 2.0
            for li2 in range(n_leaves):
                cx2, cy2 = coords[li2, 0], coords[li2, 1]
                h2 = sizes[li2]
                if abs(nx - cx2) <= h2/2 + 1e-8 and abs(ny - cy2) <= h2/2 + 1e-8:
                    if h2 < best_h:
                        best = li2
                        best_h = h2
            neighbor_idx[li, nbr] = best

    # Boundary detection
    eps = 1e-6
    boundary = (
        (coords[:, 0] - sizes/2 < eps) |
        (coords[:, 0] + sizes/2 > 1-eps) |
        (coords[:, 1] - sizes/2 < eps) |
        (coords[:, 1] + sizes/2 > 1-eps)
    )

    # Convert to JAX
    jax_nbr = jnp.array(neighbor_idx)
    jax_sizes = jnp.array(sizes)
    jax_coords = jnp.array(coords)
    jax_boundary = jnp.array(boundary)

    def K_op(u_flat):
        u_pad = jnp.concatenate([u_flat, jnp.zeros(1)])
        safe = jnp.where(jax_nbr == -1, n_leaves, jax_nbr)
        u_nbrs = u_pad[safe]
        h = jax_sizes
        lap = (4.0 * u_flat - u_nbrs.sum(axis=1)) / (h * h)
        return jnp.where(jax_boundary, u_flat, lap)

    return K_op, jax_coords, jax_sizes, n_leaves, jax_boundary, leaf_info


def make_rom_solver(model, params, max_iters=GN_MAX_ITERS, tol=GN_TOL):
    """Create NM-ROM GN solver."""

    def rom_solve(z_init, tree_tuple, K_op, F_vec, boundary_mask):
        """
        GN solve in latent space.
        z_init: (K_DIM,)
        tree_tuple: padded JAX arrays
        K_op: Poisson operator on leaves
        F_vec: (n_leaves,) forcing
        boundary_mask: (n_leaves,) bool
        """
        keys, features, neighs, children_idx, valid, leaf_mask, split_gt, values_gt = tree_tuple

        def decode_to_leaves(z):
            """Decode z -> leaf values, apply boundary."""
            val_by_depth = model.apply(
                {'params': params}, z,
                keys, neighs, children_idx, valid, leaf_mask,
                method=model.decode_flat
            )
            # Collect leaf values in order
            leaf_vals = []
            for d in range(MAX_DEPTH + 1):
                lm = leaf_mask[d]
                v = val_by_depth[d] * lm.astype(jnp.float32)
                # Extract actual leaves (non-zero mask entries)
                leaf_vals.append(v)
            return leaf_vals

        def decode_flat_leaves(z):
            """Decode z -> flat leaf vector."""
            vals = decode_to_leaves(z)
            # Need to extract just the leaf entries in order
            # This is done at the padded level — we collect non-zero entries
            pieces = []
            for d in range(MAX_DEPTH + 1):
                pieces.append(vals[d])
            return pieces

        z = z_init
        res_norms = []

        for iteration in range(max_iters):
            # Forward: decode z -> padded val arrays
            val_by_depth = model.apply(
                {'params': params}, z,
                keys, neighs, children_idx, valid, leaf_mask,
                method=model.decode_flat
            )

            # Collect leaf values into flat vector (matching leaf_info order)
            # For the Poisson operator, we need values in the same order as leaves
            # The decode_flat returns padded arrays by depth
            # We need to extract the leaf entries

            # Compute u_pred on leaves: iterate through depths, pick leaf entries
            u_pred_list = []
            for d in range(MAX_DEPTH + 1):
                lm = leaf_mask[d]
                vals_d = val_by_depth[d]  # (M_d,)
                # We need the actual leaf values in the right order
                # Since leaf_info is built from valid & leaf_mask, and entries are
                # stored at indices 0..N_real-1, we can just take those entries
                u_pred_list.append(vals_d)

            # Actually, let's use a simpler approach: build the mapping once
            # For now, compute residual at the padded level and project
            # This is cleaner: define a function z -> u_leaves_flat
            break

        # Simpler approach: define decode z -> flat leaf array using leaf_info
        # This requires knowing leaf_info at JIT time, which we can't.
        # Instead, pass leaf_info as static data.

        # We'll use a non-JIT approach for the GN solver (it's only ~12 iterations)
        return _gn_solve_eager(z_init, model, params, tree_tuple, K_op, F_vec,
                               boundary_mask, max_iters, tol)

    return rom_solve


def _gn_solve_eager(z_init, model, params, tree_tuple, K_op, F_vec,
                     boundary_mask, max_iters, tol):
    """Eager (non-JIT) GN solver. Fine for ~12 iterations."""
    keys, features, neighs, children_idx, valid, leaf_mask, split_gt, values_gt = tree_tuple

    # Build leaf extraction indices: for each depth, which padded indices are leaves
    leaf_indices = []  # list of (depth, array_of_padded_indices)
    for d in range(MAX_DEPTH + 1):
        lm = np.array(leaf_mask[d])
        idxs = np.where(lm)[0]
        leaf_indices.append((d, jnp.array(idxs)))

    n_leaves = sum(len(idx) for _, idx in leaf_indices)

    def decode_to_flat_leaves(z):
        """z (K_DIM,) -> u_leaves (n_leaves,)"""
        val_by_depth = model.apply(
            {'params': params}, z,
            keys, neighs, children_idx, valid, leaf_mask,
            method=model.decode_flat
        )
        pieces = []
        for d, idxs in leaf_indices:
            if len(idxs) > 0:
                pieces.append(val_by_depth[d][idxs])
        u = jnp.concatenate(pieces) if pieces else jnp.zeros(0)
        # Apply boundary
        u = jnp.where(boundary_mask, 0.0, u)
        return u

    z = z_init
    res_norms = []

    for iteration in range(max_iters):
        u_pred, vjp_fn = jax.vjp(decode_to_flat_leaves, z)

        R = K_op(u_pred) - F_vec
        R = jnp.where(boundary_mask, 0.0, R)

        r_red = vjp_fn(R)[0]
        res_norm = float(jnp.linalg.norm(r_red))
        res_norms.append(res_norm)
        print(f"    GN iter {iteration}: ||r_red|| = {res_norm:.6e}")

        if res_norm < tol:
            print(f"    Converged at iteration {iteration}")
            break

        def gn_matvec(dz):
            _, J_dz = jax.jvp(decode_to_flat_leaves, (z,), (dz,))
            K_J_dz = K_op(J_dz)
            K_J_dz = jnp.where(boundary_mask, 0.0, K_J_dz)
            return vjp_fn(K_J_dz)[0]

        delta_z, _ = jax_linalg.cg(gn_matvec, -r_red, tol=1e-3, maxiter=50)
        z = z + delta_z

    u_final = decode_to_flat_leaves(z)
    return z, u_final, res_norms


# ==========================================
# 10. Benchmark
# ==========================================

def benchmark_nmrom(model, params):
    print("\n" + "=" * 60)
    print("NM-ROM Benchmark")
    print("=" * 60)

    test_cases = [(1.5, 2.3), (2.1, 1.4), (1.2, 2.8), (3.5, 1.7), (2.0, 3.0), (4.0, 2.0)]
    results = []

    for ci, (k1, k2) in enumerate(test_cases):
        print(f"\n--- Test {ci+1}: k=({k1}, {k2}) ---")

        lk, lv, leaves, _, _ = generate_sample(k1, k2)
        tree_np = build_padded_tree(lk, lv)
        tt = tree_to_jax_tuple(tree_np)

        K_op, leaf_coords, leaf_sizes, n_leaves, boundary, leaf_info = assemble_poisson_on_leaves(tree_np)

        # Analytical F and u on leaves
        x, y = leaf_coords[:, 0], leaf_coords[:, 1]
        F_vec = jnp.sin(2*jnp.pi*k1*x) * jnp.sin(2*jnp.pi*k2*y)
        F_vec = jnp.where(boundary, 0.0, F_vec)
        c = 1.0 / (4*jnp.pi**2*(k1**2 + k2**2))
        u_exact = c * jnp.sin(2*jnp.pi*k1*x) * jnp.sin(2*jnp.pi*k2*y)

        print(f"  Leaves: {n_leaves}, boundary: {int(boundary.sum())}")

        z_init = jnp.zeros(K_DIM)
        t0 = time.perf_counter()
        z_final, u_pred, res_norms = _gn_solve_eager(
            z_init, model, params, tt, K_op, F_vec, boundary, GN_MAX_ITERS, GN_TOL)
        t_rom = time.perf_counter() - t0

        rel_l2 = float(jnp.linalg.norm(u_exact - u_pred) / (jnp.linalg.norm(u_exact) + 1e-12))
        max_err = float(jnp.max(jnp.abs(u_exact - u_pred)))

        print(f"  ROM time: {t_rom:.4f}s, rel_L2: {rel_l2:.4e}, max_err: {max_err:.4e}, iters: {len(res_norms)}")

        results.append({
            'k1': k1, 'k2': k2, 'n_leaves': n_leaves,
            'rel_l2': rel_l2, 'max_err': max_err,
            'time': t_rom, 'gn_iters': len(res_norms),
            'res_norms': res_norms,
            'u_exact': np.array(u_exact), 'u_pred': np.array(u_pred),
            'leaf_coords': np.array(leaf_coords), 'leaf_sizes': np.array(leaf_sizes),
            'boundary': np.array(boundary),
        })

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    avg_err = np.mean([r['rel_l2'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    print(f"Avg rel_L2: {avg_err:.4e}, Avg time: {avg_time:.4f}s")
    for r in results:
        print(f"  k=({r['k1']},{r['k2']}): rel_L2={r['rel_l2']:.4e} time={r['time']:.3f}s iters={r['gn_iters']}")

    plot_results(results)
    return results


def plot_results(results):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5*n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(results):
        coords = r['leaf_coords']
        sizes = r['leaf_sizes']
        u_ex = r['u_exact']
        u_pr = r['u_pred']
        err = np.abs(u_ex - u_pr)

        vmin, vmax = u_ex.min(), u_ex.max()
        emax = max(err.max(), 1e-12)

        for j, (data, title, cmap_name, lo, hi) in enumerate([
            (u_ex, f"Analytical k=({r['k1']},{r['k2']})", 'inferno', vmin, vmax),
            (u_pr, f"NM-ROM rel_L2={r['rel_l2']:.2e}", 'inferno', vmin, vmax),
            (err, f"|Error| max={r['max_err']:.2e}", 'hot', 0, emax),
        ]):
            ax = axes[i, j]
            for k in range(len(coords)):
                x = coords[k, 0] - sizes[k]/2
                y = coords[k, 1] - sizes[k]/2
                c = plt.cm.get_cmap(cmap_name)((data[k] - lo) / max(hi - lo, 1e-12))
                ax.add_patch(plt.Rectangle((x, y), sizes[k], sizes[k], fc=c, ec='k', lw=0.15))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_title(title)

    plt.suptitle(f"Quadtree NM-ROM 2D Poisson (z_dim={K_DIM})", fontsize=14)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'plots' / 'nmrom_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.semilogy(r['res_norms'], 'o-', label=f"k=({r['k1']},{r['k2']})")
    ax.set_xlabel('GN Iteration'); ax.set_ylabel('||r_red||')
    ax.set_title('GN Convergence'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'plots' / 'gn_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plots saved.")


# ==========================================
# 11. Main
# ==========================================

def main():
    print(f"\nStarting Quadtree NM-ROM pipeline...")
    model, params, losses = train_ae()
    check_ae_quality(model, params)
    results = benchmark_nmrom(model, params)
    print("\n=== QUADTREE NM-ROM COMPLETE ===")
    return results

if __name__ == '__main__':
    main()
