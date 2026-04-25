"""
compressed_quadtree_ae.py

Trains a COMPRESSED quadtree autoencoder for the Poisson forcing function f.

Key idea — coarse-level bottleneck:
  Encoder  : bottom-up QuadConv + pooling  →  keeps ONLY depth D_BOTTLE embeddings
             Fixed bottleneck: at most 4^D_BOTTLE nodes × EMB_DIM floats
             (D_BOTTLE=3 → max 64 nodes × 128 dims = 8 192 floats, regardless
              of how many adaptive leaves the input tree has)

  Decoder  : autoregressive top-down expansion from D_BOTTLE → MAX_LEVEL
             At each depth it predicts:
               - split logit   (should this node expand further?)
               - value         (what is f here if this is a leaf?)
             Training uses teacher forcing: GT topology drives the expansion.
             Inference uses threshold on split logits (autoregressive).

PDE context:
  f(x,y) = sin(2πk1 x) sin(2πk2 y)   on [0,1]², k1,k2 ~ Uniform{1..5}
  Adaptive quadtree refined by |∇f| — denser where f oscillates fastest.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# Hyper-parameters
# ══════════════════════════════════════════════════════════════════════════════
MAX_LEVEL = 7        # deepest quadtree depth
MIN_LEVEL = 6        # always refine at least to this depth
D_BOTTLE  = 3        # bottleneck depth (max 4^3 = 64 nodes)
EMB_DIM   = 128
HIDDEN    = 128
POS_FREQS = 6
NOISE_STD = 0.05
POS_DIM   = 3 + 2 * POS_FREQS * 3   # 39  (x, y, d/D, sin/cos Fourier features)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Morton-code utilities
# ══════════════════════════════════════════════════════════════════════════════

class Morton2D:
    @staticmethod
    def _interleave(x):
        x = torch.as_tensor(x, dtype=torch.long)
        x &= 0x0000FFFF
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        return x

    @staticmethod
    def _deinterleave(x):
        x = torch.as_tensor(x, dtype=torch.long)
        x &= 0x55555555
        x = (x | (x >> 1)) & 0x33333333
        x = (x | (x >> 2)) & 0x0F0F0F0F
        x = (x | (x >> 4)) & 0x00FF00FF
        x = (x | (x >> 8)) & 0x0000FFFF
        return x

    @staticmethod
    def xy2key(x, y):
        return (Morton2D._interleave(x) | (Morton2D._interleave(y) << 1)).long()

    @staticmethod
    def key2xy(key):
        key = torch.as_tensor(key, dtype=torch.long)
        return Morton2D._deinterleave(key).long(), Morton2D._deinterleave(key >> 1).long()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Positional encoding
# ══════════════════════════════════════════════════════════════════════════════

def node_centers(keys, depth, device=None):
    """Return (N, 3) — normalised x, y, depth/MAX_LEVEL."""
    if device is None:
        device = keys.device
    if keys.numel() == 0:
        return torch.zeros((0, 3), device=device)
    ix, iy = Morton2D.key2xy(keys)
    res  = float(1 << depth)
    x    = (ix.float() + 0.5) / res
    y    = (iy.float() + 0.5) / res
    dnrm = torch.full_like(x, depth / MAX_LEVEL)
    return torch.stack([x, y, dnrm], dim=1).to(device)


def fourier_encode(pos, num_freqs=POS_FREQS):
    """(N, POS_DIM) Fourier feature positional encoding."""
    if pos.numel() == 0:
        return pos
    freqs = (2.0 ** torch.arange(num_freqs, device=pos.device,
                                  dtype=pos.dtype)).view(1, 1, -1)
    x   = pos.unsqueeze(-1) * np.pi * 2.0 * freqs
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(pos.shape[0], -1)
    return torch.cat([pos, enc], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Spatial conv / pool helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_neighs(keys, depth, device):
    """(N, 9) sorted index of 3×3 same-depth neighbourhood (-1 = absent)."""
    N = len(keys)
    if N == 0:
        return torch.empty((0, 9), dtype=torch.long, device=device)
    if depth == 0:
        ni = torch.full((N, 9), -1, dtype=torch.long, device=device)
        ni[:, 4] = 0
        return ni
    ix, iy = Morton2D.key2xy(keys)
    offs   = torch.tensor([[-1,-1],[-1,0],[-1,1],
                            [ 0,-1],[ 0,0],[ 0,1],
                            [ 1,-1],[ 1,0],[ 1,1]],
                          device=device, dtype=torch.long)
    nc     = torch.stack([ix, iy], 1).unsqueeze(1) + offs.unsqueeze(0)
    res    = 1 << depth
    nx, ny = nc[..., 0], nc[..., 1]
    valid  = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
    nk     = torch.full((N, 9), -1, dtype=torch.long, device=device)
    if valid.any():
        nk[valid] = Morton2D.xy2key(nx[valid], ny[valid])
    idx   = torch.searchsorted(keys, nk.clamp(min=0)).clamp(0, N - 1)
    found = valid & (keys[idx] == nk)
    idx[~found] = -1
    return idx


def build_children_idx(keys_parent, keys_child, device):
    """(Np, 4) indices of children of each parent node into keys_child (-1 absent)."""
    Np = len(keys_parent)
    if Np == 0:
        return torch.empty((0, 4), dtype=torch.long, device=device)
    Nc = len(keys_child)
    if Nc == 0:
        return torch.full((Np, 4), -1, dtype=torch.long, device=device)
    ck    = (keys_parent.unsqueeze(1) << 2) + torch.arange(4, device=device).view(1, 4)
    idx   = torch.searchsorted(keys_child, ck).clamp(0, Nc - 1)
    found = keys_child[idx] == ck
    return torch.where(found, idx, torch.full_like(idx, -1))


def pool_up(child_feat, ch_idx):
    """Average-pool child features to parent.  ch_idx: (Np, 4)."""
    Np = len(ch_idx)
    C  = child_feat.shape[1]
    pooled = torch.zeros((Np, C), device=child_feat.device)
    cnt    = torch.zeros((Np, 1), device=child_feat.device)
    for c in range(4):
        idx  = ch_idx[:, c]
        mask = idx != -1
        if mask.any():
            pooled[mask] += child_feat[idx[mask]]
            cnt[mask]    += 1.0
    return pooled / cnt.clamp(min=1.0)


class QuadConv(nn.Module):
    """Gather 3×3 neighbourhood and linearly mix."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.fc = nn.Linear(9 * in_c, out_c)

    def forward(self, feat, neighs):
        N = neighs.shape[0]
        if N == 0:
            return torch.zeros((0, self.fc.out_features), device=feat.device)
        pad = torch.zeros((1, feat.shape[1]), device=feat.device)
        fp  = torch.cat([feat, pad], 0)
        gi  = neighs.clone(); gi[gi == -1] = N
        return self.fc(fp[gi].view(N, -1))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Data: f on adaptive quadtree
# ══════════════════════════════════════════════════════════════════════════════

def f_fn(x, y, k1, k2):
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)

def grad_mag_f(x, y, k1, k2):
    gx = 2*np.pi*k1 * np.cos(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y)
    gy = 2*np.pi*k2 * np.sin(2*np.pi*k1*x) * np.cos(2*np.pi*k2*y)
    return np.sqrt(gx**2 + gy**2)


class QuadNode:
    __slots__ = ('x', 'y', 'size', 'level', 'children', 'f_val')
    def __init__(self, x, y, size, level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.children = []
        self.f_val    = None

    def subdivide(self, k1, k2, thresh=0.3):
        if self.level >= MAX_LEVEL:
            return
        cx, cy = self.x + self.size / 2, self.y + self.size / 2
        force  = self.level < MIN_LEVEL
        g_norm = grad_mag_f(cx, cy, k1, k2) / (2 * np.pi * np.sqrt(k1**2 + k2**2) + 1e-8)
        if force or g_norm > thresh * (1 + 0.1 * self.level):
            h = self.size / 2
            self.children = [
                QuadNode(self.x,   self.y,   h, self.level + 1),
                QuadNode(self.x+h, self.y,   h, self.level + 1),
                QuadNode(self.x,   self.y+h, h, self.level + 1),
                QuadNode(self.x+h, self.y+h, h, self.level + 1),
            ]
            for c in self.children:
                c.subdivide(k1, k2, thresh)

    def collect_leaves(self, out, k1, k2):
        if not self.children:
            cx, cy  = self.x + self.size / 2, self.y + self.size / 2
            self.f_val = f_fn(cx, cy, k1, k2)
            out.append(self)
        else:
            for c in self.children:
                c.collect_leaves(out, k1, k2)


def generate_f_tree(noise_std=NOISE_STD):
    """
    Returns
    -------
    leaf_keys  : list[Tensor]  (MAX_LEVEL+1,)  Morton keys at each depth
    f_clean    : list[Tensor]  (N_d, 1)        analytic f values
    f_noisy    : list[Tensor]  (N_d, 1)        f + Gaussian noise
    leaves     : list[QuadNode]
    k1, k2     : int
    """
    k1, k2 = random.randint(1, 5), random.randint(1, 5)
    root   = QuadNode(0, 0, 1.0, 0)
    root.subdivide(k1, k2)
    leaves = []
    root.collect_leaves(leaves, k1, k2)

    lk = [[] for _ in range(MAX_LEVEL + 1)]
    lv = [[] for _ in range(MAX_LEVEL + 1)]
    for nd in leaves:
        d   = nd.level
        res = 1 << d
        ix  = max(0, min(res - 1, int(nd.x * res)))
        iy  = max(0, min(res - 1, int(nd.y * res)))
        lk[d].append(int(Morton2D.xy2key(ix, iy).item()))
        lv[d].append([float(nd.f_val)])

    leaf_keys, f_clean, f_noisy = [], [], []
    for d in range(MAX_LEVEL + 1):
        if not lk[d]:
            leaf_keys.append(torch.empty((0,), dtype=torch.long))
            f_clean.append(torch.empty((0, 1)))
            f_noisy.append(torch.empty((0, 1)))
        else:
            kt = torch.tensor(lk[d], dtype=torch.long)
            fc = torch.tensor(lv[d], dtype=torch.float32)
            leaf_keys.append(kt)
            f_clean.append(fc)
            f_noisy.append(fc + noise_std * torch.randn_like(fc))

    return leaf_keys, f_clean, f_noisy, leaves, k1, k2


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TreeData  — ancestor-closed key sets + all structural indices
# ══════════════════════════════════════════════════════════════════════════════

class TreeData:
    """
    Lightweight container built once per sample.  Holds:
      keys      [d] : (N_d,)       sorted Morton keys (ancestor-closed)
      feat      [d] : (N_d, C)     noisy input features
      values_gt [d] : (N_d, C)     clean target values
      neighs    [d] : (N_d, 9)     same-depth neighbour indices
      ch_idx    [d] : (N_d, 4)     children-in-(d+1) indices (d < MAX_LEVEL)
      leaf_mask [d] : (N_d,) bool  True  iff node has no children
      split_gt  [d] : (N_d,) float 1.0 iff node should split (d < MAX_LEVEL)
    """
    def __init__(self, leaf_keys_by_depth, leaf_vals_noisy,
                 leaf_vals_clean, device):
        self.device    = device
        self.keys      = [None] * (MAX_LEVEL + 1)
        self.feat      = [None] * (MAX_LEVEL + 1)
        self.values_gt = [None] * (MAX_LEVEL + 1)
        self.neighs    = [None] * (MAX_LEVEL + 1)
        self.ch_idx    = [None] * MAX_LEVEL
        self.leaf_mask = [None] * (MAX_LEVEL + 1)
        self.split_gt  = [None] * MAX_LEVEL
        self._build(leaf_keys_by_depth, leaf_vals_noisy, leaf_vals_clean)

    # ── internal build ────────────────────────────────────────────────────────
    def _build(self, lkbd, lv_noisy, lv_clean):
        dev = self.device
        C   = next((lv_noisy[d].shape[1]
                    for d in range(MAX_LEVEL + 1)
                    if lv_noisy[d].numel() > 0), 1)

        # Step 1: seed keys and features from leaf input
        for d in range(MAX_LEVEL + 1):
            lk = lkbd[d].to(dev)
            lv = lv_noisy[d].to(dev).float()
            if lk.numel() == 0:
                self.keys[d] = torch.empty((0,), dtype=torch.long, device=dev)
                self.feat[d] = torch.zeros((0, C), device=dev)
            else:
                uk, inv = torch.unique(lk, sorted=True, return_inverse=True)
                self.keys[d] = uk
                feat = torch.zeros((len(uk), C), device=dev)
                cnt  = torch.zeros((len(uk), 1), device=dev)
                feat.index_add_(0, inv, lv)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=dev))
                self.feat[d] = feat / cnt.clamp(min=1)

        # Root must always exist
        if self.keys[0].numel() == 0:
            self.keys[0] = torch.tensor([0], dtype=torch.long, device=dev)
            self.feat[0] = torch.zeros((1, C), device=dev)

        # Step 2: ancestor closure — walk upward and add missing parents
        for d in range(MAX_LEVEL, 0, -1):
            if self.keys[d].numel() == 0:
                continue
            parents       = torch.unique(self.keys[d] >> 2, sorted=True)
            self.keys[d-1] = torch.unique(
                torch.cat([self.keys[d-1], parents]), sorted=True)

        # Step 3: rebuild features for the (now expanded) key sets
        for d in range(MAX_LEVEL + 1):
            kd   = self.keys[d]
            feat = torch.zeros((len(kd), C), device=dev)
            lk   = lkbd[d].to(dev)
            lv   = lv_noisy[d].to(dev).float()
            if lk.numel() > 0:
                uk, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled  = torch.zeros((len(uk), C), device=dev)
                cnt     = torch.zeros((len(uk), 1), device=dev)
                pooled.index_add_(0, inv, lv)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=dev))
                pooled /= cnt.clamp(min=1)
                idx     = torch.searchsorted(kd, uk).clamp(0, len(kd) - 1)
                ok      = kd[idx] == uk
                feat[idx[ok]] = pooled[ok]
            self.feat[d] = feat

        # Step 4: children indices at every depth
        for d in range(MAX_LEVEL):
            self.ch_idx[d] = build_children_idx(
                self.keys[d], self.keys[d + 1], dev)

        # Step 5: neighbour indices
        for d in range(MAX_LEVEL + 1):
            self.neighs[d] = build_neighs(self.keys[d], d, dev)

        # Step 6: leaf mask and split GT
        for d in range(MAX_LEVEL + 1):
            if d == MAX_LEVEL:
                self.leaf_mask[d] = torch.ones(
                    len(self.keys[d]), dtype=torch.bool, device=dev)
            else:
                has_child         = (self.ch_idx[d] != -1).any(dim=1)
                self.leaf_mask[d] = ~has_child
                self.split_gt[d]  = has_child.float()

        # Step 7: clean GT values (needed for reconstruction loss)
        for d in range(MAX_LEVEL + 1):
            kd = self.keys[d]
            gt = torch.zeros((len(kd), C), device=dev)
            lk = lkbd[d].to(dev)
            lv = lv_clean[d].to(dev).float()
            if lk.numel() > 0:
                uk, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled  = torch.zeros((len(uk), C), device=dev)
                cnt     = torch.zeros((len(uk), 1), device=dev)
                pooled.index_add_(0, inv, lv)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=dev))
                pooled /= cnt.clamp(min=1)
                idx     = torch.searchsorted(kd, uk).clamp(0, len(kd) - 1)
                ok      = kd[idx] == uk
                gt[idx[ok]] = pooled[ok]
            self.values_gt[d] = gt


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Compressed Encoder
#     Bottom-up from MAX_LEVEL → D_BOTTLE.  Only returns depth-D_BOTTLE tensor.
# ══════════════════════════════════════════════════════════════════════════════

class CompressedEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=HIDDEN, emb_dim=EMB_DIM,
                 d_bottle=D_BOTTLE, max_depth=MAX_LEVEL):
        super().__init__()
        self.d_bottle  = d_bottle
        self.max_depth = max_depth
        self.hidden    = hidden
        self.emb_dim   = emb_dim

        self.in_proj  = nn.Linear(in_c + POS_DIM, hidden)
        # One QuadConv per level used during encoding (D_BOTTLE … MAX_LEVEL-1)
        self.convs    = nn.ModuleList(
            [QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.to_emb   = nn.Linear(hidden, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)

    def forward(self, td: TreeData):
        """Returns z: (N_bottle, emb_dim) — embeddings at D_BOTTLE."""
        dev = td.device

        # Project each depth's features into hidden space
        h = []
        for d in range(self.max_depth + 1):
            fin = td.feat[d]
            if fin is None or fin.numel() == 0:
                h.append(torch.zeros((0, self.hidden), device=dev))
                continue
            pos  = fourier_encode(node_centers(td.keys[d], d, dev))
            h.append(F.relu(self.in_proj(torch.cat([fin, pos], dim=1))))

        # Pool upward from MAX_LEVEL down to D_BOTTLE
        for d in range(self.max_depth, self.d_bottle, -1):
            if h[d].numel() == 0:
                continue
            # Add pooled children into parent
            h[d-1] = h[d-1] + pool_up(h[d], td.ch_idx[d - 1])
            if h[d-1].numel() > 0:
                h[d-1] = F.relu(self.convs[d-1](h[d-1], td.neighs[d-1]))

        # Bottleneck projection at D_BOTTLE
        if h[self.d_bottle].numel() == 0:
            return torch.zeros((0, self.emb_dim), device=dev)

        z = self.emb_norm(self.to_emb(h[self.d_bottle]))  # (N_bottle, emb_dim)
        return z


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Autoregressive Decoder
#     Top-down from D_BOTTLE → MAX_LEVEL with teacher-forcing during training.
# ══════════════════════════════════════════════════════════════════════════════

class AutoregressiveDecoder(nn.Module):
    def __init__(self, hidden=HIDDEN, emb_dim=EMB_DIM, out_c=1,
                 d_bottle=D_BOTTLE, max_depth=MAX_LEVEL):
        super().__init__()
        self.d_bottle  = d_bottle
        self.max_depth = max_depth
        self.hidden    = hidden
        self.emb_dim   = emb_dim

        # Fuse [hidden_state | skip_z | pos] → hidden
        self.fuse = nn.Sequential(
            nn.Linear(hidden + emb_dim + POS_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.skip_norm = nn.LayerNorm(emb_dim)

        # Spatial mixing at each depth during decode
        self.mix_convs = nn.ModuleList(
            [QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

        # Split hidden into 4 child hidden states
        self.child_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4 * hidden),
        )

        # Output heads
        self.val_head   = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))
        self.split_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        # Learnable start token at D_BOTTLE (one per bottleneck node)
        self.bottle_token = nn.Parameter(torch.zeros(1, hidden))

    def forward(self, z_bottle, td: TreeData):
        """
        Parameters
        ----------
        z_bottle : (N_bottle, emb_dim)  encoder output at D_BOTTLE
        td        : TreeData             GT tree for teacher-forced expansion

        Returns
        -------
        val_pred   : list[Tensor | None]  per-depth (N_d, out_c)  predictions
        split_pred : list[Tensor | None]  per-depth (N_d,)         split logits
        """
        dev      = td.device
        N_bottle = z_bottle.shape[0]

        # Initialise hidden at D_BOTTLE
        h = [None] * (self.max_depth + 1)
        h[self.d_bottle] = self.bottle_token.expand(N_bottle, -1).to(dev)

        val_pred   = [None] * (self.max_depth + 1)
        split_pred = [None] * self.max_depth

        for d in range(self.d_bottle, self.max_depth + 1):
            hd = h[d]
            Nd = len(td.keys[d])

            if hd is None or hd.numel() == 0 or Nd == 0:
                val_pred[d] = torch.zeros((Nd, 1), device=dev)
                if d < self.max_depth:
                    split_pred[d] = torch.zeros((Nd,), device=dev)
                continue

            # Fuse hidden state with skip (z at bottle depth) and position
            pos = fourier_encode(node_centers(td.keys[d], d, dev))
            if d == self.d_bottle:
                skip = self.skip_norm(z_bottle)
            else:
                skip = torch.zeros((Nd, self.emb_dim), device=dev)

            hd = self.fuse(torch.cat([hd, skip, pos], dim=1))

            # Spatial mixing among neighbours
            hd = F.relu(self.mix_convs[d](hd, td.neighs[d]))

            # Predict values at this depth
            val_pred[d] = self.val_head(hd)

            if d == self.max_depth:
                break

            # Predict split logits
            split_pred[d] = self.split_head(hd).squeeze(-1)

            # ── Teacher forcing: scatter child states to GT child positions ──
            ch   = td.ch_idx[d]                          # (Nd, 4)
            N_next = len(td.keys[d + 1])
            h_next = torch.zeros((N_next, self.hidden), device=dev)

            has_child = (ch != -1).any(dim=1)
            if has_child.any():
                child_feats  = self.child_head(hd[has_child]).view(-1, 4, self.hidden)
                parent_rows  = torch.nonzero(has_child, as_tuple=False).squeeze(-1)
                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1:
                            h_next[ci] = child_feats[t, c]

            h[d + 1] = h_next

        return val_pred, split_pred


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Full compressed AE
# ══════════════════════════════════════════════════════════════════════════════

class CompressedQuadtreeAE(nn.Module):
    """
    encoder  :  TreeData  →  z (N_bottle, EMB_DIM)    (N_bottle ≤ 4^D_BOTTLE)
    decoder  :  z, TreeData  →  val_pred, split_pred   (teacher-forced)
    """
    def __init__(self, in_c=1, hidden=HIDDEN, emb_dim=EMB_DIM,
                 d_bottle=D_BOTTLE, max_depth=MAX_LEVEL):
        super().__init__()
        self.encoder  = CompressedEncoder(in_c, hidden, emb_dim, d_bottle, max_depth)
        self.decoder  = AutoregressiveDecoder(hidden, emb_dim, in_c, d_bottle, max_depth)
        self.d_bottle = d_bottle
        self.emb_dim  = emb_dim

    def forward(self, td: TreeData):
        z                    = self.encoder(td)
        val_pred, split_pred = self.decoder(z, td)
        return z, val_pred, split_pred

    @property
    def max_bottleneck_floats(self):
        return (4 ** self.d_bottle) * self.emb_dim

    @torch.no_grad()
    def infer(self, td: TreeData, split_thresh=0.0):
        """
        Autoregressive inference WITHOUT teacher forcing.
        Builds the predicted tree top-down by thresholding split logits.
        Returns (pred_keys_by_depth, val_pred, split_pred).
        """
        self.eval()
        dev      = td.device
        z        = self.encoder(td)
        N_bottle = z.shape[0]
        hd_cur   = self.decoder.bottle_token.expand(N_bottle, -1).to(dev)

        pred_keys = [None] * (MAX_LEVEL + 1)
        val_preds = [None] * (MAX_LEVEL + 1)

        # Bottle keys come from the actual tree (encoder saw them)
        cur_keys = td.keys[D_BOTTLE].clone()

        for d in range(D_BOTTLE, MAX_LEVEL + 1):
            pred_keys[d] = cur_keys
            Nd           = len(cur_keys)

            pos  = fourier_encode(node_centers(cur_keys, d, dev))
            skip = self.decoder.skip_norm(z) if d == D_BOTTLE else \
                   torch.zeros((Nd, EMB_DIM), device=dev)

            hd = self.decoder.fuse(torch.cat([hd_cur, skip, pos], dim=1))

            # Build live neighs for the predicted keys
            live_neighs = build_neighs(cur_keys, d, dev)
            hd          = F.relu(self.decoder.mix_convs[d](hd, live_neighs))

            val_preds[d] = self.decoder.val_head(hd)

            if d == MAX_LEVEL:
                break

            split_logits = self.decoder.split_head(hd).squeeze(-1)

            # Force-split below MIN_LEVEL (mirrors data generation)
            if d < MIN_LEVEL:
                split_mask = torch.ones(Nd, dtype=torch.bool, device=dev)
            else:
                split_mask = split_logits > split_thresh

            keys_to_split = cur_keys[split_mask]
            if keys_to_split.numel() == 0:
                # No more splits — remaining depths stay empty
                for dd in range(d + 1, MAX_LEVEL + 1):
                    pred_keys[dd] = torch.empty((0,), dtype=torch.long, device=dev)
                    val_preds[dd] = torch.empty((0, 1), device=dev)
                break

            child_keys_all = (keys_to_split.unsqueeze(1) << 2) + \
                              torch.arange(4, device=dev).view(1, 4)
            next_keys = torch.unique(child_keys_all.view(-1), sorted=True)

            # Build h for next depth via child_head
            h_next    = torch.zeros((len(next_keys), HIDDEN), device=dev)
            cf        = self.decoder.child_head(hd[split_mask]).view(-1, 4, HIDDEN)
            for t, pk in enumerate(keys_to_split):
                child4 = (pk << 2) + torch.arange(4, device=dev)
                for c in range(4):
                    ci = torch.searchsorted(next_keys, child4[c]).clamp(0, len(next_keys)-1)
                    if next_keys[ci] == child4[c]:
                        h_next[ci] = cf[t, c]

            cur_keys = next_keys
            hd_cur   = h_next

        return pred_keys, val_preds


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Loss
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(td, val_pred, split_pred, mse, bce, split_w=0.5):
    """
    Reconstruction loss only over depths D_BOTTLE … MAX_LEVEL.
    val loss   : MSE at leaf nodes
    split loss : BCE on split decisions at non-leaf nodes
    """
    dev     = td.device
    L_val   = torch.tensor(0.0, device=dev)
    L_split = torch.tensor(0.0, device=dev)
    n_val   = n_split = 0

    for d in range(D_BOTTLE, MAX_LEVEL + 1):
        mask = td.leaf_mask[d]
        if mask is None or not mask.any():
            continue
        if val_pred[d] is None:
            continue
        L_val += mse(val_pred[d][mask], td.values_gt[d][mask])
        n_val += 1

    for d in range(D_BOTTLE, MAX_LEVEL):
        sg = td.split_gt[d]
        sp = split_pred[d]
        if sg is None or sp is None or sg.numel() == 0:
            continue
        L_split += bce(sp, sg)
        n_split += 1

    if n_val   > 0: L_val   /= n_val
    if n_split > 0: L_split /= n_split
    return L_val + split_w * L_split, L_val, L_split


# ══════════════════════════════════════════════════════════════════════════════
# 10.  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _draw_leaves(ax, leaves, val_fn, title, cmap):
    vals = [val_fn(nd) for nd in leaves]
    lo, hi = min(vals), max(vals)
    rng = max(hi - lo, 1e-8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"{title}\n[{lo:.3f}, {hi:.3f}]", fontsize=9)
    for nd, v in zip(leaves, vals):
        col = cmap(np.clip((v - lo) / rng, 0, 1))
        ax.add_patch(patches.Rectangle(
            (nd.x, nd.y), nd.size, nd.size, lw=0.3, ec='k', fc=col))


def _draw_pred(ax, td, val_pred, title, cmap, vmin=None, vmax=None):
    parts = []
    for d in range(MAX_LEVEL + 1):
        if td.leaf_mask[d] is not None and td.leaf_mask[d].any() and val_pred[d] is not None:
            parts.append(val_pred[d][td.leaf_mask[d]].detach().cpu().flatten())
    if not parts:
        ax.axis('off'); ax.set_title(title); return
    all_v = torch.cat(parts).numpy()
    lo  = all_v.min() if vmin is None else vmin
    hi  = all_v.max() if vmax is None else vmax
    rng = max(hi - lo, 1e-8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"{title}\n[{lo:.3f}, {hi:.3f}]", fontsize=9)
    for d in range(MAX_LEVEL + 1):
        if td.leaf_mask[d] is None or not td.leaf_mask[d].any():
            continue
        if val_pred[d] is None:
            continue
        mask = td.leaf_mask[d]
        kd   = td.keys[d][mask].cpu()
        vals = val_pred[d][mask].detach().cpu().flatten().numpy()
        ix, iy = Morton2D.key2xy(kd)
        res  = 1 << d; sz = 1.0 / res
        for i in range(len(kd)):
            col = cmap(np.clip((vals[i] - lo) / rng, 0, 1))
            ax.add_patch(patches.Rectangle(
                (ix[i].item() * sz, iy[i].item() * sz),
                sz, sz, lw=0.3, ec='k', fc=col))


def plot_diagnostics(model, td, leaves, k1, k2, history, out_dir, step):
    """
    2-row × 3-col figure:
      [0,0] GT f          [0,1] Predicted f    [0,2] Squared-error map
      [1,0] Loss curves   [1,1] MSE per depth   [1,2] Leaf-error histogram
    """
    model.eval()
    with torch.no_grad():
        z, val_pred, _ = model(td)

    cmap     = plt.get_cmap('viridis')
    err_cmap = plt.get_cmap('hot')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ── row 0 ────────────────────────────────────────────────────────────────
    _draw_leaves(axes[0, 0], leaves, lambda nd: nd.f_val,
                 f"GT f  (k1={k1}, k2={k2}, leaves={len(leaves)})", cmap)

    _draw_pred(axes[0, 1], td, val_pred, "Predicted f (denoised)", cmap)

    # Build per-leaf errors for the error map and histogram
    err_map = {}   # (d, ix, iy) → sq_error
    all_sq  = []
    for d in range(MAX_LEVEL + 1):
        mask = td.leaf_mask[d]
        if mask is None or not mask.any() or val_pred[d] is None:
            continue
        kd = td.keys[d][mask].cpu()
        pr = val_pred[d][mask].detach().cpu().flatten()
        gt = td.values_gt[d][mask].cpu().flatten()
        sq = (pr - gt).pow(2).numpy()
        ix, iy = Morton2D.key2xy(kd)
        for i in range(len(kd)):
            err_map[(d, int(ix[i]), int(iy[i]))] = float(sq[i])
        all_sq.extend(sq.tolist())

    max_err = max(all_sq) if all_sq else 1e-6
    ax_err  = axes[0, 2]
    ax_err.set_xlim(0, 1); ax_err.set_ylim(0, 1)
    ax_err.set_aspect('equal'); ax_err.axis('off')
    ax_err.set_title(f"Squared Error Map  [0, {max_err:.5f}]", fontsize=9)
    for nd in leaves:
        d = nd.level; res = 1 << d
        ix = max(0, min(res-1, int(nd.x * res)))
        iy = max(0, min(res-1, int(nd.y * res)))
        sq  = err_map.get((d, ix, iy), 0.0)
        col = err_cmap(np.clip(sq / max(max_err, 1e-10), 0, 1))
        ax_err.add_patch(patches.Rectangle(
            (nd.x, nd.y), nd.size, nd.size, lw=0.3, ec='k', fc=col))

    # ── row 1 ────────────────────────────────────────────────────────────────
    ax_lc = axes[1, 0]
    if history['total']:
        ax_lc.semilogy(history['total'],  color='navy',      lw=1.5, label='Total')
        ax_lc.semilogy(history['val'],    color='steelblue', lw=1.0, alpha=0.8, label='Value')
        ax_lc.semilogy(history['split'],  color='coral',     lw=1.0, alpha=0.8, label='Split')
        ax_lc.legend(fontsize=8)
    ax_lc.set_title('Loss Curves (log scale)', fontsize=9)
    ax_lc.set_xlabel('Step'); ax_lc.grid(True, alpha=0.3)

    depth_mse = {}
    for d in range(MAX_LEVEL + 1):
        mask = td.leaf_mask[d]
        if mask is None or not mask.any() or val_pred[d] is None:
            continue
        pr = val_pred[d][mask].detach().cpu().flatten()
        gt = td.values_gt[d][mask].cpu().flatten()
        depth_mse[d] = float((pr - gt).pow(2).mean())

    ax_db = axes[1, 1]
    if depth_mse:
        ds = list(depth_mse.keys()); vs = [depth_mse[d] for d in ds]
        bars = ax_db.bar(ds, vs, color='steelblue', edgecolor='black')
        for b, v in zip(bars, vs):
            ax_db.text(b.get_x() + b.get_width()/2, b.get_height(),
                       f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax_db.set_title('Avg MSE per Depth', fontsize=9)
    ax_db.set_xticks(range(MAX_LEVEL + 1)); ax_db.grid(True, axis='y', alpha=0.3)

    ax_hist = axes[1, 2]
    if all_sq:
        ax_hist.hist(all_sq, bins=40, color='coral', alpha=0.75, edgecolor='k')
        ax_hist.axvline(np.mean(all_sq), color='red', ls='--',
                        label=f'mean = {np.mean(all_sq):.5f}')
        ax_hist.legend(fontsize=8)
    ax_hist.set_title('Per-Leaf Squared Error Histogram', fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    n_bottle = z.shape[0]
    fig.suptitle(
        f"Compressed Quadtree AE — Step {step}  |  "
        f"D_bottle={D_BOTTLE}  N_bottle={n_bottle}/{4**D_BOTTLE}  "
        f"Latent={n_bottle * EMB_DIM} floats  (max {4**D_BOTTLE * EMB_DIM})",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/diag_{step:05d}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    model.train()


def _draw_ar_pred(ax, pred_keys, val_preds, title, cmap):
    """Draw autoregressive predictions using their own key sets (all nodes are leaves at their depth)."""
    parts = []
    for d in range(MAX_LEVEL + 1):
        if pred_keys[d] is not None and pred_keys[d].numel() > 0 and val_preds[d] is not None and val_preds[d].numel() > 0:
            # In AR inference, the deepest nodes at each branch are leaves
            # For simplicity, draw all nodes that have no children in pred_keys
            has_children = False
            if d < MAX_LEVEL and pred_keys[d + 1] is not None and pred_keys[d + 1].numel() > 0:
                parent_of_next = pred_keys[d + 1] >> 2
                has_children = True
                is_leaf = ~torch.isin(pred_keys[d], parent_of_next)
            else:
                is_leaf = torch.ones(len(pred_keys[d]), dtype=torch.bool)
            if is_leaf.any():
                parts.append((d, pred_keys[d][is_leaf].cpu(), val_preds[d][is_leaf].detach().cpu().flatten()))
    if not parts:
        ax.axis('off'); ax.set_title(title); return
    all_v = torch.cat([p[2] for p in parts]).numpy()
    lo, hi = all_v.min(), all_v.max()
    rng = max(hi - lo, 1e-8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"{title}\n[{lo:.3f}, {hi:.3f}]", fontsize=9)
    for d, kd, vals in parts:
        ix, iy = Morton2D.key2xy(kd)
        res = 1 << d; sz = 1.0 / res
        for i in range(len(kd)):
            col = cmap(np.clip((vals[i].item() - lo) / rng, 0, 1))
            ax.add_patch(patches.Rectangle(
                (ix[i].item() * sz, iy[i].item() * sz),
                sz, sz, lw=0.3, ec='k', fc=col))


def plot_infer_comparison(model, td, leaves, k1, k2, out_dir, step):
    """
    Side-by-side: GT | teacher-forced pred | free-running pred
    Demonstrates that the AE also works autoregressively at inference.
    """
    model.eval()
    with torch.no_grad():
        _, val_tf, _ = model(td)                      # teacher-forced
        pred_keys, val_ar = model.infer(td)            # free-running

    cmap  = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    _draw_leaves(axes[0], leaves, lambda nd: nd.f_val,
                 f"GT f  k1={k1} k2={k2}", cmap)
    _draw_pred(axes[1], td, val_tf,
               "Teacher-forced reconstruction", cmap)
    _draw_ar_pred(axes[2], pred_keys, val_ar,
                  "Free-running (autoregressive) reconstruction", cmap)

    fig.suptitle(f"Inference Comparison — Step {step}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/infer_{step:05d}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    model.train()


# ══════════════════════════════════════════════════════════════════════════════
# 11.  Training
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device           : {device}")
    print(f"Bottleneck depth : {D_BOTTLE}  →  max {4**D_BOTTLE} nodes  "
          f"×  {EMB_DIM} dims  =  {4**D_BOTTLE * EMB_DIM} floats")

    model = CompressedQuadtreeAE().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters       : {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=3000, eta_min=1e-5)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"plots/compressed_ae_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}\n")

    num_steps = 3000
    patience  = 200
    split_w   = 0.5

    history   = {'total': [], 'val': [], 'split': []}
    best      = {'loss': float('inf'), 'step': 0, 'counter': 0}
    best_path = f'{out_dir}/best_model.pt'

    for step in range(num_steps):
        # ── sample ──────────────────────────────────────────────────────────
        leaf_keys, f_clean, f_noisy, leaves, k1, k2 = generate_f_tree()
        td = TreeData(leaf_keys, f_noisy, f_clean, device)

        # ── forward ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        z, val_pred, split_pred = model(td)
        L_total, L_val, L_split = compute_loss(
            td, val_pred, split_pred, mse, bce, split_w)
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history['total'].append(L_total.item())
        history['val'].append(L_val.item())
        history['split'].append(L_split.item())

        # ── checkpoint ──────────────────────────────────────────────────────
        if L_total.item() < best['loss']:
            best.update({'loss': L_total.item(), 'step': step, 'counter': 0})
            torch.save({
                'step': step, 'loss': L_total.item(),
                'd_bottle': D_BOTTLE, 'emb_dim': EMB_DIM,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_path)
        else:
            best['counter'] += 1

        # ── logging ─────────────────────────────────────────────────────────
        if step % 10 == 0:
            print(f"Step {step:4d} | "
                  f"total={L_total.item():.5f}  "
                  f"val={L_val.item():.5f}  "
                  f"split={L_split.item():.5f} | "
                  f"best={best['loss']:.5f}@{best['step']}  "
                  f"patience={best['counter']}/{patience} | "
                  f"k1={k1} k2={k2}  leaves={len(leaves)}  "
                  f"bottle_nodes={z.shape[0]}")

        # ── visualise ───────────────────────────────────────────────────────
        if step % 200 == 0 or step == num_steps - 1:
            plot_diagnostics(model, td, leaves, k1, k2, history, out_dir, step)
            plot_infer_comparison(model, td, leaves, k1, k2, out_dir, step)
            print(f"  → plots saved at step {step}")

        if best['counter'] >= patience:
            print(f"\nEarly stop at step {step}  (stalled for {patience} steps).")
            break

    print(f"\nDone.")
    print(f"Best loss : {best['loss']:.6f}  at step {best['step']}")
    print(f"Checkpoint: {best_path}")
    print(f"Bottleneck: {4**D_BOTTLE} nodes × {EMB_DIM} dims "
          f"= {4**D_BOTTLE * EMB_DIM} floats")


if __name__ == '__main__':
    main()
