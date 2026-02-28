"""
validate_poisson_ae.py

Validates BOTH autoencoders (model_f and model_u) trained by train_poisson_ae.py.

For each sample it produces an 8-panel figure:
  Row 0 (f): GT_f | Pred_f | LossMap_f | AvgLoss_f_by_depth
  Row 1 (u): GT_u | Pred_u | LossMap_u | AvgLoss_u_by_depth

It also:
  - Prints per-depth embedding stats for both AEs
  - Saves raw embedding vectors to .txt files
  - Produces a combined summary loss-by-depth plot

Usage:
    python validate_poisson_ae.py \
        --model_f path/to/best_model_f.pt \
        --model_u path/to/best_model_u.pt
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from datetime import datetime

# ==========================================
# 1. Morton Code Utilities  (copied verbatim)
# ==========================================

class Morton2D:
    @staticmethod
    def _interleave_bits(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long)
        x = x.long()
        x = x & 0x0000FFFF
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        return x

    @staticmethod
    def _deinterleave_bits(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long)
        x = x.long()
        x = x & 0x55555555
        x = (x | (x >> 1)) & 0x33333333
        x = (x | (x >> 2)) & 0x0F0F0F0F
        x = (x | (x >> 4)) & 0x00FF00FF
        x = (x | (x >> 8)) & 0x0000FFFF
        return x

    @staticmethod
    def xy2key(x, y, depth=16):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.long)
        if not torch.is_tensor(y): y = torch.tensor(y, dtype=torch.long)
        x = x.long(); y = y.long()
        kx = Morton2D._interleave_bits(x)
        ky = Morton2D._interleave_bits(y)
        return (kx | (ky << 1)).long()

    @staticmethod
    def key2xy(key, depth=16):
        if not torch.is_tensor(key): key = torch.tensor(key, dtype=torch.long)
        key = key.long()
        x = Morton2D._deinterleave_bits(key)
        y = Morton2D._deinterleave_bits(key >> 1)
        return x.long(), y.long()


# ==========================================
# 2. Positional Encoding
# ==========================================

def node_centers_from_keys(keys, depth, max_depth, device=None):
    if device is None: device = keys.device
    if keys.numel() == 0: return torch.zeros((0, 3), device=device)
    ix, iy = Morton2D.key2xy(keys, depth=depth)
    res = float(1 << depth)
    x = (ix.float() + 0.5) / res
    y = (iy.float() + 0.5) / res
    dnorm = torch.full_like(x, float(depth) / float(max_depth))
    return torch.stack([x, y, dnorm], dim=1)


def fourier_encode(pos, num_freqs=6):
    if pos.numel() == 0: return pos
    freqs = (2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)).view(1, 1, -1)
    x = pos.unsqueeze(-1) * np.pi * 2.0 * freqs
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    enc = enc.view(pos.shape[0], -1)
    return torch.cat([pos, enc], dim=1)


# ==========================================
# 3. Quadtree
# ==========================================

class Quadtree:
    def __init__(self, max_depth, device='cpu'):
        self.max_depth = max_depth
        self.device = device
        self.keys         = [None] * (max_depth + 1)
        self.neighs       = [None] * (max_depth + 1)
        self.features_in  = [None] * (max_depth + 1)
        self.children_idx = [None] * (max_depth + 1)
        self.parent_idx   = [None] * (max_depth + 1)
        self.split_gt     = [None] * (max_depth + 1)
        self.leaf_mask    = [None] * (max_depth + 1)
        self.values_gt    = [None] * (max_depth + 1)

    def build_from_leaves(self, leaf_keys_by_depth, leaf_vals_input_by_depth,
                          leaf_vals_target_by_depth=None):
        if leaf_vals_target_by_depth is None:
            leaf_vals_target_by_depth = leaf_vals_input_by_depth

        C_in = None
        for d in range(self.max_depth + 1):
            if leaf_vals_input_by_depth[d].numel() > 0:
                C_in = leaf_vals_input_by_depth[d].shape[1]; break
        if C_in is None: C_in = 1

        for d in range(self.max_depth + 1):
            lk = leaf_keys_by_depth[d].to(self.device).long()
            lv = leaf_vals_input_by_depth[d].to(self.device).float()
            if lk.numel() == 0:
                self.keys[d] = torch.empty((0,), dtype=torch.long, device=self.device)
                self.features_in[d] = torch.zeros((0, C_in), dtype=torch.float, device=self.device)
            else:
                lk_unique, inv = torch.unique(lk, sorted=True, return_inverse=True)
                self.keys[d] = lk_unique
                feat = torch.zeros((len(lk_unique), C_in), device=self.device)
                cnt  = torch.zeros((len(lk_unique), 1), device=self.device)
                feat.index_add_(0, inv, lv)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=self.device))
                self.features_in[d] = feat / cnt.clamp(min=1)

        if self.keys[0].numel() == 0:
            self.keys[0] = torch.tensor([0], dtype=torch.long, device=self.device)
            self.features_in[0] = torch.zeros((1, C_in), dtype=torch.float, device=self.device)

        for d in range(self.max_depth, 0, -1):
            if self.keys[d].numel() == 0: continue
            parents = torch.unique(self.keys[d] >> 2, sorted=True)
            self.keys[d-1] = torch.unique(torch.cat([self.keys[d-1], parents]), sorted=True)

        for d in range(self.max_depth + 1):
            kd = self.keys[d]
            feat = torch.zeros((len(kd), C_in), device=self.device)
            lk = leaf_keys_by_depth[d].to(self.device).long()
            lv = leaf_vals_input_by_depth[d].to(self.device).float()
            if lk.numel() > 0:
                lk_unique, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled = torch.zeros((len(lk_unique), C_in), device=self.device)
                cnt = torch.zeros((len(lk_unique), 1), device=self.device)
                pooled.index_add_(0, inv, lv)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=self.device))
                pooled = pooled / cnt.clamp(min=1)
                idx = torch.searchsorted(kd, lk_unique).clamp(0, len(kd)-1)
                found = kd[idx] == lk_unique
                feat[idx[found]] = pooled[found]
            self.features_in[d] = feat

        for d in range(self.max_depth):
            kd = self.keys[d]; kn = self.keys[d+1]
            if kd.numel() == 0:
                self.children_idx[d] = torch.empty((0, 4), dtype=torch.long, device=self.device); continue
            if kn.numel() == 0:
                self.children_idx[d] = torch.full((len(kd), 4), -1, dtype=torch.long, device=self.device); continue
            child_keys = (kd.unsqueeze(1) << 2) + torch.arange(4, device=self.device).view(1, 4)
            idx = torch.searchsorted(kn, child_keys).clamp(0, len(kn)-1)
            found = kn[idx] == child_keys
            self.children_idx[d] = torch.where(found, idx, torch.full_like(idx, -1))
        self.children_idx[self.max_depth] = None

        self.parent_idx[0] = None
        for d in range(1, self.max_depth + 1):
            kd = self.keys[d]; kp = self.keys[d-1]
            if kd.numel() == 0:
                self.parent_idx[d] = torch.empty((0,), dtype=torch.long, device=self.device); continue
            self.parent_idx[d] = torch.searchsorted(kp, kd >> 2).clamp(0, len(kp)-1)

        for d in range(self.max_depth + 1):
            self._construct_neigh(d)

        for d in range(self.max_depth + 1):
            kd = self.keys[d]
            if kd.numel() == 0:
                self.values_gt[d] = torch.zeros((0, C_in), device=self.device)
                self.leaf_mask[d] = torch.zeros((0,), dtype=torch.bool, device=self.device)
                self.split_gt[d] = None; continue

            target_feat = torch.zeros((len(kd), C_in), device=self.device)
            lk = leaf_keys_by_depth[d].to(self.device).long()
            lv_t = leaf_vals_target_by_depth[d].to(self.device).float()
            if lk.numel() > 0:
                lk_unique, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled = torch.zeros((len(lk_unique), C_in), device=self.device)
                cnt = torch.zeros((len(lk_unique), 1), device=self.device)
                pooled.index_add_(0, inv, lv_t)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=self.device))
                pooled = pooled / cnt.clamp(min=1)
                idx = torch.searchsorted(kd, lk_unique).clamp(0, len(kd)-1)
                found = kd[idx] == lk_unique
                target_feat[idx[found]] = pooled[found]
            self.values_gt[d] = target_feat

            if d == self.max_depth:
                self.leaf_mask[d] = torch.ones((len(kd),), dtype=torch.bool, device=self.device)
                self.split_gt[d] = None
            else:
                ch = self.children_idx[d]
                is_leaf = (ch == -1).all(dim=1)
                self.leaf_mask[d] = is_leaf
                self.split_gt[d] = (~is_leaf).float()

    def _construct_neigh(self, depth):
        keys = self.keys[depth]; N = len(keys)
        if N == 0:
            self.neighs[depth] = torch.empty((0, 9), dtype=torch.long, device=self.device); return
        if depth == 0:
            neigh = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
            neigh[:, 4] = 0; self.neighs[depth] = neigh; return
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor(
            [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],
            device=self.device, dtype=torch.long)
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        res = 1 << depth
        nx, ny = n_coords[...,0], n_coords[...,1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
        n_keys = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
        if valid.any():
            n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)
        idx = torch.searchsorted(keys, n_keys.clamp(min=0)).clamp(0, len(keys)-1)
        found = valid & (keys[idx] == n_keys)
        idx[~found] = -1
        self.neighs[depth] = idx


# ==========================================
# 4. Network (identical to training script)
# ==========================================

class QuadConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Linear(9 * in_channels, out_channels)

    def forward(self, features, quadtree, depth):
        neigh_idx = quadtree.neighs[depth]
        N = neigh_idx.shape[0]
        if N == 0:
            return torch.zeros((0, self.weights.out_features), device=features.device)
        pad_vec = torch.zeros((1, features.shape[1]), device=features.device)
        feat_padded = torch.cat([features, pad_vec], dim=0)
        gather_idx = neigh_idx.clone(); gather_idx[gather_idx == -1] = N
        return self.weights(feat_padded[gather_idx].view(N, -1))


class QuadPool(nn.Module):
    def forward(self, child_features, quadtree, depth_child):
        d_parent = depth_child - 1
        Np = len(quadtree.keys[d_parent]); C = child_features.shape[1]
        if Np == 0: return torch.zeros((0, C), device=child_features.device)
        ch = quadtree.children_idx[d_parent]
        pooled = torch.zeros((Np, C), device=child_features.device)
        cnt    = torch.zeros((Np, 1), device=child_features.device)
        for c in range(4):
            idx = ch[:, c]; mask = idx != -1
            if mask.any():
                pooled[mask] += child_features[idx[mask]]; cnt[mask] += 1.0
        return pooled / cnt.clamp(min=1.0)


class TreeEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=128, emb_dim=128, max_depth=7, pos_freqs=6):
        super().__init__()
        self.pos_freqs = pos_freqs; self.max_depth = max_depth
        self.hidden = hidden; self.emb_dim = emb_dim
        pos_dim = 3 + 2 * pos_freqs * 3
        self.in_proj = nn.Linear(in_c + pos_dim, hidden)
        self.convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool = QuadPool()
        self.to_emb  = nn.ModuleList([nn.Linear(hidden, emb_dim) for _ in range(max_depth + 1)])
        self.emb_norm = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(max_depth + 1)])
        self.depth_gain = nn.Parameter(torch.ones(max_depth + 1))

    def forward(self, qt):
        h = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            fin = qt.features_in[d]; kd = qt.keys[d]
            if fin is None or fin.numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=qt.device); continue
            pos = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, qt.device), self.pos_freqs)
            h[d] = self.in_proj(torch.cat([fin, pos], dim=1))
        for d in range(self.max_depth, 0, -1):
            if h[d].numel() == 0: continue
            pooled = self.pool(h[d], qt, d)
            h[d-1] = h[d-1] + pooled
            if d-1 >= 1 and h[d-1].numel() > 0:
                h[d-1] = F.relu(self.convs[d-1](h[d-1], qt, d-1))
        E = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            if h[d] is None or h[d].numel() == 0:
                E[d] = torch.zeros((0, self.emb_dim), device=qt.device)
            else:
                z = self.emb_norm[d](self.to_emb[d](h[d]))
                E[d] = self.depth_gain[d] * z
        return E


class TreeDecoder(nn.Module):
    def __init__(self, hidden=128, emb_dim=128, out_c=1, max_depth=7, pos_freqs=6):
        super().__init__()
        self.max_depth = max_depth; self.pos_freqs = pos_freqs
        self.hidden = hidden; self.emb_dim = emb_dim
        pos_dim = 3 + 2 * pos_freqs * 3
        self.root_token = nn.Parameter(torch.zeros(1, hidden))
        self.fuse = nn.Sequential(
            nn.Linear(hidden + emb_dim + pos_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.skip_norm  = nn.LayerNorm(emb_dim)
        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4 * hidden))
        self.val_head   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))
        self.mix_convs  = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

    def forward(self, qt, emb_list):
        h_by_depth = [None] * (self.max_depth + 1)
        N0 = len(qt.keys[0])
        h_by_depth[0] = self.root_token.expand(N0, -1).to(qt.device) if N0 > 0 else \
                        torch.zeros((0, self.hidden), device=qt.device)
        split_logits = [None] * self.max_depth
        val_pred     = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=qt.device)
                if d < self.max_depth: split_logits[d] = torch.zeros((0,), device=qt.device)
                continue
            kd = qt.keys[d]
            pos = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, qt.device), self.pos_freqs)
            if emb_list is not None and emb_list[d] is not None and emb_list[d].shape[0] == h.shape[0]:
                skip = self.skip_norm(emb_list[d].to(h.device))
            else:
                skip = torch.zeros((h.shape[0], self.emb_dim), device=h.device)
            h = self.fuse(torch.cat([h, skip, pos], dim=1))
            if d >= 1 and h.numel() > 0:
                h = F.relu(self.mix_convs[d](h, qt, d))
            val_pred[d] = self.val_head(h)
            if d == self.max_depth: break
            split_logits[d] = self.split_head(h).squeeze(-1)
            ch = qt.children_idx[d]; has_child = (ch != -1).any(dim=1)
            N_next = len(qt.keys[d+1])
            h_next = torch.zeros((N_next, self.hidden), device=h.device)
            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, self.hidden)
                parent_rows = torch.nonzero(has_child).squeeze(-1)
                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1: h_next[ci] = child_feats[t, c]
            h_by_depth[d+1] = h_next
        return split_logits, val_pred


class TreeAE(nn.Module):
    def __init__(self, in_c=1, hidden=128, emb_dim=128, max_depth=7, pos_freqs=6):
        super().__init__()
        self.encoder = TreeEncoder(in_c=in_c, hidden=hidden, emb_dim=emb_dim,
                                   max_depth=max_depth, pos_freqs=pos_freqs)
        self.decoder = TreeDecoder(hidden=hidden, emb_dim=emb_dim, out_c=in_c,
                                   max_depth=max_depth, pos_freqs=pos_freqs)

    def forward(self, qt):
        E = self.encoder(qt)
        split_logits, val_pred = self.decoder(qt, E)
        return split_logits, val_pred, E


# ==========================================
# 5. Poisson Data Generation (same as train)
# ==========================================

MAX_LEVEL = 7
MIN_LEVEL = 6
NOISE_STD = 0.05


def f_function(x, y, k1, k2):
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)


def u_function(x, y, k1, k2):
    denom = 4 * np.pi**2 * (k1**2 + k2**2)
    return f_function(x, y, k1, k2) / denom


def gradient_magnitude_f(x, y, k1, k2):
    dfdx = 2*np.pi*k1 * np.cos(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y)
    dfdy = 2*np.pi*k2 * np.sin(2*np.pi*k1*x) * np.cos(2*np.pi*k2*y)
    return np.sqrt(dfdx**2 + dfdy**2)


class QuadNode:
    def __init__(self, x, y, size, level, max_level, min_level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.max_level = max_level; self.min_level = min_level
        self.children = []; self.f_val = None; self.u_val = None

    def gradient_subdivide(self, k1, k2, grad_threshold=0.3):
        if self.level >= self.max_level: return
        cx = self.x + self.size / 2; cy = self.y + self.size / 2
        force_split = self.level < self.min_level
        grad_mag = gradient_magnitude_f(cx, cy, k1, k2)
        max_grad = 2 * np.pi * np.sqrt(k1**2 + k2**2)
        normalized_grad = grad_mag / max_grad
        depth_factor = 1.0 + 0.1 * self.level
        effective_threshold = grad_threshold * depth_factor
        if force_split or (normalized_grad > effective_threshold):
            half = self.size / 2
            self.children = [
                QuadNode(self.x,       self.y,       half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half,  self.y,       half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x,       self.y+half,  half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half,  self.y+half,  half, self.level+1, self.max_level, self.min_level),
            ]
            for child in self.children:
                child.gradient_subdivide(k1, k2, grad_threshold)

    def collect_leaves(self, leaves_list, k1, k2):
        if not self.children:
            cx = self.x + self.size / 2; cy = self.y + self.size / 2
            self.f_val = f_function(cx, cy, k1, k2)
            self.u_val = u_function(cx, cy, k1, k2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves(leaves_list, k1, k2)


def generate_poisson_pair(noise_std=NOISE_STD, grad_threshold=0.3):
    k1 = random.randint(1, 5); k2 = random.randint(1, 5)
    root = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root.gradient_subdivide(k1, k2, grad_threshold=grad_threshold)
    leaves = []; root.collect_leaves(leaves, k1, k2)

    leaf_keys  = [[] for _ in range(MAX_LEVEL + 1)]
    f_vals_raw = [[] for _ in range(MAX_LEVEL + 1)]
    u_vals_raw = [[] for _ in range(MAX_LEVEL + 1)]

    for node in leaves:
        d = node.level; res = 1 << d
        ix = max(0, min(res-1, int(node.x * res)))
        iy = max(0, min(res-1, int(node.y * res)))
        k  = Morton2D.xy2key(ix, iy)
        leaf_keys[d].append(int(k.item()))
        f_vals_raw[d].append([float(node.f_val)])
        u_vals_raw[d].append([float(node.u_val)])

    leaf_keys_by_depth    = []
    f_vals_clean_by_depth = []; f_vals_noisy_by_depth = []
    u_vals_clean_by_depth = []; u_vals_noisy_by_depth = []

    for d in range(MAX_LEVEL + 1):
        if len(leaf_keys[d]) == 0:
            leaf_keys_by_depth.append(torch.empty((0,), dtype=torch.long))
            f_vals_clean_by_depth.append(torch.empty((0,1))); f_vals_noisy_by_depth.append(torch.empty((0,1)))
            u_vals_clean_by_depth.append(torch.empty((0,1))); u_vals_noisy_by_depth.append(torch.empty((0,1)))
        else:
            keys_t  = torch.tensor(leaf_keys[d], dtype=torch.long)
            f_clean = torch.tensor(f_vals_raw[d], dtype=torch.float32)
            u_clean = torch.tensor(u_vals_raw[d], dtype=torch.float32)
            leaf_keys_by_depth.append(keys_t)
            f_vals_clean_by_depth.append(f_clean)
            f_vals_noisy_by_depth.append(f_clean + noise_std * torch.randn_like(f_clean))
            u_vals_clean_by_depth.append(u_clean)
            u_vals_noisy_by_depth.append(u_clean + noise_std * torch.randn_like(u_clean))

    return (leaf_keys_by_depth,
            f_vals_clean_by_depth, f_vals_noisy_by_depth,
            u_vals_clean_by_depth, u_vals_noisy_by_depth,
            leaves, k1, k2)


# ==========================================
# 6. Validation Utilities
# ==========================================

def collect_preds(model, qt):
    with torch.no_grad():
        _, val_pred, _ = model(qt)
    pred_dict = {}; loss_dict = {}
    loss_by_depth = {d: [] for d in range(MAX_LEVEL + 1)}
    for d in range(MAX_LEVEL + 1):
        if qt.keys[d].numel() == 0: continue
        mask = qt.leaf_mask[d]
        if mask is None or mask.numel() == 0 or not mask.any(): continue
        kd = qt.keys[d][mask].cpu()
        ix, iy = Morton2D.key2xy(kd, depth=d)
        pr = val_pred[d][mask].cpu().numpy().flatten()
        gt = qt.values_gt[d][mask].cpu().numpy().flatten()
        for i in range(len(kd)):
            key = (d, int(ix[i].item()), int(iy[i].item()))
            pred_dict[key] = pr[i]
            ll = (pr[i] - gt[i]) ** 2
            loss_dict[key] = ll; loss_by_depth[d].append(ll)
    return pred_dict, loss_dict, loss_by_depth


def _node_key(node):
    d = node.level; res = 1 << d
    ix = max(0, min(res-1, int(node.x * res)))
    iy = max(0, min(res-1, int(node.y * res)))
    return (d, ix, iy)


def draw_field(ax, leaves, val_fn, title, cmap):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=9)
    for nd in leaves:
        v = val_fn(nd)
        color = cmap(np.clip((v + 1) / 2, 0, 1))
        ax.add_patch(patches.Rectangle(
            (nd.x, nd.y), nd.size, nd.size,
            linewidth=0.3, edgecolor='black', facecolor=color))


def draw_pred(ax, leaves, pred_dict, title, cmap):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=9)
    for nd in leaves:
        v = pred_dict.get(_node_key(nd), 0.0)
        color = cmap(np.clip((v + 1) / 2, 0, 1))
        ax.add_patch(patches.Rectangle(
            (nd.x, nd.y), nd.size, nd.size,
            linewidth=0.3, edgecolor='black', facecolor=color))


def draw_loss_map(ax, leaves, loss_dict, max_loss, title, loss_cmap):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=9)
    for nd in leaves:
        ll = loss_dict.get(_node_key(nd), 0.0)
        color = loss_cmap(np.clip(ll / max_loss, 0, 1))
        ax.add_patch(patches.Rectangle(
            (nd.x, nd.y), nd.size, nd.size,
            linewidth=0.3, edgecolor='black', facecolor=color))


def draw_depth_bar(ax, loss_by_depth, title, color):
    depths = [d for d in range(MAX_LEVEL+1) if loss_by_depth[d]]
    avgs   = [np.mean(loss_by_depth[d]) for d in depths]
    if depths:
        bars = ax.bar(depths, avgs, color=color, edgecolor='black')
        for bar, v in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.5f}', ha='center', va='bottom', fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(MAX_LEVEL+1)); ax.grid(True, axis='y', alpha=0.3)


def print_embedding_stats(label, E_list, max_depth):
    print(f"  [{label}] embedding stats:")
    for d in range(max_depth + 1):
        e = E_list[d]
        if e is None or e.numel() == 0: continue
        print(f"    d={d}: N={e.shape[0]:4d}  mean={e.mean():+.4f}  "
              f"std={e.std():.4f}  min={e.min():+.3f}  max={e.max():+.3f}  "
              f"L2_mean={e.norm(dim=1).mean():.4f}")


def save_embedding_txt(filepath, qt, E_list, max_depth, label, num_nodes=10):
    with open(filepath, 'w') as f:
        f.write(f"{'='*72}\n")
        f.write(f"EMBEDDING DUMP  [{label}]\n")
        f.write(f"{'='*72}\n\n")
        for d in range(max_depth + 1):
            e = E_list[d]; keys = qt.keys[d]
            if e is None or e.numel() == 0:
                f.write(f"Depth {d}: empty\n\n"); continue
            data = e.detach().cpu().numpy()
            ix, iy = Morton2D.key2xy(keys.cpu(), depth=d)
            f.write(f"{'='*50}\n")
            f.write(f"DEPTH {d}: {len(data)} nodes, dim={data.shape[1]}\n")
            f.write(f"  min={data.min():.4f} max={data.max():.4f} "
                    f"mean={data.mean():.4f} std={data.std():.4f}\n\n")
            for ni in range(min(num_nodes, len(data))):
                vec = data[ni]
                f.write(f"  Node {ni} @ ({ix[ni].item()}, {iy[ni].item()}):\n")
                for cs in range(0, len(vec), 16):
                    ce = min(cs+16, len(vec))
                    vals = " ".join([f"{v:+.3f}" for v in vec[cs:ce]])
                    f.write(f"    [{cs:3d}-{ce-1:3d}]: {vals}\n")
                f.write(f"    -> L2={np.linalg.norm(vec):.4f}, "
                        f"active={int((np.abs(vec)>0.1).sum())}/{len(vec)}\n\n")
        f.write(f"{'='*72}\nEND\n")


# ==========================================
# 7. Main Validation Loop
# ==========================================

def validate(model_f_path, model_u_path, output_dir, num_samples=10, device='cpu'):
    print(f"\n{'='*64}")
    print(f"Validating Poisson AE pair")
    print(f"  model_f: {model_f_path}")
    print(f"  model_u: {model_u_path}")
    print(f"{'='*64}\n")

    # Load model_f
    ckpt_f = torch.load(model_f_path, map_location=device)
    model_f = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    model_f.load_state_dict(ckpt_f['model_state_dict'])
    model_f.eval()
    print(f"model_f loaded  — step={ckpt_f['step']}  total_loss={ckpt_f['total_loss']:.6f}")

    # Load model_u
    ckpt_u = torch.load(model_u_path, map_location=device)
    model_u = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    model_u.load_state_dict(ckpt_u['model_state_dict'])
    model_u.eval()
    print(f"model_u loaded  — step={ckpt_u['step']}  total_loss={ckpt_u['total_loss']:.6f}\n")

    os.makedirs(output_dir, exist_ok=True)

    cmap      = plt.get_cmap('viridis')
    loss_cmap = plt.get_cmap('hot')

    # Accumulate losses for summary plot
    all_lbd_f = {d: [] for d in range(MAX_LEVEL + 1)}
    all_lbd_u = {d: [] for d in range(MAX_LEVEL + 1)}

    with torch.no_grad():
        for i in range(num_samples):
            (leaf_keys, f_clean, f_noisy, u_clean, u_noisy,
             leaves, k1, k2) = generate_poisson_pair()

            qt_f = Quadtree(max_depth=MAX_LEVEL, device=device)
            qt_f.build_from_leaves(leaf_keys, f_noisy, f_clean)

            qt_u = Quadtree(max_depth=MAX_LEVEL, device=device)
            qt_u.build_from_leaves(leaf_keys, u_noisy, u_clean)

            # Encode to get embeddings
            E_f = model_f.encoder(qt_f)
            E_u = model_u.encoder(qt_u)

            print(f"Sample {i+1}/{num_samples}  k1={k1}  k2={k2}  leaves={len(leaves)}")
            print_embedding_stats('f', E_f, MAX_LEVEL)
            print_embedding_stats('u', E_u, MAX_LEVEL)
            print()

            # Collect predictions + losses
            pred_f, loss_f, lbd_f = collect_preds(model_f, qt_f)
            pred_u, loss_u, lbd_u = collect_preds(model_u, qt_u)

            max_lf = max(max(loss_f.values()) if loss_f else 1e-6, 1e-6)
            max_lu = max(max(loss_u.values()) if loss_u else 1e-6, 1e-6)

            for d in range(MAX_LEVEL + 1):
                if lbd_f[d]: all_lbd_f[d].extend(lbd_f[d])
                if lbd_u[d]: all_lbd_u[d].extend(lbd_u[d])

            # ==========================================
            # 8-panel figure: 2 rows x 4 cols
            #   Row 0 (f): GT_f | Pred_f | LossMap_f | AvgLoss_f/depth
            #   Row 1 (u): GT_u | Pred_u | LossMap_u | AvgLoss_u/depth
            # ==========================================
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))

            # --- f row ---
            draw_field(axes[0,0], leaves, lambda nd: nd.f_val,
                       f"GT f  (k1={k1}, k2={k2})", cmap)
            draw_pred (axes[0,1], leaves, pred_f, "Pred f (denoised)", cmap)
            draw_loss_map(axes[0,2], leaves, loss_f, max_lf, "Loss map f", loss_cmap)
            sm = plt.cm.ScalarMappable(cmap=loss_cmap, norm=plt.Normalize(0, max_lf))
            sm.set_array([]); plt.colorbar(sm, ax=axes[0,2], fraction=0.046, pad=0.04).set_label('MSE')
            draw_depth_bar(axes[0,3], lbd_f, "Avg loss/depth (f)", 'steelblue')

            # --- u row ---
            draw_field(axes[1,0], leaves, lambda nd: nd.u_val,
                       f"GT u  (analytic solution)", cmap)
            draw_pred (axes[1,1], leaves, pred_u, "Pred u (denoised)", cmap)
            draw_loss_map(axes[1,2], leaves, loss_u, max_lu, "Loss map u", loss_cmap)
            sm2 = plt.cm.ScalarMappable(cmap=loss_cmap, norm=plt.Normalize(0, max_lu))
            sm2.set_array([]); plt.colorbar(sm2, ax=axes[1,2], fraction=0.046, pad=0.04).set_label('MSE')
            draw_depth_bar(axes[1,3], lbd_u, "Avg loss/depth (u)", 'coral')

            plt.suptitle(
                f"Sample {i+1}  k1={k1}  k2={k2}  |  "
                f"f: step={ckpt_f['step']} loss={ckpt_f['total_loss']:.5f}  |  "
                f"u: step={ckpt_u['step']} loss={ckpt_u['total_loss']:.5f}",
                fontsize=11)
            plt.tight_layout()
            fig_path = f"{output_dir}/sample_{i+1:02d}_k{k1}_{k2}.png"
            plt.savefig(fig_path, dpi=140, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> {fig_path}")

            # Save raw embedding txt
            for label, qt, E in [('f', qt_f, E_f), ('u', qt_u, E_u)]:
                txt_path = f"{output_dir}/sample_{i+1:02d}_k{k1}_{k2}_emb_{label}.txt"
                save_embedding_txt(txt_path, qt, E, MAX_LEVEL, label, num_nodes=10)
                print(f"  -> {txt_path}")

    # ==========================================
    # Summary: avg loss by depth across all samples
    # ==========================================
    fig_sum, axes_sum = plt.subplots(1, 2, figsize=(16, 6))

    for ax, all_lbd, label, color in [
        (axes_sum[0], all_lbd_f, 'f', 'steelblue'),
        (axes_sum[1], all_lbd_u, 'u', 'coral'),
    ]:
        depths = [d for d in range(MAX_LEVEL+1) if all_lbd[d]]
        avgs   = [np.mean(all_lbd[d]) for d in depths]
        stds   = [np.std (all_lbd[d]) for d in depths]
        if depths:
            bars = ax.bar(depths, avgs, yerr=stds, color=color, edgecolor='black',
                          capsize=5, alpha=0.8)
            for bar, v, s in zip(bars, avgs, stds):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + s + 1e-6,
                        f'{v:.5f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'Avg loss/depth ({label})  across {num_samples} samples', fontsize=11)
        ax.set_xlabel('Depth'); ax.set_ylabel('Mean MSE')
        ax.set_xticks(range(MAX_LEVEL+1)); ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Poisson AE Validation — Summary', fontsize=13)
    plt.tight_layout()
    summary_path = f"{output_dir}/summary_loss_by_depth.png"
    plt.savefig(summary_path, dpi=140, bbox_inches='tight')
    plt.close(fig_sum)
    print(f"\n  -> Summary: {summary_path}")

    # Print overall stats
    print(f"\n{'='*64}")
    print("VALIDATION SUMMARY")
    print(f"{'='*64}")
    for label, all_lbd in [('f', all_lbd_f), ('u', all_lbd_u)]:
        all_losses = [v for d in range(MAX_LEVEL+1) for v in all_lbd[d]]
        if all_losses:
            print(f"  [{label}]  mean_leaf_MSE={np.mean(all_losses):.6f}  "
                  f"std={np.std(all_losses):.6f}  "
                  f"max={np.max(all_losses):.6f}")


# ==========================================
# 8. Entry Point
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_f', required=True, help='Path to best_model_f.pt')
    parser.add_argument('--model_u', required=True, help='Path to best_model_u.pt')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"plots/val_poisson_{timestamp}"

    device = args.device
    print(f"Device: {device}")

    validate(args.model_f, args.model_u, output_dir,
             num_samples=args.samples, device=device)
    print(f"\nAll outputs saved to: {output_dir}")
