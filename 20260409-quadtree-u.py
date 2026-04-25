"""
train_poisson_ae.py

Trains TWO independent autoencoders on paired Poisson data:
  - model_f: autoencoder for f (right-hand side)
  - model_u: autoencoder for u (solution, computed analytically)

PDE:  Laplacian(u) = f  on [0,1]^2, homogeneous Dirichlet BC
f(x,y)  = sin(2*pi*k1*x) * sin(2*pi*k2*y)
u(x,y)  = -f(x,y) / (4*pi^2*(k1^2 + k2^2))   [exact analytic solution]

f and u are discretised on SEPARATE adaptive quadtrees:
  - f's tree is refined by gradient of f
  - u's tree is refined by gradient of u
This allows each autoencoder to learn on its own optimal tree structure.

Key design:
  - generate_poisson_pair() returns separate (f_keys, f_vals, f_leaves) and (u_keys, u_vals, u_leaves)
  - model_f and model_u are independent TreeAE instances
  - trained simultaneously, separate optimizers, separate loss logs
  - best checkpoints saved separately: best_model_f.pt / best_model_u.pt
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

# ==========================================
# 1. Morton Code Utilities
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
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        x = x.long(); y = y.long()
        kx = Morton2D._interleave_bits(x)
        ky = Morton2D._interleave_bits(y)
        return (kx | (ky << 1)).long()

    @staticmethod
    def key2xy(key, depth=16):
        if not torch.is_tensor(key):
            key = torch.tensor(key, dtype=torch.long)
        key = key.long()
        x = Morton2D._deinterleave_bits(key)
        y = Morton2D._deinterleave_bits(key >> 1)
        return x.long(), y.long()


# ==========================================
# 2. Positional Encoding
# ==========================================

def node_centers_from_keys(keys, depth, max_depth, device=None):
    if device is None:
        device = keys.device
    if keys.numel() == 0:
        return torch.zeros((0, 3), device=device)
    ix, iy = Morton2D.key2xy(keys, depth=depth)
    res = float(1 << depth)
    x = (ix.float() + 0.5) / res
    y = (iy.float() + 0.5) / res
    dnorm = torch.full_like(x, float(depth) / float(max_depth))
    return torch.stack([x, y, dnorm], dim=1)


def fourier_encode(pos, num_freqs=6):
    if pos.numel() == 0:
        return pos
    freqs = (2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)).view(1, 1, -1)
    x = pos.unsqueeze(-1) * np.pi * 2.0 * freqs
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    enc = enc.view(pos.shape[0], -1)
    return torch.cat([pos, enc], dim=1)


# ==========================================
# 3. Quadtree Data Structure
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
                C_in = leaf_vals_input_by_depth[d].shape[1]
                break
        if C_in is None:
            C_in = 1

        # Initialize keys and features at leaf positions
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

        # Ensure ancestor closure
        for d in range(self.max_depth, 0, -1):
            if self.keys[d].numel() == 0:
                continue
            parents = torch.unique(self.keys[d] >> 2, sorted=True)
            self.keys[d-1] = torch.unique(torch.cat([self.keys[d-1], parents]), sorted=True)

        # Rebuild features_in for expanded keys
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

        # Build children_idx
        for d in range(self.max_depth):
            kd = self.keys[d]; kn = self.keys[d+1]
            if kd.numel() == 0:
                self.children_idx[d] = torch.empty((0, 4), dtype=torch.long, device=self.device)
                continue
            if kn.numel() == 0:
                self.children_idx[d] = torch.full((len(kd), 4), -1, dtype=torch.long, device=self.device)
                continue
            child_keys = (kd.unsqueeze(1) << 2) + torch.arange(4, device=self.device).view(1, 4)
            idx = torch.searchsorted(kn, child_keys).clamp(0, len(kn)-1)
            found = (kn[idx] == child_keys)
            self.children_idx[d] = torch.where(found, idx, torch.full_like(idx, -1))
        self.children_idx[self.max_depth] = None

        # Build parent_idx
        self.parent_idx[0] = None
        for d in range(1, self.max_depth + 1):
            kd = self.keys[d]; kp = self.keys[d-1]
            if kd.numel() == 0:
                self.parent_idx[d] = torch.empty((0,), dtype=torch.long, device=self.device)
                continue
            pidx = torch.searchsorted(kp, kd >> 2).clamp(0, len(kp)-1)
            self.parent_idx[d] = pidx

        # Build neighbors
        for d in range(self.max_depth + 1):
            self._construct_neigh(d)

        # Build GT labels
        for d in range(self.max_depth + 1):
            kd = self.keys[d]
            if kd.numel() == 0:
                self.values_gt[d] = torch.zeros((0, C_in), device=self.device)
                self.leaf_mask[d] = torch.zeros((0,), dtype=torch.bool, device=self.device)
                self.split_gt[d] = None
                continue

            target_feat = torch.zeros((len(kd), C_in), device=self.device)
            lk = leaf_keys_by_depth[d].to(self.device).long()
            lv_target = leaf_vals_target_by_depth[d].to(self.device).float()
            if lk.numel() > 0:
                lk_unique, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled = torch.zeros((len(lk_unique), C_in), device=self.device)
                cnt = torch.zeros((len(lk_unique), 1), device=self.device)
                pooled.index_add_(0, inv, lv_target)
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
        keys = self.keys[depth]
        N = len(keys)
        if N == 0:
            self.neighs[depth] = torch.empty((0, 9), dtype=torch.long, device=self.device)
            return
        if depth == 0:
            neigh = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
            neigh[:, 4] = 0
            self.neighs[depth] = neigh
            return
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
# 4. Network Modules
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
        gather_idx = neigh_idx.clone()
        gather_idx[gather_idx == -1] = N
        col = feat_padded[gather_idx].view(N, -1)
        return self.weights(col)


class QuadPool(nn.Module):
    def forward(self, child_features, quadtree, depth_child):
        d_parent = depth_child - 1
        Np = len(quadtree.keys[d_parent])
        C = child_features.shape[1]
        if Np == 0:
            return torch.zeros((0, C), device=child_features.device)
        ch = quadtree.children_idx[d_parent]
        pooled = torch.zeros((Np, C), device=child_features.device)
        cnt = torch.zeros((Np, 1), device=child_features.device)
        for c in range(4):
            idx = ch[:, c]; mask = idx != -1
            if mask.any():
                pooled[mask] += child_features[idx[mask]]
                cnt[mask] += 1.0
        return pooled / cnt.clamp(min=1.0)


class TreeEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=128, emb_dim=128, max_depth=7, pos_freqs=6):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.max_depth = max_depth
        self.hidden = hidden
        self.emb_dim = emb_dim
        pos_dim = 3 + 2 * pos_freqs * 3
        self.in_proj = nn.Linear(in_c + pos_dim, hidden)
        self.convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool = QuadPool()
        self.to_emb = nn.ModuleList([nn.Linear(hidden, emb_dim) for _ in range(max_depth + 1)])
        self.emb_norm = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(max_depth + 1)])
        self.depth_gain = nn.Parameter(torch.ones(max_depth + 1))

    def forward(self, qt):
        h = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            fin = qt.features_in[d]
            kd = qt.keys[d]
            if fin is None or fin.numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=qt.device)
                continue
            pos = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, qt.device), self.pos_freqs)
            h[d] = self.in_proj(torch.cat([fin, pos], dim=1))

        for d in range(self.max_depth, 0, -1):
            if h[d].numel() == 0:
                continue
            pooled = self.pool(h[d], qt, d)
            h[d-1] = h[d-1] + pooled
            if d-1 >= 1 and h[d-1].numel() > 0:
                h[d-1] = F.relu(self.convs[d-1](h[d-1], qt, d-1))

        E = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            if h[d] is None or h[d].numel() == 0:
                E[d] = torch.zeros((0, self.emb_dim), device=qt.device)
            else:
                z = self.to_emb[d](h[d])
                z = self.emb_norm[d](z)
                z = self.depth_gain[d] * z
                E[d] = z
        return E


class TreeDecoder(nn.Module):
    def __init__(self, hidden=128, emb_dim=128, out_c=1, max_depth=7, pos_freqs=6):
        super().__init__()
        self.max_depth = max_depth
        self.pos_freqs = pos_freqs
        self.hidden = hidden
        self.emb_dim = emb_dim
        pos_dim = 3 + 2 * pos_freqs * 3
        self.root_token = nn.Parameter(torch.zeros(1, hidden))
        self.fuse = nn.Sequential(
            nn.Linear(hidden + emb_dim + pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.skip_norm = nn.LayerNorm(emb_dim)
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
                if d < self.max_depth:
                    split_logits[d] = torch.zeros((0,), device=qt.device)
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

            if d == self.max_depth:
                break

            split_logits[d] = self.split_head(h).squeeze(-1)

            ch = qt.children_idx[d]
            has_child = (ch != -1).any(dim=1)
            N_next = len(qt.keys[d+1])
            h_next = torch.zeros((N_next, self.hidden), device=h.device)

            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, self.hidden)
                parent_rows = torch.nonzero(has_child).squeeze(-1)
                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1:
                            h_next[ci] = child_feats[t, c]

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

    @torch.no_grad()
    def encode(self, qt):
        self.encoder.eval()
        return self.encoder(qt)

    @torch.no_grad()
    def decode(self, qt, E):
        self.decoder.eval()
        return self.decoder(qt, E)


# ==========================================
# 5. Poisson Data Generation
# ==========================================

MAX_LEVEL = 7
MIN_LEVEL = 6
NOISE_STD = 0.05


def f_function(x, y, k1, k2):
    """Right-hand side: f = sin(2*pi*k1*x) * sin(2*pi*k2*y)"""
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)


def u_function(x, y, k1, k2):
    """
    Exact analytic solution to Laplacian(u) = f with homogeneous Dirichlet BC.
    u = -f / (4*pi^2*(k1^2 + k2^2))
    """
    denom = 4 * np.pi**2 * (k1**2 + k2**2)
    return -f_function(x, y, k1, k2) / denom


def gradient_magnitude_f(x, y, k1, k2):
    """Gradient magnitude of f, used to drive adaptive refinement."""
    dfdx = 2*np.pi*k1 * np.cos(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y)
    dfdy = 2*np.pi*k2 * np.sin(2*np.pi*k1*x) * np.cos(2*np.pi*k2*y)
    return np.sqrt(dfdx**2 + dfdy**2)


def gradient_magnitude_u(x, y, k1, k2):
    """Gradient magnitude of u, used to drive adaptive refinement for u's tree."""
    # u = -f / (4*pi^2*(k1^2 + k2^2)), so |grad(u)| = |grad(f)| / (4*pi^2*(k1^2 + k2^2))
    denom = 4 * np.pi**2 * (k1**2 + k2**2)
    dfdx = 2*np.pi*k1 * np.cos(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y)
    dfdy = 2*np.pi*k2 * np.sin(2*np.pi*k1*x) * np.cos(2*np.pi*k2*y)
    return np.sqrt(dfdx**2 + dfdy**2) / denom


class QuadNode:
    def __init__(self, x, y, size, level, max_level, min_level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.max_level = max_level
        self.min_level = min_level
        self.children = []
        self.f_val = None
        self.u_val = None

    def gradient_subdivide_f(self, k1, k2, grad_threshold=0.3):
        """Subdivide based on gradient of f."""
        if self.level >= self.max_level:
            return
        cx = self.x + self.size / 2
        cy = self.y + self.size / 2
        force_split = self.level < self.min_level
        grad_mag = gradient_magnitude_f(cx, cy, k1, k2)
        max_grad = 2 * np.pi * np.sqrt(k1**2 + k2**2)
        normalized_grad = grad_mag / max_grad
        depth_factor = 1.0 + 0.1 * self.level
        effective_threshold = grad_threshold * depth_factor
        if force_split or (normalized_grad > effective_threshold):
            half = self.size / 2
            self.children = [
                QuadNode(self.x,        self.y,        half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half,   self.y,        half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x,        self.y+half,   half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half,   self.y+half,   half, self.level+1, self.max_level, self.min_level),
            ]
            for child in self.children:
                child.gradient_subdivide_f(k1, k2, grad_threshold)

    def gradient_subdivide_u(self, k1, k2, grad_threshold=0.3):
        """Subdivide based on gradient of u."""
        if self.level >= self.max_level:
            return
        cx = self.x + self.size / 2
        cy = self.y + self.size / 2
        force_split = self.level < self.min_level
        grad_mag = gradient_magnitude_u(cx, cy, k1, k2)
        # max gradient of u = max_grad_f / denom
        denom = 4 * np.pi**2 * (k1**2 + k2**2)
        max_grad = 2 * np.pi * np.sqrt(k1**2 + k2**2) / denom
        normalized_grad = grad_mag / max_grad if max_grad > 0 else 0
        depth_factor = 1.0 + 0.1 * self.level
        effective_threshold = grad_threshold * depth_factor
        if force_split or (normalized_grad > effective_threshold):
            half = self.size / 2
            self.children = [
                QuadNode(self.x,        self.y,        half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half,   self.y,        half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x,        self.y+half,   half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half,   self.y+half,   half, self.level+1, self.max_level, self.min_level),
            ]
            for child in self.children:
                child.gradient_subdivide_u(k1, k2, grad_threshold)

    def collect_leaves_f(self, leaves_list, k1, k2):
        """Collect leaves and compute f values."""
        if not self.children:
            cx = self.x + self.size / 2
            cy = self.y + self.size / 2
            self.f_val = f_function(cx, cy, k1, k2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves_f(leaves_list, k1, k2)

    def collect_leaves_u(self, leaves_list, k1, k2):
        """Collect leaves and compute u values."""
        if not self.children:
            cx = self.x + self.size / 2
            cy = self.y + self.size / 2
            self.u_val = u_function(cx, cy, k1, k2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves_u(leaves_list, k1, k2)


def generate_poisson_pair(noise_std=NOISE_STD, grad_threshold=0.3):
    """
    Returns (f, u) data on SEPARATE adaptive quadtrees.
    
    f's tree is refined by gradient of f.
    u's tree is refined by gradient of u.

    Returns
    -------
    f_leaf_keys_by_depth   : list[Tensor]  f's tree topology
    f_vals_clean_by_depth  : list[Tensor]  clean f at leaves
    f_vals_noisy_by_depth  : list[Tensor]  noisy f (for denoising AE)
    f_leaves               : list[QuadNode] f's leaf nodes
    u_leaf_keys_by_depth   : list[Tensor]  u's tree topology
    u_vals_clean_by_depth  : list[Tensor]  clean u at leaves
    u_vals_noisy_by_depth  : list[Tensor]  noisy u (for denoising AE)
    u_leaves               : list[QuadNode] u's leaf nodes
    k1, k2                 : int
    """
    k1 = random.randint(1, 5)
    k2 = random.randint(1, 5)

    # --- Build f's tree (refined by gradient of f) ---
    root_f = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root_f.gradient_subdivide_f(k1, k2, grad_threshold=grad_threshold)
    f_leaves = []
    root_f.collect_leaves_f(f_leaves, k1, k2)

    # --- Build u's tree (refined by gradient of u) ---
    root_u = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root_u.gradient_subdivide_u(k1, k2, grad_threshold=grad_threshold)
    u_leaves = []
    root_u.collect_leaves_u(u_leaves, k1, k2)

    # --- Convert f leaves to tensors ---
    f_leaf_keys = [[] for _ in range(MAX_LEVEL + 1)]
    f_vals_raw  = [[] for _ in range(MAX_LEVEL + 1)]
    for node in f_leaves:
        d = node.level
        res = 1 << d
        ix = max(0, min(res-1, int(node.x * res)))
        iy = max(0, min(res-1, int(node.y * res)))
        k  = Morton2D.xy2key(ix, iy)
        f_leaf_keys[d].append(int(k.item()))
        f_vals_raw[d].append([float(node.f_val)])

    f_leaf_keys_by_depth    = []
    f_vals_clean_by_depth   = []
    f_vals_noisy_by_depth   = []
    for d in range(MAX_LEVEL + 1):
        if len(f_leaf_keys[d]) == 0:
            f_leaf_keys_by_depth.append(torch.empty((0,), dtype=torch.long))
            f_vals_clean_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
            f_vals_noisy_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
        else:
            keys_t  = torch.tensor(f_leaf_keys[d], dtype=torch.long)
            f_clean = torch.tensor(f_vals_raw[d], dtype=torch.float32)
            f_noisy = f_clean + noise_std * torch.randn_like(f_clean)
            f_leaf_keys_by_depth.append(keys_t)
            f_vals_clean_by_depth.append(f_clean)
            f_vals_noisy_by_depth.append(f_noisy)

    # --- Convert u leaves to tensors ---
    # Normalize u to [-1, 1] range by multiplying by denom (since u = f/denom and f in [-1,1])
    # This way both f and u are in the same [-1, 1] range for the network
    denom = 4 * np.pi**2 * (k1**2 + k2**2)
    
    u_leaf_keys = [[] for _ in range(MAX_LEVEL + 1)]
    u_vals_raw  = [[] for _ in range(MAX_LEVEL + 1)]
    for node in u_leaves:
        d = node.level
        res = 1 << d
        ix = max(0, min(res-1, int(node.x * res)))
        iy = max(0, min(res-1, int(node.y * res)))
        k  = Morton2D.xy2key(ix, iy)
        u_leaf_keys[d].append(int(k.item()))
        # Normalize u to [-1, 1] by multiplying by denom
        u_vals_raw[d].append([float(node.u_val) * denom])

    u_leaf_keys_by_depth    = []
    u_vals_clean_by_depth   = []
    u_vals_noisy_by_depth   = []
    for d in range(MAX_LEVEL + 1):
        if len(u_leaf_keys[d]) == 0:
            u_leaf_keys_by_depth.append(torch.empty((0,), dtype=torch.long))
            u_vals_clean_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
            u_vals_noisy_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
        else:
            keys_t  = torch.tensor(u_leaf_keys[d], dtype=torch.long)
            u_clean = torch.tensor(u_vals_raw[d], dtype=torch.float32)
            u_noisy = u_clean + noise_std * torch.randn_like(u_clean)  # Same noise as f
            u_leaf_keys_by_depth.append(keys_t)
            u_vals_clean_by_depth.append(u_clean)
            u_vals_noisy_by_depth.append(u_noisy)

    return (f_leaf_keys_by_depth, f_vals_clean_by_depth, f_vals_noisy_by_depth, f_leaves,
            u_leaf_keys_by_depth, u_vals_clean_by_depth, u_vals_noisy_by_depth, u_leaves,
            k1, k2, denom)  # Return denom for denormalization


# ==========================================
# 6. Training
# ==========================================

def compute_ae_loss(model, qt, device, mse, bce, split_weight=0.5):
    """Compute total AE loss for one model on one quadtree."""
    split_logits, val_pred, E = model(qt)

    L_val = torch.tensor(0.0, device=device)
    n_val = 0
    for d in range(MAX_LEVEL + 1):
        mask = qt.leaf_mask[d]
        if mask is None or mask.numel() == 0 or not mask.any():
            continue
        L_val = L_val + mse(val_pred[d][mask], qt.values_gt[d][mask])
        n_val += 1
    if n_val > 0:
        L_val = L_val / n_val

    L_split = torch.tensor(0.0, device=device)
    n_split = 0
    for d in range(MAX_LEVEL):
        if qt.split_gt[d] is None or qt.split_gt[d].numel() == 0:
            continue
        if split_logits[d] is None or split_logits[d].numel() == 0:
            continue
        L_split = L_split + bce(split_logits[d], qt.split_gt[d])
        n_split += 1
    if n_split > 0:
        L_split = L_split / n_split

    gain_reg = 1e-4 * (model.encoder.depth_gain ** 2).mean()
    L_total = L_val + split_weight * L_split + gain_reg
    return L_total, L_val, L_split


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # --- Two independent autoencoders ---
    model_f = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    model_u = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)

    opt_f = torch.optim.Adam(model_f.parameters(), lr=1e-3)
    opt_u = torch.optim.Adam(model_u.parameters(), lr=1e-3)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    split_weight = 0.5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"plots/poisson_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    num_steps = 3000
    patience   = 150

    # Loss history for f and u
    history = {
        'f_val': [], 'f_split': [], 'f_total': [],
        'u_val': [], 'u_split': [], 'u_total': [],
    }

    best_f = {'loss': float('inf'), 'step': 0, 'counter': 0}
    best_u = {'loss': float('inf'), 'step': 0, 'counter': 0}
    best_f_path = f'{output_dir}/best_model_f.pt'
    best_u_path = f'{output_dir}/best_model_u.pt'

    sample_buffer = []  # stores recent samples for plotting

    print(f"\nTraining two independent AEs on Poisson pairs for {num_steps} steps")
    print(f"  Laplacian(u) = f,  f = sin(2πk1x)sin(2πk2y),  u = -f / (4π²(k1²+k2²))")
    print(f"  Noise std: {NOISE_STD},  Split weight: {split_weight}")
    print(f"  Patience: {patience}\n")

    for step in range(num_steps):
        (f_leaf_keys, f_clean, f_noisy, f_leaves,
         u_leaf_keys, u_clean, u_noisy, u_leaves,
         k1, k2, denom) = generate_poisson_pair()

        # Build SEPARATE trees for f and u
        qt_f = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_f.build_from_leaves(f_leaf_keys, f_noisy, f_clean)   # input=noisy, target=clean

        qt_u = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_u.build_from_leaves(u_leaf_keys, u_noisy, u_clean)

        # --- Train model_f ---
        opt_f.zero_grad()
        L_f_total, L_f_val, L_f_split = compute_ae_loss(model_f, qt_f, device, mse, bce, split_weight)
        L_f_total.backward()
        opt_f.step()

        # --- Train model_u ---
        opt_u.zero_grad()
        L_u_total, L_u_val, L_u_split = compute_ae_loss(model_u, qt_u, device, mse, bce, split_weight)
        L_u_total.backward()
        opt_u.step()

        # Record history
        history['f_val'].append(L_f_val.item())
        history['f_split'].append(L_f_split.item())
        history['f_total'].append(L_f_total.item())
        history['u_val'].append(L_u_val.item())
        history['u_split'].append(L_u_split.item())
        history['u_total'].append(L_u_total.item())

        # Checkpointing model_f
        if L_f_total.item() < best_f['loss']:
            best_f.update({'loss': L_f_total.item(), 'step': step, 'counter': 0})
            torch.save({'step': step, 'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': opt_f.state_dict(),
                        'total_loss': L_f_total.item(),
                        'val_loss': L_f_val.item(),
                        'split_loss': L_f_split.item()}, best_f_path)
        else:
            best_f['counter'] += 1

        # Checkpointing model_u
        if L_u_total.item() < best_u['loss']:
            best_u.update({'loss': L_u_total.item(), 'step': step, 'counter': 0})
            torch.save({'step': step, 'model_state_dict': model_u.state_dict(),
                        'optimizer_state_dict': opt_u.state_dict(),
                        'total_loss': L_u_total.item(),
                        'val_loss': L_u_val.item(),
                        'split_loss': L_u_split.item()}, best_u_path)
        else:
            best_u['counter'] += 1

        print(
            f"Step {step:4d} | "
            f"f: val={L_f_val.item():.5f} split={L_f_split.item():.5f} total={L_f_total.item():.5f} "
            f"best={best_f['loss']:.5f}@{best_f['step']} p={best_f['counter']}/{patience} | "
            f"u: val={L_u_val.item():.5f} split={L_u_split.item():.5f} total={L_u_total.item():.5f} "
            f"best={best_u['loss']:.5f}@{best_u['step']} p={best_u['counter']}/{patience} | "
            f"k1={k1} k2={k2} f_leaves={len(f_leaves)} u_leaves={len(u_leaves)}"
        )

        # Early stop only when BOTH converged
        if best_f['counter'] >= patience and best_u['counter'] >= patience:
            print(f"\nEarly stopping at step {step}: both AEs stalled for {patience} steps.")
            break

        # Sample buffer for visualisation
        if step % 100 == 0:
            sample_buffer.append({
                'qt_f': qt_f, 'qt_u': qt_u,
                'f_leaves': f_leaves, 'u_leaves': u_leaves,
                'k1': k1, 'k2': k2, 'denom': denom, 'step': step
            })
            if len(sample_buffer) > 3:
                sample_buffer.pop(0)

        # Visualisation every 100 steps
        if step % 100 == 0 or step == num_steps - 1:
            _plot_samples(sample_buffer, model_f, model_u, output_dir, step)
            _plot_losses(history, output_dir, step, split_weight)

    print(f"\nTraining complete.")
    print(f"  Best model_f: step={best_f['step']}, loss={best_f['loss']:.6f}  ->  {best_f_path}")
    print(f"  Best model_u: step={best_u['step']}, loss={best_u['loss']:.6f}  ->  {best_u_path}")
    
    # Run validation on best models
    validate_models(best_f_path, best_u_path, output_dir, num_samples=10, device=device)


# ==========================================
# 7. Plotting Helpers
# ==========================================

def _collect_preds(model, qt):
    """Run model on qt and return (pred_dict, loss_dict, loss_by_depth)."""
    with torch.no_grad():
        _, val_pred, _ = model(qt)
    pred_dict = {}
    loss_dict = {}
    loss_by_depth = {d: [] for d in range(MAX_LEVEL + 1)}
    for d in range(MAX_LEVEL + 1):
        if qt.keys[d].numel() == 0:
            continue
        mask = qt.leaf_mask[d]
        if mask is None or mask.numel() == 0 or not mask.any():
            continue
        kd = qt.keys[d][mask].cpu()
        ix, iy = Morton2D.key2xy(kd, depth=d)
        pr = val_pred[d][mask].cpu().numpy().flatten()
        gt = qt.values_gt[d][mask].cpu().numpy().flatten()
        for i in range(len(kd)):
            key = (d, int(ix[i].item()), int(iy[i].item()))
            pred_dict[key] = pr[i]
            ll = (pr[i] - gt[i]) ** 2
            loss_dict[key] = ll
            loss_by_depth[d].append(ll)
    return pred_dict, loss_dict, loss_by_depth


def _draw_field(ax, leaves, val_fn, title, cmap, vmin=None, vmax=None):
    """
    Draw field with automatic or specified value range scaling.
    If vmin/vmax not provided, computes from data.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    
    # Compute values first to determine range
    values = [val_fn(node) for node in leaves]
    if vmin is None:
        vmin = min(values) if values else -1
    if vmax is None:
        vmax = max(values) if values else 1
    
    # Avoid division by zero
    val_range = vmax - vmin
    if val_range < 1e-10:
        val_range = 1.0
    
    ax.set_title(f"{title}\n[{vmin:.4f}, {vmax:.4f}]", fontsize=9)
    
    for node, v in zip(leaves, values):
        normalized = (v - vmin) / val_range
        color = cmap(np.clip(normalized, 0, 1))
        rect = patches.Rectangle((node.x, node.y), node.size, node.size,
                                  linewidth=0.3, edgecolor='black', facecolor=color)
        ax.add_patch(rect)


def _draw_loss_map(ax, leaves, loss_dict, max_loss, title, loss_cmap):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=9)
    for node in leaves:
        d = node.level; res = 1 << d
        ix = max(0, min(res-1, int(node.x * res)))
        iy = max(0, min(res-1, int(node.y * res)))
        ll = loss_dict.get((d, ix, iy), 0.0)
        color = loss_cmap(np.clip(ll / max_loss, 0, 1))
        rect = patches.Rectangle((node.x, node.y), node.size, node.size,
                                  linewidth=0.3, edgecolor='black', facecolor=color)
        ax.add_patch(rect)


def _plot_samples(sample_buffer, model_f, model_u, output_dir, step):
    """
    For each sample: 3 rows x 4 cols
      Row 0 (f): GT_f | Pred_f | LossMap_f | AvgLoss_f_by_depth
      Row 1 (u): GT_u | Pred_u | LossMap_u | AvgLoss_u_by_depth
      Row 2    : f_loss_hist | u_loss_hist | [blank] | [blank]
    """
    cmap = plt.get_cmap('viridis')
    loss_cmap = plt.get_cmap('hot')

    n = len(sample_buffer)
    if n == 0:
        return

    fig, axes = plt.subplots(n * 3, 4, figsize=(24, 9 * n))
    if n == 1:
        axes = axes.reshape(3, 4)

    for row_base, sample in enumerate(sample_buffer):
        qt_f   = sample['qt_f']
        qt_u   = sample['qt_u']
        f_leaves = sample['f_leaves']
        u_leaves = sample['u_leaves']
        k1, k2 = sample['k1'], sample['k2']
        denom = sample['denom']
        s = sample['step']

        pred_f, loss_f, lbd_f = _collect_preds(model_f, qt_f)
        pred_u, loss_u, lbd_u = _collect_preds(model_u, qt_u)

        max_lf = max(max(loss_f.values()) if loss_f else 1e-6, 1e-6)
        max_lu = max(max(loss_u.values()) if loss_u else 1e-6, 1e-6)

        r0 = row_base * 3      # f row
        r1 = row_base * 3 + 1  # u row
        r2 = row_base * 3 + 2  # histograms

        # ---- f row ----
        _draw_field(axes[r0, 0], f_leaves,
                    lambda nd: nd.f_val,
                    f"GT f  (step {s}, k1={k1} k2={k2}, leaves={len(f_leaves)})", cmap)

        _draw_field(axes[r0, 1], f_leaves,
                    lambda nd, pd=pred_f: pd.get(
                        (nd.level, max(0,min((1<<nd.level)-1,int(nd.x*(1<<nd.level)))),
                                   max(0,min((1<<nd.level)-1,int(nd.y*(1<<nd.level))))), 0.0),
                    "Pred f (denoised)", cmap)

        _draw_loss_map(axes[r0, 2], f_leaves, loss_f, max_lf, "Loss map f", loss_cmap)

        ax_d = axes[r0, 3]
        depths_f = [d for d in range(MAX_LEVEL+1) if lbd_f[d]]
        avgs_f   = [np.mean(lbd_f[d]) for d in depths_f]
        if depths_f:
            ax_d.bar(depths_f, avgs_f, color='steelblue', edgecolor='black')
            for i, (d, v) in enumerate(zip(depths_f, avgs_f)):
                ax_d.text(d, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
        ax_d.set_title("Avg loss/depth (f)", fontsize=9)
        ax_d.set_xticks(range(MAX_LEVEL+1)); ax_d.grid(True, axis='y', alpha=0.3)

        # ---- u row ----
        # Note: u is normalized to [-1,1] during training (u_normalized = u_original * denom)
        # So GT display should also be normalized for fair comparison
        _draw_field(axes[r1, 0], u_leaves,
                    lambda nd, d=denom: nd.u_val * d,  # Normalize GT to [-1,1]
                    f"GT u (normalized, leaves={len(u_leaves)})", cmap)

        _draw_field(axes[r1, 1], u_leaves,
                    lambda nd, pd=pred_u: pd.get(
                        (nd.level, max(0,min((1<<nd.level)-1,int(nd.x*(1<<nd.level)))),
                                   max(0,min((1<<nd.level)-1,int(nd.y*(1<<nd.level))))), 0.0),
                    "Pred u (normalized)", cmap)

        _draw_loss_map(axes[r1, 2], u_leaves, loss_u, max_lu, "Loss map u", loss_cmap)

        ax_d2 = axes[r1, 3]
        depths_u = [d for d in range(MAX_LEVEL+1) if lbd_u[d]]
        avgs_u   = [np.mean(lbd_u[d]) for d in depths_u]
        if depths_u:
            ax_d2.bar(depths_u, avgs_u, color='coral', edgecolor='black')
            for i, (d, v) in enumerate(zip(depths_u, avgs_u)):
                ax_d2.text(d, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
        ax_d2.set_title("Avg loss/depth (u)", fontsize=9)
        ax_d2.set_xticks(range(MAX_LEVEL+1)); ax_d2.grid(True, axis='y', alpha=0.3)

        # ---- histogram row ----
        all_lf = list(loss_f.values())
        all_lu = list(loss_u.values())

        for ax_h, data, label, color in [
            (axes[r2, 0], all_lf, 'f MSE distribution', 'steelblue'),
            (axes[r2, 1], all_lu, 'u MSE distribution', 'coral'),
        ]:
            if data:
                ax_h.hist(data, bins=40, color=color, alpha=0.75, edgecolor='black')
                ax_h.axvline(np.mean(data), color='red', linestyle='--',
                             label=f'mean={np.mean(data):.5f}')
                ax_h.legend(fontsize=8)
            ax_h.set_title(label, fontsize=9)
            ax_h.grid(True, alpha=0.3)

        axes[r2, 2].axis('off')
        axes[r2, 3].axis('off')

    plt.suptitle(f"Poisson AE — step {step}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/samples_step{step:04d}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {output_dir}/samples_step{step:04d}.png")


def _plot_losses(history, output_dir, step, split_weight):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    labels = [
        ('f_val',   'f Value Loss (MSE)',    'steelblue',  axes[0, 0]),
        ('f_split', 'f Split Loss (BCE)',    'cornflowerblue', axes[0, 1]),
        ('f_total', 'f Total Loss',          'navy',       axes[0, 2]),
        ('u_val',   'u Value Loss (MSE)',    'coral',      axes[1, 0]),
        ('u_split', 'u Split Loss (BCE)',    'salmon',     axes[1, 1]),
        ('u_total', 'u Total Loss',          'darkred',    axes[1, 2]),
    ]
    for key, title, color, ax in labels:
        ax.plot(history[key], linewidth=1, alpha=0.8, color=color)
        ax.set_xlabel('Step'); ax.set_ylabel('Loss')
        ax.set_title(title); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Training Losses — step {step}  (split weight={split_weight})', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/losses_step{step:04d}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {output_dir}/losses_step{step:04d}.png")


# ==========================================
# 8. Validation Function (Post-Training)
# ==========================================

def validate_models(model_f_path, model_u_path, output_dir, num_samples=10, device='cpu'):
    """
    Load best f and u models and generate validation plots with embedding visualization.
    Creates a 'validation' subfolder with detailed analysis for both models.
    """
    from sklearn.decomposition import PCA
    
    print(f"\n{'='*60}")
    print(f"Validating models with Embedding Analysis")
    print(f"{'='*60}")
    
    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(val_dir, exist_ok=True)
    
    # Load checkpoints
    ckpt_f = torch.load(model_f_path, map_location=device)
    ckpt_u = torch.load(model_u_path, map_location=device)
    print(f"Model f: step={ckpt_f['step']}, total_loss={ckpt_f['total_loss']:.6f}")
    print(f"Model u: step={ckpt_u['step']}, total_loss={ckpt_u['total_loss']:.6f}")
    
    # Load models
    model_f = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    model_u = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    model_f.load_state_dict(ckpt_f['model_state_dict'])
    model_u.load_state_dict(ckpt_u['model_state_dict'])
    model_f.eval()
    model_u.eval()
    
    cmap = plt.get_cmap('viridis')
    loss_cmap = plt.get_cmap('hot')
    
    # Collect avg loss by depth across ALL samples for summary
    all_samples_loss_f = {d: [] for d in range(MAX_LEVEL + 1)}
    all_samples_loss_u = {d: [] for d in range(MAX_LEVEL + 1)}
    
    def get_pca_colors(embeddings):
        """Projects (N, Hidden) -> (N, 3) for RGB visualization."""
        if embeddings.shape[0] < 3:
            return np.ones((embeddings.shape[0], 3)) * 0.5
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(embeddings.detach().cpu().numpy())
        p_min = pca_features.min(axis=0)
        p_max = pca_features.max(axis=0)
        rgb = (pca_features - p_min) / (p_max - p_min + 1e-6)
        return rgb
    
    def plot_embedding_spatial(ax, qt, keys, embeddings, depth):
        """Plots the spatial quadtree colored by PCA of embeddings."""
        rgb_values = get_pca_colors(embeddings)
        ix, iy = Morton2D.key2xy(keys.cpu(), depth=depth)
        res = 1 << depth
        size = 1.0 / res
        x_coords = ix.float() * size
        y_coords = iy.float() * size
        for k in range(len(keys)):
            rect = patches.Rectangle(
                (x_coords[k], y_coords[k]), size, size,
                linewidth=0.0, edgecolor=None, facecolor=rgb_values[k]
            )
            ax.add_patch(rect)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f"Embedding PCA (Depth {depth})", fontsize=8)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate data
            (f_leaf_keys, f_clean, f_noisy, f_leaves,
             u_leaf_keys, u_clean, u_noisy, u_leaves,
             k1, k2, denom) = generate_poisson_pair()
            
            qt_f = Quadtree(max_depth=MAX_LEVEL, device=device)
            qt_f.build_from_leaves(f_leaf_keys, f_noisy, f_clean)
            qt_u = Quadtree(max_depth=MAX_LEVEL, device=device)
            qt_u.build_from_leaves(u_leaf_keys, u_noisy, u_clean)
            
            # Run models
            _, val_pred_f, E_f = model_f(qt_f)
            _, val_pred_u, E_u = model_u(qt_u)
            
            # Collect predictions and losses for f
            pred_f, loss_f, lbd_f = {}, {}, {d: [] for d in range(MAX_LEVEL + 1)}
            for d in range(MAX_LEVEL + 1):
                if qt_f.keys[d].numel() == 0:
                    continue
                mask = qt_f.leaf_mask[d]
                if mask is None or not mask.any():
                    continue
                kd = qt_f.keys[d][mask].cpu()
                ix, iy = Morton2D.key2xy(kd, depth=d)
                pr = val_pred_f[d][mask].cpu().numpy().flatten()
                gt = qt_f.values_gt[d][mask].cpu().numpy().flatten()
                for j in range(len(kd)):
                    key = (d, int(ix[j].item()), int(iy[j].item()))
                    pred_f[key] = pr[j]
                    ll = (pr[j] - gt[j]) ** 2
                    loss_f[key] = ll
                    lbd_f[d].append(ll)
            
            # Collect predictions and losses for u
            pred_u, loss_u, lbd_u = {}, {}, {d: [] for d in range(MAX_LEVEL + 1)}
            for d in range(MAX_LEVEL + 1):
                if qt_u.keys[d].numel() == 0:
                    continue
                mask = qt_u.leaf_mask[d]
                if mask is None or not mask.any():
                    continue
                kd = qt_u.keys[d][mask].cpu()
                ix, iy = Morton2D.key2xy(kd, depth=d)
                pr = val_pred_u[d][mask].cpu().numpy().flatten()
                gt = qt_u.values_gt[d][mask].cpu().numpy().flatten()
                for j in range(len(kd)):
                    key = (d, int(ix[j].item()), int(iy[j].item()))
                    pred_u[key] = pr[j]
                    ll = (pr[j] - gt[j]) ** 2
                    loss_u[key] = ll
                    lbd_u[d].append(ll)
            
            print(f"  Sample {i+1}/{num_samples}: k1={k1}, k2={k2}, f_leaves={len(f_leaves)}, u_leaves={len(u_leaves)}")
            
            # Create 4x4 plot: 2 rows for f, 2 rows for u
            fig, axes = plt.subplots(4, 4, figsize=(24, 24))
            
            max_lf = max(max(loss_f.values()) if loss_f else 1e-6, 1e-6)
            max_lu = max(max(loss_u.values()) if loss_u else 1e-6, 1e-6)
            
            # --- Row 0: f GT, Pred, Loss Map, Avg Loss by Depth ---
            _draw_field(axes[0, 0], f_leaves, lambda nd: nd.f_val,
                        f"GT f (k1={k1}, k2={k2})", cmap)
            _draw_field(axes[0, 1], f_leaves,
                        lambda nd, pd=pred_f: pd.get(
                            (nd.level, max(0,min((1<<nd.level)-1,int(nd.x*(1<<nd.level)))),
                                       max(0,min((1<<nd.level)-1,int(nd.y*(1<<nd.level))))), 0.0),
                        "Pred f (denoised)", cmap)
            _draw_loss_map(axes[0, 2], f_leaves, loss_f, max_lf, "Loss map f", loss_cmap)
            
            ax_d = axes[0, 3]
            depths_f = [d for d in range(MAX_LEVEL+1) if lbd_f[d]]
            avgs_f = [np.mean(lbd_f[d]) for d in depths_f]
            if depths_f:
                ax_d.bar(depths_f, avgs_f, color='steelblue', edgecolor='black')
                for d, v in zip(depths_f, avgs_f):
                    ax_d.text(d, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
            ax_d.set_title("Avg loss/depth (f)", fontsize=9)
            ax_d.set_xticks(range(MAX_LEVEL+1)); ax_d.grid(True, axis='y', alpha=0.3)
            
            # --- Row 1: f Embedding PCA, Histogram, Loss Histogram ---
            viz_depth = 6
            emb_f = E_f[viz_depth]
            keys_f = qt_f.keys[viz_depth]
            if emb_f is not None and len(emb_f) > 0:
                plot_embedding_spatial(axes[1, 0], qt_f, keys_f, emb_f, viz_depth)
                data_f = emb_f.detach().cpu().numpy()
                axes[1, 1].hist(data_f.flatten(), bins=50, color='steelblue', alpha=0.7)
                axes[1, 1].set_title("f Latent Activations", fontsize=8)
                axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, "No embeddings", ha='center', va='center'); axes[1, 0].axis('off')
                axes[1, 1].axis('off')
            
            all_lf = list(loss_f.values())
            if all_lf:
                axes[1, 2].hist(all_lf, bins=40, color='steelblue', alpha=0.75, edgecolor='black')
                axes[1, 2].axvline(np.mean(all_lf), color='red', linestyle='--', label=f'mean={np.mean(all_lf):.5f}')
                axes[1, 2].legend(fontsize=8)
            axes[1, 2].set_title("f Loss Distribution", fontsize=9); axes[1, 2].grid(True, alpha=0.3)
            axes[1, 3].axis('off')
            
            # --- Row 2: u GT, Pred, Loss Map, Avg Loss by Depth ---
            # Note: u is normalized to [-1,1] during training
            _draw_field(axes[2, 0], u_leaves, lambda nd, d=denom: nd.u_val * d,
                        f"GT u (normalized)", cmap)
            _draw_field(axes[2, 1], u_leaves,
                        lambda nd, pd=pred_u: pd.get(
                            (nd.level, max(0,min((1<<nd.level)-1,int(nd.x*(1<<nd.level)))),
                                       max(0,min((1<<nd.level)-1,int(nd.y*(1<<nd.level))))), 0.0),
                        "Pred u (normalized)", cmap)
            _draw_loss_map(axes[2, 2], u_leaves, loss_u, max_lu, "Loss map u", loss_cmap)
            
            ax_d2 = axes[2, 3]
            depths_u = [d for d in range(MAX_LEVEL+1) if lbd_u[d]]
            avgs_u = [np.mean(lbd_u[d]) for d in depths_u]
            if depths_u:
                ax_d2.bar(depths_u, avgs_u, color='coral', edgecolor='black')
                for d, v in zip(depths_u, avgs_u):
                    ax_d2.text(d, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
            ax_d2.set_title("Avg loss/depth (u)", fontsize=9)
            ax_d2.set_xticks(range(MAX_LEVEL+1)); ax_d2.grid(True, axis='y', alpha=0.3)
            
            # --- Row 3: u Embedding PCA, Histogram, Loss Histogram ---
            emb_u = E_u[viz_depth]
            keys_u = qt_u.keys[viz_depth]
            if emb_u is not None and len(emb_u) > 0:
                plot_embedding_spatial(axes[3, 0], qt_u, keys_u, emb_u, viz_depth)
                data_u = emb_u.detach().cpu().numpy()
                axes[3, 1].hist(data_u.flatten(), bins=50, color='coral', alpha=0.7)
                axes[3, 1].set_title("u Latent Activations", fontsize=8)
                axes[3, 1].set_yscale('log'); axes[3, 1].grid(True, alpha=0.3)
            else:
                axes[3, 0].text(0.5, 0.5, "No embeddings", ha='center', va='center'); axes[3, 0].axis('off')
                axes[3, 1].axis('off')
            
            all_lu = list(loss_u.values())
            if all_lu:
                axes[3, 2].hist(all_lu, bins=40, color='coral', alpha=0.75, edgecolor='black')
                axes[3, 2].axvline(np.mean(all_lu), color='red', linestyle='--', label=f'mean={np.mean(all_lu):.5f}')
                axes[3, 2].legend(fontsize=8)
            axes[3, 2].set_title("u Loss Distribution", fontsize=9); axes[3, 2].grid(True, alpha=0.3)
            axes[3, 3].axis('off')
            
            plt.suptitle(f'Validation Sample {i+1}: k1={k1}, k2={k2}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{val_dir}/sample_{i+1:02d}_k{k1}_{k2}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    -> Saved {val_dir}/sample_{i+1:02d}_k{k1}_{k2}.png")
            
            # Accumulate for summary
            for d in range(MAX_LEVEL + 1):
                if lbd_f[d]:
                    all_samples_loss_f[d].append(np.mean(lbd_f[d]))
                if lbd_u[d]:
                    all_samples_loss_u[d].append(np.mean(lbd_u[d]))
    
    # Summary plot: Average loss by depth across all samples
    fig_summary, axes_s = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, loss_data, title, color in [
        (axes_s[0], all_samples_loss_f, 'f: Avg Loss by Depth', 'steelblue'),
        (axes_s[1], all_samples_loss_u, 'u: Avg Loss by Depth', 'coral'),
    ]:
        depths = [d for d in range(MAX_LEVEL + 1) if loss_data[d]]
        avgs = [np.mean(loss_data[d]) for d in depths]
        stds = [np.std(loss_data[d]) for d in depths]
        if depths:
            bars = ax.bar(depths, avgs, yerr=stds, color=color, edgecolor='black', capsize=5, alpha=0.8)
            for bar, val in zip(bars, avgs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.5f}',
                        ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Depth Level', fontsize=12)
        ax.set_ylabel('Average MSE Loss', fontsize=12)
        ax.set_title(f'{title} (Across {num_samples} Samples)', fontsize=14)
        ax.set_xticks(range(MAX_LEVEL + 1))
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{val_dir}/summary_avg_loss_by_depth.png', dpi=150, bbox_inches='tight')
    plt.close(fig_summary)
    print(f"\n  -> Saved summary: {val_dir}/summary_avg_loss_by_depth.png")
    
    return ckpt_f, ckpt_u


if __name__ == '__main__':
    main()