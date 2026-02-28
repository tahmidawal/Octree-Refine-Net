#!/usr/bin/env python3

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
# 1. Low-Level Operations: Morton Codes (2D)
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
        x = x.long()
        y = y.long()
        kx = Morton2D._interleave_bits(x)
        ky = Morton2D._interleave_bits(y)
        key = kx | (ky << 1)
        return key.long()

    @staticmethod
    def key2xy(key, depth=16):
        if not torch.is_tensor(key):
            key = torch.tensor(key, dtype=torch.long)
        key = key.long()
        x = Morton2D._deinterleave_bits(key)
        y = Morton2D._deinterleave_bits(key >> 1)
        return x.long(), y.long()


# ==========================================
# 1b. Positional Encoding Helpers
# ==========================================

def node_centers_from_keys(keys: torch.Tensor, depth: int, max_depth: int, device=None):
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


def fourier_encode(pos: torch.Tensor, num_freqs: int = 6):
    if pos.numel() == 0:
        return pos
    freqs = (2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)).view(1, 1, -1)
    x = pos.unsqueeze(-1) * np.pi * 2.0 * freqs
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    enc = enc.view(pos.shape[0], -1)
    return torch.cat([pos, enc], dim=1)


# ==========================================
# 2. True Adaptive Quadtree Data Structure
# ==========================================

class Quadtree:
    def __init__(self, max_depth, device='cpu'):
        self.max_depth = max_depth
        self.device = device

        self.keys = [None] * (max_depth + 1)
        self.neighs = [None] * (max_depth + 1)
        self.features_in = [None] * (max_depth + 1)

        self.children_idx = [None] * (max_depth + 1)
        self.parent_idx   = [None] * (max_depth + 1)

        self.split_gt   = [None] * (max_depth + 1)
        self.leaf_mask  = [None] * (max_depth + 1)
        self.values_gt  = [None] * (max_depth + 1)

    def build_from_leaves(self, leaf_keys_by_depth, leaf_vals_input_by_depth, leaf_vals_target_by_depth=None):
        if leaf_vals_target_by_depth is None:
            leaf_vals_target_by_depth = leaf_vals_input_by_depth

        C_in = None
        for d in range(self.max_depth + 1):
            if leaf_vals_input_by_depth[d].numel() > 0:
                C_in = leaf_vals_input_by_depth[d].shape[1]
                break
        if C_in is None:
            C_in = 1

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
            if self.keys[d].numel() == 0:
                continue
            parents = torch.unique(self.keys[d] >> 2, sorted=True)
            self.keys[d - 1] = torch.unique(torch.cat([self.keys[d - 1], parents]), sorted=True)

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

                idx = torch.searchsorted(kd, lk_unique)
                idx = idx.clamp(0, len(kd) - 1)
                found = kd[idx] == lk_unique
                feat[idx[found]] = pooled[found]

            self.features_in[d] = feat

        for d in range(self.max_depth):
            kd = self.keys[d]
            kn = self.keys[d + 1]

            if kd.numel() == 0:
                self.children_idx[d] = torch.empty((0, 4), dtype=torch.long, device=self.device)
                continue

            if kn.numel() == 0:
                self.children_idx[d] = torch.full((len(kd), 4), -1, dtype=torch.long, device=self.device)
                continue

            child_keys = (kd.unsqueeze(1) << 2) + torch.arange(4, device=self.device).view(1, 4)
            idx = torch.searchsorted(kn, child_keys)
            idx = idx.clamp(0, len(kn) - 1)
            found = (kn[idx] == child_keys)
            self.children_idx[d] = torch.where(found, idx, torch.full_like(idx, -1))

        self.children_idx[self.max_depth] = None

        self.parent_idx[0] = None
        for d in range(1, self.max_depth + 1):
            kd = self.keys[d]
            kp = self.keys[d - 1]
            if kd.numel() == 0:
                self.parent_idx[d] = torch.empty((0,), dtype=torch.long, device=self.device)
                continue
            parent_keys = kd >> 2
            pidx = torch.searchsorted(kp, parent_keys)
            pidx = pidx.clamp(0, len(kp) - 1)
            self.parent_idx[d] = pidx

        for d in range(self.max_depth + 1):
            self._construct_neigh_geometric_safe(d)

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

                idx = torch.searchsorted(kd, lk_unique)
                idx = idx.clamp(0, len(kd) - 1)
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

    def _construct_neigh_geometric_safe(self, depth):
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
            [[-1,-1],[-1,0],[-1,1],
             [ 0,-1],[ 0,0],[ 0,1],
             [ 1,-1],[ 1,0],[ 1,1]],
            device=self.device,
            dtype=torch.long
        )
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)

        res = 1 << depth
        nx = n_coords[..., 0]
        ny = n_coords[..., 1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)

        n_keys = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
        if valid.any():
            n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)

        idx = torch.searchsorted(keys, n_keys.clamp(min=0))
        idx = idx.clamp(0, len(keys) - 1)
        found = valid & (keys[idx] == n_keys)
        idx[~found] = -1
        self.neighs[depth] = idx


# ==========================================
# 3. Sparse QuadConv Layers
# ==========================================

class QuadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.weights = nn.Linear(9 * in_channels, out_channels)

    def forward(self, features, quadtree, depth):
        neigh_idx = quadtree.neighs[depth]
        N, K = neigh_idx.shape
        if N == 0:
            return torch.zeros((0, self.weights.out_features), device=features.device)

        pad_vec = torch.zeros((1, features.shape[1]), device=features.device)
        feat_padded = torch.cat([features, pad_vec], dim=0)

        gather_idx = neigh_idx.clone()
        gather_idx[gather_idx == -1] = N
        col = feat_padded[gather_idx]
        col_flat = col.view(N, -1)
        return self.weights(col_flat)

class QuadPool(nn.Module):
    def forward(self, child_features, quadtree, depth_child):
        assert depth_child >= 1
        d_parent = depth_child - 1
        parent_keys = quadtree.keys[d_parent]
        Np = len(parent_keys)
        C = child_features.shape[1]

        if Np == 0:
            return torch.zeros((0, C), device=child_features.device)

        ch = quadtree.children_idx[d_parent]
        pooled = torch.zeros((Np, C), device=child_features.device)
        cnt = torch.zeros((Np, 1), device=child_features.device)

        for c in range(4):
            idx = ch[:, c]
            mask = (idx != -1)
            if mask.any():
                pooled[mask] += child_features[idx[mask]]
                cnt[mask] += 1.0

        pooled = pooled / cnt.clamp(min=1.0)
        return pooled


# ==========================================
# 4. Autoencoder
# ==========================================

class TreeEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=64, emb_dim=None, max_depth=7, pos_freqs=6):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.max_depth = max_depth
        self.hidden = hidden
        self.emb_dim = hidden if emb_dim is None else emb_dim

        pos_dim = 3 + 2 * pos_freqs * 3
        self.in_proj = nn.Linear(in_c + pos_dim, hidden)
        self.convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool = QuadPool()

        self.to_emb = nn.ModuleList([nn.Linear(hidden, self.emb_dim) for _ in range(max_depth + 1)])
        self.emb_norm = nn.ModuleList([nn.LayerNorm(self.emb_dim) for _ in range(max_depth + 1)])
        self.depth_gain = nn.Parameter(torch.ones(max_depth + 1))
        
        self.global_proj = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, qt: Quadtree):
        h = [None] * (self.max_depth + 1)

        for d in range(self.max_depth + 1):
            fin = qt.features_in[d]
            kd = qt.keys[d]
            if fin is None or fin.numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=qt.device)
                continue

            pos = node_centers_from_keys(kd, d, self.max_depth, device=qt.device)
            pos = fourier_encode(pos, num_freqs=self.pos_freqs)

            x = torch.cat([fin, pos], dim=1)
            h[d] = self.in_proj(x)

        for d in range(self.max_depth, 0, -1):
            if h[d].numel() == 0:
                continue
            pooled = self.pool(h[d], qt, d)
            h[d - 1] = h[d - 1] + pooled

            if d - 1 >= 1 and h[d - 1].numel() > 0:
                h[d - 1] = F.relu(self.convs[d - 1](h[d - 1], qt, d - 1))

        E = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            if h[d] is None or h[d].numel() == 0:
                E[d] = torch.zeros((0, self.emb_dim), device=qt.device)
            else:
                z = self.to_emb[d](h[d])
                z = self.emb_norm[d](z)
                z = self.depth_gain[d] * z
                E[d] = z

        if E[0] is not None and E[0].numel() > 0:
            z_global = self.global_proj(E[0].mean(dim=0, keepdim=True))
        else:
            z_global = torch.zeros((1, self.emb_dim), device=qt.device)

        return E, z_global


class TreeDecoderTeacherForced(nn.Module):
    def __init__(self, hidden=64, emb_dim=None, out_c=1, max_depth=7, pos_freqs=6):
        super().__init__()
        self.max_depth = max_depth
        self.pos_freqs = pos_freqs
        self.hidden = hidden
        self.emb_dim = hidden if emb_dim is None else emb_dim

        pos_dim = 3 + 2 * pos_freqs * 3
        self.root_token = nn.Parameter(torch.zeros(1, hidden))

        self.fuse = nn.Sequential(
            nn.Linear(hidden + self.emb_dim + pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.global_norm = nn.LayerNorm(self.emb_dim)

        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4 * hidden))
        self.val_head   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))

        self.mix_convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

    def forward(self, qt: Quadtree, z_global: torch.Tensor):
        """
        Decode on qt (teacher tree) using global latent z_global from encoder.
        z_global: (1, emb_dim) - global latent code representing PDE content
        """
        h_by_depth = [None] * (self.max_depth + 1)

        N0 = len(qt.keys[0])
        if N0 == 0:
            h_by_depth[0] = torch.zeros((0, self.hidden), device=qt.device)
        else:
            h_by_depth[0] = self.root_token.expand(N0, -1).to(qt.device)

        split_logits = [None] * self.max_depth
        val_pred     = [None] * (self.max_depth + 1)
        
        z_global_normed = self.global_norm(z_global)

        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=qt.device)
                if d < self.max_depth:
                    split_logits[d] = torch.zeros((0,), device=qt.device)
                continue

            kd = qt.keys[d]
            pos = node_centers_from_keys(kd, d, self.max_depth, device=qt.device)
            pos = fourier_encode(pos, num_freqs=self.pos_freqs)

            z_broadcast = z_global_normed.expand(h.shape[0], -1)
            h = self.fuse(torch.cat([h, z_broadcast, pos], dim=1))

            if h.numel() > 0 and d >= 1:
                h = F.relu(self.mix_convs[d](h, qt, d))

            val_pred[d] = self.val_head(h)

            if d == self.max_depth:
                break

            split_logits[d] = self.split_head(h).squeeze(-1)

            ch = qt.children_idx[d]
            has_child = (ch != -1).any(dim=1)

            N_next = len(qt.keys[d + 1])
            h_next = torch.zeros((N_next, self.hidden), device=h.device)

            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, self.hidden)
                parent_rows = torch.nonzero(has_child).squeeze(-1)

                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1:
                            h_next[ci] = child_feats[t, c]

            h_by_depth[d + 1] = h_next

        return split_logits, val_pred


class TreeAE_DirectionB(nn.Module):
    def __init__(self, in_c=1, hidden=64, emb_dim=None, max_depth=7, pos_freqs=6):
        super().__init__()
        self.encoder = TreeEncoder(in_c=in_c, hidden=hidden, emb_dim=emb_dim, max_depth=max_depth, pos_freqs=pos_freqs)
        self.decoder = TreeDecoderTeacherForced(hidden=hidden, emb_dim=(hidden if emb_dim is None else emb_dim),
                                                out_c=in_c, max_depth=max_depth, pos_freqs=pos_freqs)

    def forward(self, qt_enc: Quadtree, qt_teach: Quadtree = None):
        """
        Forward pass with separate encoder and teacher trees.
        qt_enc: tree to encode (observed PDE)
        qt_teach: tree to decode on (teacher topology). If None, uses qt_enc.
        """
        if qt_teach is None:
            qt_teach = qt_enc
        
        E, z_global = self.encoder(qt_enc)
        split_logits, val_pred = self.decoder(qt_teach, z_global)
        return split_logits, val_pred, z_global

    @torch.no_grad()
    def encode(self, qt: Quadtree):
        self.encoder.eval()
        return self.encoder(qt)

    @torch.no_grad()
    def decode(self, qt: Quadtree, z_global: torch.Tensor):
        self.decoder.eval()
        return self.decoder(qt, z_global)


# ==========================================
# 5. Data Generation 
# ==========================================

MAX_LEVEL = 7
NOISE_STD = 0.05 

def target_function(x, y, k1, k2):
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)

def gradient_magnitude(x, y, k1, k2):
    dfdx = 2 * np.pi * k1 * np.cos(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)
    dfdy = 2 * np.pi * k2 * np.sin(2 * np.pi * k1 * x) * np.cos(2 * np.pi * k2 * y)
    return np.sqrt(dfdx**2 + dfdy**2)

class QuadNode:
    def __init__(self, x, y, size, level, max_level, min_level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.max_level = max_level
        self.min_level = min_level
        self.children, self.val = [], None

    def gradient_subdivide(self, k1, k2, grad_threshold=0.3):
        if self.level >= self.max_level:
            return

        center_x = self.x + self.size / 2
        center_y = self.y + self.size / 2

        force_split = self.level < self.min_level
        grad_mag = gradient_magnitude(center_x, center_y, k1, k2)

        max_grad = 2 * np.pi * np.sqrt(k1**2 + k2**2)
        normalized_grad = grad_mag / max_grad

        depth_factor = 1.0 + 0.1 * self.level
        effective_threshold = grad_threshold * depth_factor

        should_split = force_split or (normalized_grad > effective_threshold)

        if should_split:
            half = self.size / 2
            self.children = [
                QuadNode(self.x,        self.y,        half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y,        half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x,        self.y + half, half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y + half, half, self.level + 1, self.max_level, self.min_level),
            ]
            for child in self.children:
                child.gradient_subdivide(k1, k2, grad_threshold)

    def collect_leaves(self, leaves_list, k1, k2):
        if not self.children:
            center_x = self.x + self.size / 2
            center_y = self.y + self.size / 2
            self.val = target_function(center_x, center_y, k1, k2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves(leaves_list, k1, k2)

def generate_user_data(noise_std=NOISE_STD, grad_threshold=0.3):
    k1 = random.randint(1, 5)
    k2 = random.randint(1, 5)

    # Alternate between training on (5,6) trees and (6,7) trees
    if random.random() < 0.5:
        min_lvl, max_lvl = 5, 6
    else:
        min_lvl, max_lvl = 6, 7

    root = QuadNode(0, 0, 1.0, 0, max_lvl, min_lvl)
    root.gradient_subdivide(k1, k2, grad_threshold=grad_threshold)

    leaves = []
    root.collect_leaves(leaves, k1, k2)

    # Initialize up to global MAX_LEVEL
    leaf_keys = [[] for _ in range(MAX_LEVEL + 1)]
    leaf_vals = [[] for _ in range(MAX_LEVEL + 1)]

    for node in leaves:
        d = node.level
        res = 1 << d
        ix = int(node.x * res)
        iy = int(node.y * res)
        ix = max(0, min(res - 1, ix))
        iy = max(0, min(res - 1, iy))

        k = Morton2D.xy2key(ix, iy)
        leaf_keys[d].append(int(k.item()))
        leaf_vals[d].append([float(node.val)])

    leaf_keys_by_depth = []
    leaf_vals_clean_by_depth = []
    leaf_vals_noisy_by_depth = []
    
    # Pack up to global MAX_LEVEL to keep architecture tensor sizes uniform
    for d in range(MAX_LEVEL + 1):
        if len(leaf_keys[d]) == 0:
            leaf_keys_by_depth.append(torch.empty((0,), dtype=torch.long))
            leaf_vals_clean_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
            leaf_vals_noisy_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
        else:
            leaf_keys_by_depth.append(torch.tensor(leaf_keys[d], dtype=torch.long))
            clean = torch.tensor(leaf_vals[d], dtype=torch.float32)
            noisy = clean + noise_std * torch.randn_like(clean)
            leaf_vals_clean_by_depth.append(clean)
            leaf_vals_noisy_by_depth.append(noisy)

    return leaf_keys_by_depth, leaf_vals_noisy_by_depth, leaf_vals_clean_by_depth, leaves, k1, k2


def eval_pde_on_leaf_keys(leaf_keys_by_depth, k1, k2, max_depth):
    """
    Evaluate the PDE (sin function) on arbitrary leaf keys.
    leaf_keys_by_depth[d]: LongTensor of leaf keys (cells) at depth d.
    Returns leaf_vals_clean_by_depth in same structure: FloatTensor (N_leaf_d, 1)
    """
    leaf_vals_clean_by_depth = []
    for d in range(max_depth + 1):
        keys = leaf_keys_by_depth[d]
        if keys.numel() == 0:
            leaf_vals_clean_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
            continue

        ix, iy = Morton2D.key2xy(keys, depth=d)
        res = float(1 << d)
        x = (ix.float() + 0.5) / res
        y = (iy.float() + 0.5) / res

        val = torch.sin(2 * np.pi * k1 * x) * torch.sin(2 * np.pi * k2 * y)
        leaf_vals_clean_by_depth.append(val.view(-1, 1).float().cpu())

    return leaf_vals_clean_by_depth


def make_uniform_leaf_keys(depth, max_depth):
    """
    Returns leaf_keys_by_depth where ALL leaves are at exactly 'depth'.
    Creates a uniform grid at the specified depth.
    """
    leaf_keys_by_depth = [torch.empty((0,), dtype=torch.long) for _ in range(max_depth + 1)]
    res = 1 << depth
    xs = torch.arange(res, dtype=torch.long)
    ys = torch.arange(res, dtype=torch.long)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    keys = Morton2D.xy2key(X.reshape(-1), Y.reshape(-1), depth=depth).cpu().long()
    leaf_keys_by_depth[depth] = keys
    return leaf_keys_by_depth


# ==========================================
# 6. Main Execution
# ==========================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # Set architecture max depth to 7
    model = TreeAE_DirectionB(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    mse = nn.MSELoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"plots/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    num_steps = 3000
    split_weight = 0.5
    print(f"Training Denoising AE across random trees for {num_steps} steps...")
    print(f"  - Random K1,K2 in [1,5] each step")
    print(f"  - Encoder: Adaptive topology (5,6) or (6,7)")
    print(f"  - Teacher: Uniform grid at depth 5, 6, or 7 (different from encoder)")

    val_losses = []
    split_losses = []
    total_losses = []
    sample_trees = []

    best_total_loss = float('inf')
    best_step = 0
    best_model_path = f'{output_dir}/best_model.pt'

    patience = 150
    patience_counter = 0
    bce = nn.BCEWithLogitsLoss()

    for step in range(num_steps):
        # 1) Build encoder tree (adaptive topology)
        leaf_keys_enc, leaf_vals_noisy_enc, leaf_vals_clean_enc, leaves_enc, k1, k2 = generate_user_data()

        qt_enc = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_enc.build_from_leaves(leaf_keys_enc, leaf_vals_noisy_enc, leaf_vals_clean_enc)

        # 2) Build teacher tree (uniform grid at random depth, different topology)
        Dteach = random.choice([5, 6, 7])
        leaf_keys_teach = make_uniform_leaf_keys(Dteach, MAX_LEVEL)

        # Evaluate the SAME PDE on teacher leaves
        leaf_vals_clean_teach = eval_pde_on_leaf_keys(leaf_keys_teach, k1, k2, MAX_LEVEL)

        # Add noise for denoising setup
        leaf_vals_noisy_teach = []
        for d in range(MAX_LEVEL + 1):
            clean = leaf_vals_clean_teach[d]
            if clean.numel() == 0:
                leaf_vals_noisy_teach.append(clean)
            else:
                leaf_vals_noisy_teach.append(clean + NOISE_STD * torch.randn_like(clean))

        qt_teach = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_teach.build_from_leaves(leaf_keys_teach, leaf_vals_noisy_teach, leaf_vals_clean_teach)

        # 3) Forward: ENCODE on qt_enc, DECODE on qt_teach
        optimizer.zero_grad()
        split_logits, val_pred, z_global = model(qt_enc, qt_teach)

        # 4) Losses computed on TEACHER tree targets
        L_val = torch.tensor(0.0, device=device)
        n_val_terms = 0
        for d in range(MAX_LEVEL + 1):
            mask = qt_teach.leaf_mask[d]
            if mask is None or mask.numel() == 0:
                continue
            if mask.any():
                L_val = L_val + mse(val_pred[d][mask], qt_teach.values_gt[d][mask])
                n_val_terms += 1
        if n_val_terms > 0:
            L_val = L_val / n_val_terms

        L_split = torch.tensor(0.0, device=device)
        n_split_terms = 0
        for d in range(MAX_LEVEL):
            if qt_teach.split_gt[d] is None or qt_teach.split_gt[d].numel() == 0:
                continue
            if split_logits[d] is None or split_logits[d].numel() == 0:
                continue
            L_split = L_split + bce(split_logits[d], qt_teach.split_gt[d])
            n_split_terms += 1
        if n_split_terms > 0:
            L_split = L_split / n_split_terms

        gain_reg = 1e-4 * (model.encoder.depth_gain ** 2).mean()
        L_total = L_val + split_weight * L_split + gain_reg

        L_total.backward()
        optimizer.step()

        val_losses.append(float(L_val.item()))
        split_losses.append(float(L_split.item()))
        total_losses.append(float(L_total.item()))

        n_teach_leaves = sum(lk.numel() for lk in leaf_keys_teach)

        if L_total.item() < best_total_loss:
            best_total_loss = L_total.item()
            best_step = step
            patience_counter = 0
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': best_total_loss,
                'val_loss': L_val.item(),
                'split_loss': L_split.item(),
            }, best_model_path)
        else:
            patience_counter += 1

        print(f"Step {step}: val={L_val.item():.6f} split={L_split.item():.6f} total={L_total.item():.6f}  "
              f"k1={k1} k2={k2} enc_leaves={len(leaves_enc)} teach_D={Dteach} teach_leaves={n_teach_leaves}  "
              f"best={best_total_loss:.6f}@{best_step}  patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at step {step} (no improvement for {patience} steps)")
            break

        if step % 100 == 0:
            sample_trees.append({
                'qt_enc': qt_enc, 'qt_teach': qt_teach,
                'leaves_enc': leaves_enc, 'Dteach': Dteach,
                'k1': k1, 'k2': k2, 'step': step
            })
            if len(sample_trees) > 3:
                sample_trees.pop(0)

        if step % 100 == 0 or step == num_steps - 1:
            with torch.no_grad():
                for sample in sample_trees:
                    # Decode on teacher tree using encoder's latent
                    _, val_pred_viz, _ = model(sample['qt_enc'], sample['qt_teach'])
                    sample['pred_dict'] = {}
                    sample['loss_dict'] = {}
                    sample['loss_by_depth'] = {d: [] for d in range(MAX_LEVEL + 1)}
                    sample['gt_dict'] = {}

                    # Collect predictions and losses on TEACHER tree
                    qt_teach = sample['qt_teach']
                    for d in range(MAX_LEVEL + 1):
                        if qt_teach.keys[d].numel() == 0:
                            continue
                        mask = qt_teach.leaf_mask[d]
                        if mask is None or mask.numel() == 0 or not mask.any():
                            continue
                        kd = qt_teach.keys[d][mask].cpu()
                        ix, iy = Morton2D.key2xy(kd, depth=d)
                        pr = val_pred_viz[d][mask].cpu().numpy().flatten()
                        gt = qt_teach.values_gt[d][mask].cpu().numpy().flatten()

                        for i in range(len(kd)):
                            key = (d, int(ix[i].item()), int(iy[i].item()))
                            sample['pred_dict'][key] = pr[i]
                            sample['gt_dict'][key] = gt[i]
                            leaf_loss = (pr[i] - gt[i]) ** 2
                            sample['loss_dict'][key] = leaf_loss
                            sample['loss_by_depth'][d].append(leaf_loss)

            n_samples = len(sample_trees)
            if n_samples > 0:
                fig, axes = plt.subplots(n_samples, 4, figsize=(24, 6*n_samples))
                if n_samples == 1:
                    axes = axes.reshape(1, -1)

                cmap = plt.get_cmap('viridis')
                loss_cmap = plt.get_cmap('hot')

                for row, sample in enumerate(sample_trees):
                    all_losses = list(sample['loss_dict'].values())
                    max_loss = max(all_losses) if all_losses else 1.0
                    max_loss = max(max_loss, 1e-6)
                    Dteach = sample['Dteach']
                    res_teach = 1 << Dteach
                    cell_size = 1.0 / res_teach

                    # Column 1: Encoder GT (adaptive tree)
                    ax1 = axes[row, 0]
                    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
                    ax1.set_aspect('equal'); ax1.axis('off')
                    ax1.set_title(f"Encoder GT Step {sample['step']}: k1={sample['k1']}, k2={sample['k2']}", fontsize=10)

                    for node in sample['leaves_enc']:
                        normalized_color = (node.val + 1) / 2
                        color = cmap(normalized_color)
                        rect = patches.Rectangle((node.x, node.y), node.size, node.size,
                                                 linewidth=0.5, edgecolor='black', facecolor=color)
                        ax1.add_patch(rect)

                    # Column 2: Teacher GT (uniform grid)
                    ax2 = axes[row, 1]
                    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
                    ax2.set_aspect('equal'); ax2.axis('off')
                    ax2.set_title(f"Teacher GT (D={Dteach})", fontsize=10)

                    for (d, ix_t, iy_t), gt_val in sample['gt_dict'].items():
                        x = ix_t * cell_size
                        y = iy_t * cell_size
                        normalized_color = np.clip((gt_val + 1) / 2, 0, 1)
                        color = cmap(normalized_color)
                        rect = patches.Rectangle((x, y), cell_size, cell_size,
                                                 linewidth=0.3, edgecolor='gray', facecolor=color)
                        ax2.add_patch(rect)

                    # Column 3: Teacher Predicted (uniform grid)
                    ax3 = axes[row, 2]
                    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
                    ax3.set_aspect('equal'); ax3.axis('off')
                    ax3.set_title(f"Teacher Predicted (D={Dteach})", fontsize=10)

                    for (d, ix_t, iy_t), pred_val in sample['pred_dict'].items():
                        x = ix_t * cell_size
                        y = iy_t * cell_size
                        normalized_color = np.clip((pred_val + 1) / 2, 0, 1)
                        color = cmap(normalized_color)
                        rect = patches.Rectangle((x, y), cell_size, cell_size,
                                                 linewidth=0.3, edgecolor='gray', facecolor=color)
                        ax3.add_patch(rect)

                    # Column 4: Loss by depth
                    ax4 = axes[row, 3]
                    depths, avg_losses = [], []
                    for d in range(MAX_LEVEL + 1):
                        if sample['loss_by_depth'][d]:
                            depths.append(d)
                            avg_losses.append(np.mean(sample['loss_by_depth'][d]))

                    if depths:
                        bars = ax4.bar(depths, avg_losses, color='steelblue', edgecolor='black')
                        ax4.set_xlabel('Depth Level', fontsize=9)
                        ax4.set_ylabel('Avg MSE Loss', fontsize=9)
                        ax4.set_title(f"Avg Loss by Depth (Teacher)", fontsize=10)
                        ax4.set_xticks(range(MAX_LEVEL + 1))
                        for bar, val in zip(bars, avg_losses):
                            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                     f'{val:.4f}', ha='center', va='bottom', fontsize=7)

                plt.tight_layout()
                plt.savefig(f'{output_dir}/samples_step{step:04d}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

            fig_loss, axes_loss = plt.subplots(1, 3, figsize=(18, 5))
            ax1 = axes_loss[0]
            ax1.plot(val_losses, linewidth=1, alpha=0.7)
            ax1.set_xlabel('Step'); ax1.set_ylabel('Value Loss (MSE)'); ax1.set_title('Value Loss'); ax1.grid(True, alpha=0.3)

            ax2 = axes_loss[1]
            ax2.plot(split_losses, linewidth=1, alpha=0.7)
            ax2.set_xlabel('Step'); ax2.set_ylabel('Split Loss (BCE)'); ax2.set_title('Split Loss'); ax2.grid(True, alpha=0.3)

            ax3 = axes_loss[2]
            ax3.plot(total_losses, linewidth=1, alpha=0.7)
            ax3.set_xlabel('Step'); ax3.set_ylabel('Total Loss'); ax3.set_title(f'Total Loss'); ax3.grid(True, alpha=0.3)

            plt.suptitle(f'Training Losses (Step {step})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/loss_curve_step{step:04d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig_loss)

    print(f"Training complete. Best model saved at step {best_step}")

if __name__ == "__main__":
    main()