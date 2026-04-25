"""
compressed_bottleneck_ae.py

Compressed Quadtree Autoencoder with configurable bottleneck depth.
Based on the proven feb17 DirectionB architecture, modified to:
  - Only output encoder embeddings E[0..D_BOTTLE] as the latent
  - Decoder receives skip connections ONLY at depths 0..D_BOTTLE
  - Decoder uses QuadConv spatial mixing at ALL depths (including below bottleneck)
  - Two-phase training: Phase 1 = teacher forcing, Phase 2 = autoregressive fine-tuning

Key constraint: At inference, the decoder works standalone with ONLY the
coarse embeddings + coarse topology. No GT fine structure, no encoder.

PDE: f(x,y) = sin(2*pi*k1*x) * sin(2*pi*k2*y)  on [0,1]^2

Usage:
  python compressed_bottleneck_ae.py --d_bottle 3
  python compressed_bottleneck_ae.py --d_bottle 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import argparse
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
    def xy2key(x, y, depth=None):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        return (Morton2D._interleave_bits(x) | (Morton2D._interleave_bits(y) << 1)).long()

    @staticmethod
    def key2xy(key, depth=None):
        if not torch.is_tensor(key):
            key = torch.tensor(key, dtype=torch.long)
        x = Morton2D._deinterleave_bits(key)
        y = Morton2D._deinterleave_bits(key >> 1)
        return x.long(), y.long()


# ==========================================
# 1b. Positional Encoding
# ==========================================

POS_FREQS = 6
POS_DIM = 3 + 2 * POS_FREQS * 3  # 39

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


def fourier_encode(pos, num_freqs=POS_FREQS):
    if pos.numel() == 0:
        return pos
    freqs = (2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)).view(1, 1, -1)
    x = pos.unsqueeze(-1) * np.pi * 2.0 * freqs
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(pos.shape[0], -1)
    return torch.cat([pos, enc], dim=1)


# ==========================================
# 2. Quadtree Data Structure (from feb17)
# ==========================================

class Quadtree:
    def __init__(self, max_depth, device='cpu'):
        self.max_depth = max_depth
        self.device = device
        self.keys = [None] * (max_depth + 1)
        self.neighs = [None] * (max_depth + 1)
        self.features_in = [None] * (max_depth + 1)
        self.children_idx = [None] * (max_depth + 1)
        self.parent_idx = [None] * (max_depth + 1)
        self.split_gt = [None] * (max_depth + 1)
        self.leaf_mask = [None] * (max_depth + 1)
        self.values_gt = [None] * (max_depth + 1)

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

        # Init keys and features from leaves
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
                cnt = torch.zeros((len(lk_unique), 1), device=self.device)
                feat.index_add_(0, inv, lv)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=self.device))
                self.features_in[d] = feat / cnt.clamp(min=1)

        if self.keys[0].numel() == 0:
            self.keys[0] = torch.tensor([0], dtype=torch.long, device=self.device)
            self.features_in[0] = torch.zeros((1, C_in), dtype=torch.float, device=self.device)

        # Ancestor closure
        for d in range(self.max_depth, 0, -1):
            if self.keys[d].numel() == 0:
                continue
            parents = torch.unique(self.keys[d] >> 2, sorted=True)
            self.keys[d - 1] = torch.unique(torch.cat([self.keys[d - 1], parents]), sorted=True)

        # Rebuild features for expanded keys
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
                idx = torch.searchsorted(kd, lk_unique).clamp(0, len(kd) - 1)
                found = kd[idx] == lk_unique
                feat[idx[found]] = pooled[found]
            self.features_in[d] = feat

        # Children indices
        for d in range(self.max_depth):
            kd, kn = self.keys[d], self.keys[d + 1]
            if kd.numel() == 0:
                self.children_idx[d] = torch.empty((0, 4), dtype=torch.long, device=self.device)
                continue
            if kn.numel() == 0:
                self.children_idx[d] = torch.full((len(kd), 4), -1, dtype=torch.long, device=self.device)
                continue
            child_keys = (kd.unsqueeze(1) << 2) + torch.arange(4, device=self.device).view(1, 4)
            idx = torch.searchsorted(kn, child_keys).clamp(0, len(kn) - 1)
            self.children_idx[d] = torch.where(kn[idx] == child_keys, idx, torch.full_like(idx, -1))
        self.children_idx[self.max_depth] = None

        # Parent indices
        self.parent_idx[0] = None
        for d in range(1, self.max_depth + 1):
            kd, kp = self.keys[d], self.keys[d - 1]
            if kd.numel() == 0:
                self.parent_idx[d] = torch.empty((0,), dtype=torch.long, device=self.device)
                continue
            pidx = torch.searchsorted(kp, kd >> 2).clamp(0, len(kp) - 1)
            self.parent_idx[d] = pidx

        # Neighbours
        for d in range(self.max_depth + 1):
            self._build_neighs(d)

        # GT labels
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
                idx = torch.searchsorted(kd, lk_unique).clamp(0, len(kd) - 1)
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

    def _build_neighs(self, depth):
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
            device=self.device, dtype=torch.long)
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        res = 1 << depth
        nx, ny = n_coords[..., 0], n_coords[..., 1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
        n_keys = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
        if valid.any():
            n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)
        idx = torch.searchsorted(keys, n_keys.clamp(min=0)).clamp(0, N - 1)
        found = valid & (keys[idx] == n_keys)
        idx[~found] = -1
        self.neighs[depth] = idx


# ==========================================
# 3. QuadConv and QuadPool (from feb17)
# ==========================================

class QuadConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Linear(9 * in_channels, out_channels)

    def forward(self, features, qt_or_neighs, depth=None):
        # Accept either (qt, depth) or raw neighs tensor
        if isinstance(qt_or_neighs, Quadtree):
            neigh_idx = qt_or_neighs.neighs[depth]
        else:
            neigh_idx = qt_or_neighs
        N, K = neigh_idx.shape
        if N == 0:
            return torch.zeros((0, self.weights.out_features), device=features.device)
        pad_vec = torch.zeros((1, features.shape[1]), device=features.device)
        feat_padded = torch.cat([features, pad_vec], dim=0)
        gather_idx = neigh_idx.clone()
        gather_idx[gather_idx == -1] = N
        col = feat_padded[gather_idx]
        return self.weights(col.view(N, -1))


class QuadPool(nn.Module):
    def forward(self, child_features, qt, depth_child):
        d_parent = depth_child - 1
        Np = len(qt.keys[d_parent])
        C = child_features.shape[1]
        if Np == 0:
            return torch.zeros((0, C), device=child_features.device)
        ch = qt.children_idx[d_parent]
        pooled = torch.zeros((Np, C), device=child_features.device)
        cnt = torch.zeros((Np, 1), device=child_features.device)
        for c in range(4):
            idx = ch[:, c]
            mask = idx != -1
            if mask.any():
                pooled[mask] += child_features[idx[mask]]
                cnt[mask] += 1.0
        return pooled / cnt.clamp(min=1.0)


# ==========================================
# 4. Compressed Encoder
# ==========================================

class CompressedEncoder(nn.Module):
    """
    Bottom-up encoder (same as feb17), but only outputs E[0..d_bottle].
    """
    def __init__(self, in_c=1, hidden=128, emb_dim=128, max_depth=7, d_bottle=3):
        super().__init__()
        self.max_depth = max_depth
        self.d_bottle = d_bottle
        self.hidden = hidden
        self.emb_dim = emb_dim

        self.in_proj = nn.Linear(in_c + POS_DIM, hidden)
        self.convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool = QuadPool()

        # Embedding heads only for depths 0..d_bottle
        self.to_emb = nn.ModuleList([
            nn.Linear(hidden, emb_dim) if d <= d_bottle else nn.Identity()
            for d in range(max_depth + 1)
        ])
        self.emb_norm = nn.ModuleList([
            nn.LayerNorm(emb_dim) if d <= d_bottle else nn.Identity()
            for d in range(max_depth + 1)
        ])
        self.depth_gain = nn.Parameter(torch.ones(d_bottle + 1))

    def forward(self, qt):
        h = [None] * (self.max_depth + 1)

        # Project inputs at every depth
        for d in range(self.max_depth + 1):
            fin = qt.features_in[d]
            if fin is None or fin.numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=qt.device)
                continue
            pos = fourier_encode(node_centers_from_keys(qt.keys[d], d, self.max_depth, device=qt.device))
            h[d] = self.in_proj(torch.cat([fin, pos], dim=1))

        # Bottom-up pooling (processes ALL depths for rich features)
        for d in range(self.max_depth, 0, -1):
            if h[d].numel() == 0:
                continue
            pooled = self.pool(h[d], qt, d)
            h[d - 1] = h[d - 1] + pooled
            if d - 1 >= 1 and h[d - 1].numel() > 0:
                h[d - 1] = F.relu(self.convs[d - 1](h[d - 1], qt, d - 1))

        # Extract bottleneck embeddings ONLY at depths 0..d_bottle
        E = [None] * (self.max_depth + 1)
        for d in range(self.d_bottle + 1):
            if h[d] is not None and h[d].numel() > 0:
                z = self.to_emb[d](h[d])
                z = self.emb_norm[d](z)
                z = self.depth_gain[d] * z
                E[d] = z
            else:
                E[d] = torch.zeros((0, self.emb_dim), device=qt.device)

        return E


# ==========================================
# 5. Compressed Decoder (with QuadConv at all depths)
# ==========================================

class CompressedDecoder(nn.Module):
    """
    Top-down decoder with:
    - Skip connections from encoder at depths 0..d_bottle
    - Below d_bottle: global bottleneck conditioning (mean-pooled coarse embeddings)
    - QuadConv spatial mixing at EVERY depth (critical for quality below bottleneck)
    - Vectorized child scatter (no Python for-loops)
    """
    def __init__(self, hidden=128, emb_dim=128, out_c=1, max_depth=7, d_bottle=3):
        super().__init__()
        self.max_depth = max_depth
        self.d_bottle = d_bottle
        self.hidden = hidden
        self.emb_dim = emb_dim

        self.root_token = nn.Parameter(torch.zeros(1, hidden))
        self.skip_norm = nn.LayerNorm(emb_dim)

        self.fuse = nn.Sequential(
            nn.Linear(hidden + emb_dim + POS_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # QuadConv at every depth including below bottleneck
        self.mix_convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4 * hidden))
        self.val_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))

    def _ancestor_skip(self, qt, emb_list, d, device):
        """
        For each node at depth d, find its ancestor at d_bottle and return
        that ancestor's embedding. For below-bottleneck nodes, also adds
        the d_bottle-1 ancestor's embedding (multi-scale conditioning).
        """
        if d <= self.d_bottle:
            if emb_list[d] is not None and emb_list[d].shape[0] == len(qt.keys[d]):
                return self.skip_norm(emb_list[d])
            return torch.zeros((len(qt.keys[d]), self.emb_dim), device=device)

        # Walk up from depth d to d_bottle using key bit-shifts
        keys_d = qt.keys[d]
        Nd = len(keys_d)
        if Nd == 0:
            return torch.zeros((0, self.emb_dim), device=device)

        # Primary: ancestor at d_bottle
        shift = 2 * (d - self.d_bottle)
        ancestor_keys = keys_d >> shift

        bottle_keys = qt.keys[self.d_bottle]
        if bottle_keys.numel() == 0 or emb_list[self.d_bottle] is None:
            return torch.zeros((Nd, self.emb_dim), device=device)

        idx = torch.searchsorted(bottle_keys, ancestor_keys).clamp(0, len(bottle_keys) - 1)
        found = bottle_keys[idx] == ancestor_keys

        bottle_emb = emb_list[self.d_bottle]
        skip = torch.zeros((Nd, self.emb_dim), device=device)
        skip[found] = self.skip_norm(bottle_emb[idx[found]])

        # Multi-scale: also add ancestor at d_bottle-1 (coarser scale)
        if self.d_bottle >= 1 and emb_list[self.d_bottle - 1] is not None:
            coarser_d = self.d_bottle - 1
            shift2 = 2 * (d - coarser_d)
            ancestor_keys2 = keys_d >> shift2
            coarser_keys = qt.keys[coarser_d]
            if coarser_keys.numel() > 0:
                idx2 = torch.searchsorted(coarser_keys, ancestor_keys2).clamp(0, len(coarser_keys) - 1)
                found2 = coarser_keys[idx2] == ancestor_keys2
                coarser_emb = emb_list[coarser_d]
                skip[found2] = skip[found2] + self.skip_norm(coarser_emb[idx2[found2]])

        return skip

    def _ancestor_skip_ar(self, cur_keys, emb_list, coarse_keys, d, device):
        """Ancestor skip for autoregressive decoding (uses coarse_keys, not qt).
        Multi-scale: adds d_bottle-1 ancestor embedding too."""
        if d <= self.d_bottle:
            if emb_list[d] is not None and emb_list[d].shape[0] == len(cur_keys):
                return self.skip_norm(emb_list[d])
            return torch.zeros((len(cur_keys), self.emb_dim), device=device)

        Nd = len(cur_keys)
        if Nd == 0:
            return torch.zeros((0, self.emb_dim), device=device)

        shift = 2 * (d - self.d_bottle)
        ancestor_keys = cur_keys >> shift

        bottle_keys = coarse_keys[self.d_bottle]
        if bottle_keys is None or bottle_keys.numel() == 0 or emb_list[self.d_bottle] is None:
            return torch.zeros((Nd, self.emb_dim), device=device)

        idx = torch.searchsorted(bottle_keys, ancestor_keys).clamp(0, len(bottle_keys) - 1)
        found = bottle_keys[idx] == ancestor_keys

        bottle_emb = emb_list[self.d_bottle]
        skip = torch.zeros((Nd, self.emb_dim), device=device)
        skip[found] = self.skip_norm(bottle_emb[idx[found]])

        # Multi-scale: also add ancestor at d_bottle-1
        if self.d_bottle >= 1 and emb_list[self.d_bottle - 1] is not None:
            coarser_d = self.d_bottle - 1
            shift2 = 2 * (d - coarser_d)
            ancestor_keys2 = cur_keys >> shift2
            coarser_keys = coarse_keys[coarser_d]
            if coarser_keys is not None and coarser_keys.numel() > 0:
                idx2 = torch.searchsorted(coarser_keys, ancestor_keys2).clamp(0, len(coarser_keys) - 1)
                found2 = coarser_keys[idx2] == ancestor_keys2
                coarser_emb = emb_list[coarser_d]
                skip[found2] = skip[found2] + self.skip_norm(coarser_emb[idx2[found2]])

        return skip

    def forward(self, qt, emb_list):
        """Teacher-forced decoding using GT topology."""
        h_by_depth = [None] * (self.max_depth + 1)
        N0 = len(qt.keys[0])
        h_by_depth[0] = self.root_token.expand(N0, -1).to(qt.device) if N0 > 0 else torch.zeros((0, self.hidden), device=qt.device)

        split_logits = [None] * self.max_depth
        val_pred = [None] * (self.max_depth + 1)

        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=qt.device)
                if d < self.max_depth:
                    split_logits[d] = torch.zeros((0,), device=qt.device)
                continue

            pos = fourier_encode(node_centers_from_keys(qt.keys[d], d, self.max_depth, device=qt.device))

            # Ancestor-indexed skip: each node gets its D_BOTTLE ancestor's embedding
            skip = self._ancestor_skip(qt, emb_list, d, qt.device)

            h = self.fuse(torch.cat([h, skip, pos], dim=1))

            # QuadConv spatial mixing at ALL depths (key for below-bottleneck quality)
            if h.numel() > 0 and d >= 1:
                h = F.relu(self.mix_convs[d](h, qt, d))

            val_pred[d] = self.val_head(h)

            if d == self.max_depth:
                break

            split_logits[d] = self.split_head(h).squeeze(-1)

            # Vectorized child scatter
            ch = qt.children_idx[d]
            has_child = (ch != -1).any(dim=1)
            N_next = len(qt.keys[d + 1])
            h_next = torch.zeros((N_next, self.hidden), device=h.device)

            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, self.hidden)
                ch_sub = ch[has_child]
                valid = ch_sub != -1
                h_next[ch_sub[valid]] = child_feats[valid]

            h_by_depth[d + 1] = h_next

        return split_logits, val_pred

    def decode_autoregressive(self, emb_list, coarse_keys, max_depth, min_level=6, split_thresh=0.0):
        """
        Standalone autoregressive decoding from ONLY coarse embeddings.
        No GT tree needed. Builds topology by thresholding split predictions.

        Args:
            emb_list: list of tensors E[0..d_bottle], None for depths above d_bottle
            coarse_keys: list of Morton key tensors for depths 0..d_bottle
            max_depth: how deep to generate
            min_level: force splits below this depth
            split_thresh: threshold for split decisions
        Returns:
            pred_keys: list of key tensors per depth
            val_preds: list of value tensors per depth
        """
        dev = emb_list[0].device if emb_list[0] is not None else 'cpu'
        d_bottle = self.d_bottle

        # Initialize all-depth key/value storage
        pred_keys = [None] * (max_depth + 1)
        val_preds = [None] * (max_depth + 1)

        # Copy coarse keys
        for d in range(d_bottle + 1):
            pred_keys[d] = coarse_keys[d].clone() if coarse_keys[d] is not None else torch.empty((0,), dtype=torch.long, device=dev)

        # Start from root token
        N0 = len(pred_keys[0])
        h_by_depth = [None] * (max_depth + 1)
        h_by_depth[0] = self.root_token.expand(N0, -1).to(dev) if N0 > 0 else torch.zeros((0, self.hidden), device=dev)

        for d in range(max_depth + 1):
            h = h_by_depth[d]
            cur_keys = pred_keys[d]

            if h is None or h.numel() == 0 or cur_keys is None or cur_keys.numel() == 0:
                val_preds[d] = torch.zeros((0, 1), device=dev)
                continue

            Nd = len(cur_keys)
            pos = fourier_encode(node_centers_from_keys(cur_keys, d, max_depth, device=dev))

            # Ancestor-indexed skip: each node gets its D_BOTTLE ancestor's embedding
            skip = self._ancestor_skip_ar(cur_keys, emb_list, coarse_keys, d, dev)

            h = self.fuse(torch.cat([h, skip, pos], dim=1))

            # Build live neighbours for predicted keys and apply QuadConv
            if d >= 1 and Nd > 0:
                live_neighs = self._build_neighs_for_keys(cur_keys, d, dev)
                h = F.relu(self.mix_convs[d](h, live_neighs))

            val_preds[d] = self.val_head(h)

            if d == max_depth:
                break

            # Predict splits
            split_logits = self.split_head(h).squeeze(-1)

            if d < min_level:
                split_mask = torch.ones(Nd, dtype=torch.bool, device=dev)
            else:
                split_mask = split_logits > split_thresh

            if not split_mask.any():
                for dd in range(d + 1, max_depth + 1):
                    pred_keys[dd] = torch.empty((0,), dtype=torch.long, device=dev)
                    val_preds[dd] = torch.empty((0, 1), device=dev)
                break

            # Generate child keys
            keys_to_split = cur_keys[split_mask]
            child_keys_all = (keys_to_split.unsqueeze(1) << 2) + torch.arange(4, device=dev).view(1, 4)
            next_keys = torch.unique(child_keys_all.view(-1), sorted=True)
            pred_keys[d + 1] = next_keys

            # Generate child hidden states (vectorized)
            N_next = len(next_keys)
            h_next = torch.zeros((N_next, self.hidden), device=dev)
            child_feats = self.child_head(h[split_mask]).view(-1, 4, self.hidden)

            # Map each child to its position in next_keys
            for c in range(4):
                ck = child_keys_all[:, c]
                ci = torch.searchsorted(next_keys, ck).clamp(0, N_next - 1)
                valid = next_keys[ci] == ck
                h_next[ci[valid]] = child_feats[:, c][valid]

            h_by_depth[d + 1] = h_next

        return pred_keys, val_preds

    @staticmethod
    def _build_neighs_for_keys(keys, depth, device):
        """Build neighbour indices for arbitrary key set (used in AR inference)."""
        N = len(keys)
        if N == 0:
            return torch.empty((0, 9), dtype=torch.long, device=device)
        if depth == 0:
            neigh = torch.full((N, 9), -1, dtype=torch.long, device=device)
            neigh[:, 4] = 0
            return neigh
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor(
            [[-1,-1],[-1,0],[-1,1],
             [ 0,-1],[ 0,0],[ 0,1],
             [ 1,-1],[ 1,0],[ 1,1]],
            device=device, dtype=torch.long)
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        res = 1 << depth
        nx, ny = n_coords[..., 0], n_coords[..., 1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
        n_keys = torch.full((N, 9), -1, dtype=torch.long, device=device)
        if valid.any():
            n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)
        idx = torch.searchsorted(keys, n_keys.clamp(min=0)).clamp(0, N - 1)
        found = valid & (keys[idx] == n_keys)
        idx[~found] = -1
        return idx


# ==========================================
# 6. Full Compressed AE Model
# ==========================================

class CompressedBottleneckAE(nn.Module):
    def __init__(self, in_c=1, hidden=128, emb_dim=128, max_depth=7, d_bottle=3):
        super().__init__()
        self.d_bottle = d_bottle
        self.max_depth = max_depth
        self.encoder = CompressedEncoder(in_c, hidden, emb_dim, max_depth, d_bottle)
        self.decoder = CompressedDecoder(hidden, emb_dim, in_c, max_depth, d_bottle)

    def forward(self, qt):
        E = self.encoder(qt)
        split_logits, val_pred = self.decoder(qt, E)
        return split_logits, val_pred, E

    def encode_compressed(self, qt):
        """Returns only the coarse embeddings and keys needed for standalone decode."""
        E = self.encoder(qt)
        coarse_E = E[:self.d_bottle + 1]
        coarse_keys = [qt.keys[d].clone() for d in range(self.d_bottle + 1)]
        return coarse_E, coarse_keys

    @torch.no_grad()
    def decode_from_latent(self, coarse_E, coarse_keys, split_thresh=0.0, min_level=6):
        """Standalone decode from coarse embeddings only. No encoder, no GT tree."""
        self.eval()
        # Pad emb_list to max_depth+1 with None
        emb_list = list(coarse_E) + [None] * (self.max_depth - self.d_bottle)
        pred_keys, val_preds = self.decoder.decode_autoregressive(
            emb_list, coarse_keys, self.max_depth, min_level, split_thresh)
        return pred_keys, val_preds


# ==========================================
# 7. Data Generation (same as feb17)
# ==========================================

MAX_LEVEL = 7
MIN_LEVEL = 6
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
        self.max_level, self.min_level = max_level, min_level
        self.children, self.val = [], None

    def gradient_subdivide(self, k1, k2, grad_threshold=0.3):
        if self.level >= self.max_level:
            return
        cx, cy = self.x + self.size / 2, self.y + self.size / 2
        force_split = self.level < self.min_level
        grad_mag = gradient_magnitude(cx, cy, k1, k2)
        max_grad = 2 * np.pi * np.sqrt(k1**2 + k2**2)
        normalized_grad = grad_mag / max_grad
        depth_factor = 1.0 + 0.1 * self.level
        if force_split or (normalized_grad > grad_threshold * depth_factor):
            half = self.size / 2
            self.children = [
                QuadNode(self.x,        self.y,        half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y,        half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x,        self.y + half, half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y + half, half, self.level + 1, self.max_level, self.min_level),
            ]
            for c in self.children:
                c.gradient_subdivide(k1, k2, grad_threshold)

    def collect_leaves(self, leaves_list, k1, k2):
        if not self.children:
            cx, cy = self.x + self.size / 2, self.y + self.size / 2
            self.val = target_function(cx, cy, k1, k2)
            leaves_list.append(self)
        else:
            for c in self.children:
                c.collect_leaves(leaves_list, k1, k2)


def generate_data(noise_std=NOISE_STD):
    k1, k2 = random.randint(1, 5), random.randint(1, 5)
    root = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root.gradient_subdivide(k1, k2)
    leaves = []
    root.collect_leaves(leaves, k1, k2)

    leaf_keys = [[] for _ in range(MAX_LEVEL + 1)]
    leaf_vals = [[] for _ in range(MAX_LEVEL + 1)]
    for node in leaves:
        d = node.level
        res = 1 << d
        ix = max(0, min(res - 1, int(node.x * res)))
        iy = max(0, min(res - 1, int(node.y * res)))
        leaf_keys[d].append(int(Morton2D.xy2key(ix, iy).item()))
        leaf_vals[d].append([float(node.val)])

    keys_t, clean_t, noisy_t = [], [], []
    for d in range(MAX_LEVEL + 1):
        if not leaf_keys[d]:
            keys_t.append(torch.empty((0,), dtype=torch.long))
            clean_t.append(torch.empty((0, 1), dtype=torch.float32))
            noisy_t.append(torch.empty((0, 1), dtype=torch.float32))
        else:
            kt = torch.tensor(leaf_keys[d], dtype=torch.long)
            c = torch.tensor(leaf_vals[d], dtype=torch.float32)
            keys_t.append(kt)
            clean_t.append(c)
            noisy_t.append(c + noise_std * torch.randn_like(c))

    return keys_t, noisy_t, clean_t, leaves, k1, k2


# ==========================================
# 8. Plotting (matching feb17 format)
# ==========================================

def plot_samples(sample_trees, output_dir, step, max_level):
    """3 samples × 4 columns: GT | Predicted | Error Map | MSE-by-Depth"""
    n_samples = len(sample_trees)
    if n_samples == 0:
        return

    fig, axes = plt.subplots(n_samples, 4, figsize=(24, 6 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    cmap = plt.get_cmap('viridis')
    loss_cmap = plt.get_cmap('hot')

    for row, sample in enumerate(sample_trees):
        all_losses = list(sample['loss_dict'].values())
        max_loss = max(all_losses) if all_losses else 1.0
        max_loss = max(max_loss, 1e-6)

        # GT
        ax1 = axes[row, 0]
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.set_aspect('equal'); ax1.axis('off')
        ax1.set_title(f"GT (Clean) Step {sample['step']}: k1={sample['k1']}, k2={sample['k2']}", fontsize=10)
        for node in sample['leaves']:
            color = cmap((node.val + 1) / 2)
            ax1.add_patch(patches.Rectangle((node.x, node.y), node.size, node.size,
                                            linewidth=0.5, edgecolor='black', facecolor=color))

        # Predicted
        ax2 = axes[row, 1]
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.set_aspect('equal'); ax2.axis('off')
        ax2.set_title(f"Predicted (Denoised) Step {sample['step']}", fontsize=10)
        for node in sample['leaves']:
            d = node.level; res = 1 << d
            ix = max(0, min(res - 1, int(node.x * res)))
            iy = max(0, min(res - 1, int(node.y * res)))
            pred_val = sample['pred_dict'].get((d, ix, iy), 0.0)
            color = cmap(np.clip((pred_val + 1) / 2, 0, 1))
            ax2.add_patch(patches.Rectangle((node.x, node.y), node.size, node.size,
                                            linewidth=0.5, edgecolor='black', facecolor=color))

        # Error map
        ax3 = axes[row, 2]
        ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
        ax3.set_aspect('equal'); ax3.axis('off')
        ax3.set_title(f"Per-Leaf Loss (MSE) Step {sample['step']}", fontsize=10)
        for node in sample['leaves']:
            d = node.level; res = 1 << d
            ix = max(0, min(res - 1, int(node.x * res)))
            iy = max(0, min(res - 1, int(node.y * res)))
            leaf_loss = sample['loss_dict'].get((d, ix, iy), 0.0)
            color = loss_cmap(np.clip(leaf_loss / max_loss, 0, 1))
            ax3.add_patch(patches.Rectangle((node.x, node.y), node.size, node.size,
                                            linewidth=0.5, edgecolor='black', facecolor=color))
        sm = plt.cm.ScalarMappable(cmap=loss_cmap, norm=plt.Normalize(0, max_loss))
        sm.set_array([])
        plt.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04).set_label('MSE Loss', fontsize=8)

        # MSE by depth
        ax4 = axes[row, 3]
        depths, avg_losses = [], []
        for d in range(max_level + 1):
            if sample['loss_by_depth'][d]:
                depths.append(d)
                avg_losses.append(np.mean(sample['loss_by_depth'][d]))
        if depths:
            bars = ax4.bar(depths, avg_losses, color='steelblue', edgecolor='black')
            ax4.set_xlabel('Depth Level'); ax4.set_ylabel('Avg MSE Loss')
            ax4.set_title(f"Avg Loss by Depth (Step {sample['step']})")
            ax4.set_xticks(range(max_level + 1))
            for bar, val in zip(bars, avg_losses):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/samples_step{step:04d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_losses(val_losses, split_losses, total_losses, output_dir, step, split_weight):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(val_losses, linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Value Loss (MSE)'); axes[0].set_title('Value Loss'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(split_losses, linewidth=1, alpha=0.7, color='coral')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Split Loss (BCE)'); axes[1].set_title('Split Loss'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(total_losses, linewidth=1, alpha=0.7, color='navy')
    axes[2].set_xlabel('Step'); axes[2].set_ylabel('Total Loss'); axes[2].set_title(f'Total Loss (val + {split_weight}*split)'); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f'Training Losses (Step {step})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_curve_step{step:04d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ar_inference(model, qt, leaves, k1, k2, output_dir, step, max_level):
    """Side-by-side: GT | Teacher-forced | Autoregressive (standalone decode)"""
    model.eval()
    with torch.no_grad():
        _, val_tf, _ = model(qt)
        coarse_E, coarse_keys = model.encode_compressed(qt)
        pred_keys, val_ar = model.decode_from_latent(coarse_E, coarse_keys)

    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # GT
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"GT f  k1={k1} k2={k2}  leaves={len(leaves)}", fontsize=10)
    for node in leaves:
        color = cmap((node.val + 1) / 2)
        ax.add_patch(patches.Rectangle((node.x, node.y), node.size, node.size,
                                       linewidth=0.5, edgecolor='black', facecolor=color))

    # Teacher-forced
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Teacher-forced reconstruction", fontsize=10)
    for d in range(max_level + 1):
        mask = qt.leaf_mask[d]
        if mask is None or not mask.any() or val_tf[d] is None:
            continue
        kd = qt.keys[d][mask].cpu()
        vals = val_tf[d][mask].detach().cpu().flatten().numpy()
        ix, iy = Morton2D.key2xy(kd, depth=d)
        res = 1 << d; sz = 1.0 / res
        for i in range(len(kd)):
            color = cmap(np.clip((vals[i] + 1) / 2, 0, 1))
            ax.add_patch(patches.Rectangle((ix[i].item() * sz, iy[i].item() * sz),
                                           sz, sz, linewidth=0.3, edgecolor='k', facecolor=color))

    # Autoregressive (standalone)
    ax = axes[2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.axis('off')
    n_ar_leaves = sum(pk.numel() for pk in pred_keys if pk is not None and pk.numel() > 0)
    ax.set_title(f"Autoregressive (standalone)  nodes={n_ar_leaves}", fontsize=10)

    for d in range(max_level + 1):
        pk = pred_keys[d]
        vp = val_ar[d]
        if pk is None or pk.numel() == 0 or vp is None or vp.numel() == 0:
            continue
        # Determine leaves: nodes without children in next depth
        if d < max_level and pred_keys[d + 1] is not None and pred_keys[d + 1].numel() > 0:
            parents_of_next = pred_keys[d + 1] >> 2
            is_leaf = ~torch.isin(pk, parents_of_next)
        else:
            is_leaf = torch.ones(len(pk), dtype=torch.bool)

        if not is_leaf.any():
            continue

        kd = pk[is_leaf].cpu()
        vals = vp[is_leaf].detach().cpu().flatten().numpy()
        ix, iy = Morton2D.key2xy(kd, depth=d)
        res = 1 << d; sz = 1.0 / res
        for i in range(len(kd)):
            color = cmap(np.clip((vals[i] + 1) / 2, 0, 1))
            ax.add_patch(patches.Rectangle((ix[i].item() * sz, iy[i].item() * sz),
                                           sz, sz, linewidth=0.3, edgecolor='k', facecolor=color))

    fig.suptitle(f"Inference Comparison — Step {step}  |  D_BOTTLE={model.d_bottle}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/infer_step{step:04d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    model.train()


# ==========================================
# 9. Training Loop (Two-Phase)
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_bottle', type=int, default=2, help='Bottleneck depth')
    parser.add_argument('--emb_dim', type=int, default=10, help='Embedding dimension per node')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension (internal)')
    args = parser.parse_args()

    D_BOTTLE = args.d_bottle
    EMB_DIM = args.emb_dim
    HIDDEN = args.hidden
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_nodes = sum(4**d for d in range(D_BOTTLE + 1))
    total_floats = max_nodes * EMB_DIM

    print(f"=" * 70)
    print(f"ULTRA Compressed Bottleneck AE")
    print(f"D_BOTTLE={D_BOTTLE}  |  emb_dim={EMB_DIM}  |  hidden={HIDDEN}")
    print(f"Max nodes at bottleneck: {max_nodes}  |  Latent: {total_floats} floats")
    print(f"Device: {device}")
    print(f"=" * 70)

    model = CompressedBottleneckAE(
        in_c=1, hidden=HIDDEN, emb_dim=EMB_DIM,
        max_depth=MAX_LEVEL, d_bottle=D_BOTTLE
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"plots/ultra_d{D_BOTTLE}_e{EMB_DIM}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}\n")

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    split_weight = 0.5

    # Phase 1: Teacher forcing
    phase1_steps = 4000
    phase2_steps = 4000

    val_losses, split_losses, total_losses = [], [], []
    sample_trees = []
    best = {'loss': float('inf'), 'step': 0, 'counter': 0}
    best_path = f'{output_dir}/best_model.pt'

    # ─── PHASE 1: Teacher Forcing ────────────────────────────────────────
    print("=" * 40)
    print("PHASE 1: Teacher Forcing")
    print("=" * 40)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_steps, eta_min=1e-5)

    for step in range(phase1_steps):
        keys, noisy, clean, leaves, k1, k2 = generate_data()
        qt = Quadtree(MAX_LEVEL, device)
        qt.build_from_leaves(keys, noisy, clean)

        optimizer.zero_grad()
        split_logits, val_pred, E = model(qt)

        # Value loss on leaf nodes
        L_val = torch.tensor(0.0, device=device)
        n_val = 0
        for d in range(MAX_LEVEL + 1):
            mask = qt.leaf_mask[d]
            if mask is not None and mask.any():
                L_val = L_val + mse(val_pred[d][mask], qt.values_gt[d][mask])
                n_val += 1
        if n_val > 0:
            L_val = L_val / n_val

        # Split loss
        L_split = torch.tensor(0.0, device=device)
        n_split = 0
        for d in range(MAX_LEVEL):
            if qt.split_gt[d] is not None and qt.split_gt[d].numel() > 0:
                if split_logits[d] is not None and split_logits[d].numel() > 0:
                    L_split = L_split + bce(split_logits[d], qt.split_gt[d])
                    n_split += 1
        if n_split > 0:
            L_split = L_split / n_split

        gain_reg = 1e-4 * (model.encoder.depth_gain ** 2).mean()
        L_total = L_val + split_weight * L_split + gain_reg

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        val_losses.append(L_val.item())
        split_losses.append(L_split.item())
        total_losses.append(L_total.item())

        if L_total.item() < best['loss']:
            best.update({'loss': L_total.item(), 'step': step, 'counter': 0})
            torch.save({
                'step': step, 'phase': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': best['loss'],
            }, best_path)
        else:
            best['counter'] += 1

        # Compression stats
        latent_nodes = sum(E[d].shape[0] for d in range(D_BOTTLE + 1) if E[d] is not None)
        total_nodes = sum(len(qt.keys[d]) for d in range(MAX_LEVEL + 1))

        print(f"[P1] Step {step:4d} | val={L_val.item():.6f} split={L_split.item():.6f} "
              f"total={L_total.item():.6f} | k1={k1} k2={k2} leaves={len(leaves)} | "
              f"compress={latent_nodes}/{total_nodes} | best={best['loss']:.6f}@{best['step']} "
              f"patience={best['counter']}/300")

        # Store sample trees for plotting
        if step % 100 == 0:
            sample_trees.append({'qt': qt, 'leaves': leaves, 'k1': k1, 'k2': k2, 'step': step})
            if len(sample_trees) > 3:
                sample_trees.pop(0)

        # Plotting
        if step % 100 == 0 or step == phase1_steps - 1:
            with torch.no_grad():
                for sample in sample_trees:
                    _, val_pred_viz, _ = model(sample['qt'])
                    sample['pred_dict'] = {}
                    sample['loss_dict'] = {}
                    sample['loss_by_depth'] = {d: [] for d in range(MAX_LEVEL + 1)}
                    for d in range(MAX_LEVEL + 1):
                        if sample['qt'].keys[d].numel() == 0:
                            continue
                        mask = sample['qt'].leaf_mask[d]
                        if mask is None or not mask.any():
                            continue
                        kd = sample['qt'].keys[d][mask].cpu()
                        ix, iy = Morton2D.key2xy(kd, depth=d)
                        pr = val_pred_viz[d][mask].cpu().numpy().flatten()
                        gt = sample['qt'].values_gt[d][mask].cpu().numpy().flatten()
                        for i in range(len(kd)):
                            key = (d, int(ix[i].item()), int(iy[i].item()))
                            sample['pred_dict'][key] = pr[i]
                            leaf_loss = (pr[i] - gt[i]) ** 2
                            sample['loss_dict'][key] = leaf_loss
                            sample['loss_by_depth'][d].append(leaf_loss)

            plot_samples(sample_trees, output_dir, step, MAX_LEVEL)
            plot_losses(val_losses, split_losses, total_losses, output_dir, step, split_weight)
            print(f"  -> Plots saved at step {step}")

        if step % 500 == 0 or step == phase1_steps - 1:
            plot_ar_inference(model, qt, leaves, k1, k2, output_dir, step, MAX_LEVEL)
            print(f"  -> AR inference plot saved at step {step}")

        if best['counter'] >= 300:
            print(f"\nPhase 1 early stop at step {step}")
            break

    print(f"\nPhase 1 complete. Best loss: {best['loss']:.6f} at step {best['step']}")

    # ─── PHASE 2: Autoregressive Fine-tuning ─────────────────────────────
    print("\n" + "=" * 40)
    print("PHASE 2: Autoregressive Fine-tuning")
    print("=" * 40)

    # Load best Phase 1 model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=phase2_steps, eta_min=1e-6)
    best2 = {'loss': float('inf'), 'step': 0, 'counter': 0}
    best_path2 = f'{output_dir}/best_model_phase2.pt'

    for step in range(phase2_steps):
        global_step = phase1_steps + step

        keys, noisy, clean, leaves, k1, k2 = generate_data()
        qt = Quadtree(MAX_LEVEL, device)
        qt.build_from_leaves(keys, noisy, clean)

        optimizer2.zero_grad()

        # Encode
        E = model.encoder(qt)
        coarse_E = E[:D_BOTTLE + 1]
        coarse_keys = [qt.keys[d].clone() for d in range(D_BOTTLE + 1)]

        # Autoregressive decode (builds its own topology)
        emb_list = list(coarse_E) + [None] * (MAX_LEVEL - D_BOTTLE)
        pred_keys, val_ar = model.decoder.decode_autoregressive(
            emb_list, coarse_keys, MAX_LEVEL, MIN_LEVEL, split_thresh=0.0)

        # Also do teacher-forced for split supervision
        split_logits, val_pred = model.decoder(qt, E)

        # Phase 2 loss: teacher-forced split + AR value reconstruction
        L_val_tf = torch.tensor(0.0, device=device)
        n_val = 0
        for d in range(MAX_LEVEL + 1):
            mask = qt.leaf_mask[d]
            if mask is not None and mask.any() and val_pred[d] is not None:
                L_val_tf = L_val_tf + mse(val_pred[d][mask], qt.values_gt[d][mask])
                n_val += 1
        if n_val > 0:
            L_val_tf = L_val_tf / n_val

        L_split = torch.tensor(0.0, device=device)
        n_split = 0
        for d in range(MAX_LEVEL):
            if qt.split_gt[d] is not None and qt.split_gt[d].numel() > 0:
                if split_logits[d] is not None and split_logits[d].numel() > 0:
                    L_split = L_split + bce(split_logits[d], qt.split_gt[d])
                    n_split += 1
        if n_split > 0:
            L_split = L_split / n_split

        gain_reg = 1e-4 * (model.encoder.depth_gain ** 2).mean()
        L_total = L_val_tf + split_weight * L_split + gain_reg

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer2.step()
        scheduler2.step()

        val_losses.append(L_val_tf.item())
        split_losses.append(L_split.item())
        total_losses.append(L_total.item())

        if L_total.item() < best2['loss']:
            best2.update({'loss': L_total.item(), 'step': global_step, 'counter': 0})
            torch.save({
                'step': global_step, 'phase': 2,
                'model_state_dict': model.state_dict(),
                'total_loss': best2['loss'],
            }, best_path2)
        else:
            best2['counter'] += 1

        latent_nodes = sum(coarse_E[d].shape[0] for d in range(D_BOTTLE + 1) if coarse_E[d] is not None)
        total_nodes = sum(len(qt.keys[d]) for d in range(MAX_LEVEL + 1))

        print(f"[P2] Step {global_step:4d} | val={L_val_tf.item():.6f} split={L_split.item():.6f} "
              f"total={L_total.item():.6f} | k1={k1} k2={k2} leaves={len(leaves)} | "
              f"compress={latent_nodes}/{total_nodes} | best={best2['loss']:.6f}@{best2['step']} "
              f"patience={best2['counter']}/300")

        if step % 100 == 0:
            sample_trees.append({'qt': qt, 'leaves': leaves, 'k1': k1, 'k2': k2, 'step': global_step})
            if len(sample_trees) > 3:
                sample_trees.pop(0)

        if step % 100 == 0 or step == phase2_steps - 1:
            with torch.no_grad():
                for sample in sample_trees:
                    _, val_pred_viz, _ = model(sample['qt'])
                    sample['pred_dict'] = {}
                    sample['loss_dict'] = {}
                    sample['loss_by_depth'] = {d: [] for d in range(MAX_LEVEL + 1)}
                    for d in range(MAX_LEVEL + 1):
                        if sample['qt'].keys[d].numel() == 0:
                            continue
                        mask = sample['qt'].leaf_mask[d]
                        if mask is None or not mask.any():
                            continue
                        kd = sample['qt'].keys[d][mask].cpu()
                        ix, iy = Morton2D.key2xy(kd, depth=d)
                        pr = val_pred_viz[d][mask].cpu().numpy().flatten()
                        gt = sample['qt'].values_gt[d][mask].cpu().numpy().flatten()
                        for i in range(len(kd)):
                            key = (d, int(ix[i].item()), int(iy[i].item()))
                            sample['pred_dict'][key] = pr[i]
                            leaf_loss = (pr[i] - gt[i]) ** 2
                            sample['loss_dict'][key] = leaf_loss
                            sample['loss_by_depth'][d].append(leaf_loss)

            plot_samples(sample_trees, output_dir, global_step, MAX_LEVEL)
            plot_losses(val_losses, split_losses, total_losses, output_dir, global_step, split_weight)
            print(f"  -> Plots saved at step {global_step}")

        if step % 500 == 0 or step == phase2_steps - 1:
            plot_ar_inference(model, qt, leaves, k1, k2, output_dir, global_step, MAX_LEVEL)
            print(f"  -> AR inference plot saved at step {global_step}")

        if best2['counter'] >= 300:
            print(f"\nPhase 2 early stop at step {global_step}")
            break

    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"Phase 1 best: {best['loss']:.6f} at step {best['step']}")
    print(f"Phase 2 best: {best2['loss']:.6f} at step {best2['step']}")
    print(f"Bottleneck: D={D_BOTTLE}, max {max_nodes} nodes × {EMB_DIM} dims = {total_floats} floats")
    print(f"Checkpoints: {best_path}, {best_path2}")
    print(f"Plots: {output_dir}/")


if __name__ == '__main__':
    main()
