"""
20260409-quadtree-u-crossattn.py

Option C: Cross-Attention Bottleneck Compressed AE for u only.

Goal: compress u into a small global latent z ∈ R^96 for use in
Gauss-Newton nonlinear ROM. The decoder maps z → (tree topology + nodal values)
with NO quadtree structure given at inference — it builds the tree from scratch.

Architecture:
  Encoder (bottom-up QuadConv → cross-attention bottleneck):
    - Same bottom-up QuadConv as Feb18 produces h[d] per node
    - Flatten all node embeddings across all depths into one set
    - m=8 learned query tokens cross-attend into the node set
      (permutation-invariant, handles variable tree sizes)
    - Output: z = flatten(cross_attn_output) ∈ R^(8×128) → linear → R^96

  Decoder (top-down, conditioned on z):
    - z is decoded through a small MLP into per-depth context vectors c[d] ∈ R^hidden
    - At each depth d: fuse(c[d] broadcast, positional_enc) → hidden
    - Predict split logit and value
    - Children propagated via child_head (teacher forcing at training)

  Latent dim: 96 total
  Jacobian: dense (all nodes at all depths depend on all dims of z)

Training:
  - Only trains model_u
  - Same Poisson data generation as Feb18
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

        self.parent_idx[0] = None
        for d in range(1, self.max_depth + 1):
            kd = self.keys[d]; kp = self.keys[d-1]
            if kd.numel() == 0:
                self.parent_idx[d] = torch.empty((0,), dtype=torch.long, device=self.device)
                continue
            pidx = torch.searchsorted(kp, kd >> 2).clamp(0, len(kp)-1)
            self.parent_idx[d] = pidx

        for d in range(self.max_depth + 1):
            self._construct_neigh(d)

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


# Latent configuration
NUM_QUERIES  = 8     # number of cross-attention query tokens
QUERY_DIM    = 128   # dimension of each query token
LATENT_TOTAL = 96    # final compressed dim after linear projection


class CrossAttentionBottleneck(nn.Module):
    """
    Compresses a variable-size set of node embeddings into a fixed-size latent.

    m learned query tokens cross-attend into the node embeddings (keys/values).
    Output: (m, query_dim) → flattened + linear → z ∈ R^latent_total

    This is permutation-invariant w.r.t. the node set and handles variable sizes.
    """
    def __init__(self, node_dim=128, num_queries=NUM_QUERIES, query_dim=QUERY_DIM,
                 latent_total=LATENT_TOTAL, num_heads=4):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim   = query_dim

        # Learned query tokens
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)

        # Project node embeddings to key/value space
        self.key_proj = nn.Linear(node_dim, query_dim)
        self.val_proj = nn.Linear(node_dim, query_dim)

        # Multi-head cross attention
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(query_dim)

        # Compress to latent
        self.to_latent = nn.Sequential(
            nn.Linear(num_queries * query_dim, latent_total * 2),
            nn.ReLU(),
            nn.Linear(latent_total * 2, latent_total),
        )
        self.latent_norm = nn.LayerNorm(latent_total)

    def forward(self, node_embeddings):
        """
        node_embeddings : (N_total, node_dim)  — all nodes across all depths
        returns z       : (latent_total,)
        """
        N = node_embeddings.shape[0]

        # Query tokens: (1, m, query_dim) — batch size 1
        Q = self.queries.unsqueeze(0)                          # (1, m, qd)
        K = self.key_proj(node_embeddings).unsqueeze(0)        # (1, N, qd)
        V = self.val_proj(node_embeddings).unsqueeze(0)        # (1, N, qd)

        # Cross-attention: queries attend to all nodes
        attn_out, _ = self.attn(Q, K, V)                      # (1, m, qd)
        attn_out = self.norm(attn_out.squeeze(0))              # (m, qd)

        # Flatten and compress
        flat = attn_out.view(1, -1)                            # (1, m*qd)
        z = self.to_latent(flat).squeeze(0)                    # (latent_total,)
        z = self.latent_norm(z)
        return z


class CrossAttnEncoder(nn.Module):
    """
    Bottom-up QuadConv encoder → CrossAttentionBottleneck → z ∈ R^96.
    """
    def __init__(self, in_c=1, hidden=128, max_depth=7, pos_freqs=6,
                 num_queries=NUM_QUERIES, query_dim=QUERY_DIM, latent_total=LATENT_TOTAL):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.max_depth = max_depth
        self.hidden    = hidden
        pos_dim = 3 + 2 * pos_freqs * 3

        self.in_proj = nn.Linear(in_c + pos_dim, hidden)
        self.convs   = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool    = QuadPool()
        self.node_norm = nn.LayerNorm(hidden)

        self.bottleneck = CrossAttentionBottleneck(
            node_dim=hidden, num_queries=num_queries,
            query_dim=query_dim, latent_total=latent_total
        )

    def forward(self, qt):
        h = [None] * (self.max_depth + 1)

        for d in range(self.max_depth + 1):
            fin = qt.features_in[d]
            kd  = qt.keys[d]
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
            if d - 1 >= 1 and h[d-1].numel() > 0:
                h[d-1] = F.relu(self.convs[d-1](h[d-1], qt, d-1))

        # Collect all non-empty node embeddings across all depths
        all_nodes = [h[d] for d in range(self.max_depth + 1) if h[d].numel() > 0]
        if len(all_nodes) == 0:
            return torch.zeros(LATENT_TOTAL, device=qt.device), h
        node_set = torch.cat(all_nodes, dim=0)   # (N_total, hidden)
        node_set = self.node_norm(node_set)

        z = self.bottleneck(node_set)   # (latent_total,)
        return z, h


class CrossAttnDecoder(nn.Module):
    """
    Top-down decoder conditioned on z ∈ R^96.

    z is decoded into per-depth context vectors c[d] via a learned MLP,
    then broadcast to all nodes at each depth and fused with positional encoding.

    Teacher forcing on tree topology at training time.
    """
    def __init__(self, hidden=128, out_c=1, max_depth=7, pos_freqs=6,
                 latent_total=LATENT_TOTAL):
        super().__init__()
        self.max_depth   = max_depth
        self.pos_freqs   = pos_freqs
        self.hidden      = hidden
        self.latent_total = latent_total
        pos_dim = 3 + 2 * pos_freqs * 3

        # Decode z into per-depth context: z → c[d] ∈ R^hidden for each depth
        self.z_to_context = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_total, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(max_depth + 1)
        ])

        # Fuse context + pos → hidden
        self.fuse = nn.Sequential(
            nn.Linear(hidden + pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.mix_convs  = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4 * hidden))
        self.val_head   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))

    def forward(self, qt, z):
        """
        qt : Quadtree (GT topology for teacher forcing)
        z  : (latent_total,) global latent
        """
        split_logits = [None] * self.max_depth
        val_pred     = [None] * (self.max_depth + 1)
        device       = qt.device

        for d in range(self.max_depth + 1):
            Nd = len(qt.keys[d])
            if Nd == 0:
                val_pred[d] = torch.zeros((0, 1), device=device)
                if d < self.max_depth:
                    split_logits[d] = torch.zeros((0,), device=device)
                continue

            kd = qt.keys[d]
            pos = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, device), self.pos_freqs)

            # Context from z for this depth, broadcast to all nodes
            c_d = self.z_to_context[d](z.unsqueeze(0)).expand(Nd, -1)  # (Nd, hidden)

            h = self.fuse(torch.cat([c_d, pos], dim=1))                  # (Nd, hidden)
            if d >= 1:
                h = F.relu(self.mix_convs[d](h, qt, d))

            val_pred[d] = self.val_head(h)

            if d == self.max_depth:
                break

            split_logits[d] = self.split_head(h).squeeze(-1)

            ch = qt.children_idx[d]
            has_child = (ch != -1).any(dim=1)
            N_next = len(qt.keys[d+1])
            h_next = torch.zeros((N_next, self.hidden), device=device)

            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, self.hidden)
                parent_rows = torch.nonzero(has_child).squeeze(-1)
                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1:
                            h_next[ci] = child_feats[t, c]

        return split_logits, val_pred


class CrossAttnAE(nn.Module):
    """
    Full cross-attention bottleneck autoencoder.
    Encodes u quadtree → z ∈ R^96, decodes z → (split logits, val predictions).
    """
    def __init__(self, in_c=1, hidden=128, max_depth=7, pos_freqs=6,
                 num_queries=NUM_QUERIES, query_dim=QUERY_DIM, latent_total=LATENT_TOTAL):
        super().__init__()
        self.encoder = CrossAttnEncoder(in_c=in_c, hidden=hidden, max_depth=max_depth,
                                        pos_freqs=pos_freqs, num_queries=num_queries,
                                        query_dim=query_dim, latent_total=latent_total)
        self.decoder = CrossAttnDecoder(hidden=hidden, out_c=in_c, max_depth=max_depth,
                                        pos_freqs=pos_freqs, latent_total=latent_total)
        self.latent_total = latent_total

    def forward(self, qt):
        z, _ = self.encoder(qt)
        split_logits, val_pred = self.decoder(qt, z)
        return split_logits, val_pred, z

    @torch.no_grad()
    def encode(self, qt):
        self.encoder.eval()
        z, _ = self.encoder(qt)
        return z   # (96,)

    @torch.no_grad()
    def decode(self, qt, z):
        self.decoder.eval()
        return self.decoder(qt, z)


# ==========================================
# 5. Poisson Data Generation (u only)
# ==========================================

MAX_LEVEL = 7
MIN_LEVEL = 6
NOISE_STD = 0.05


def u_function(x, y, k1, k2):
    denom = 4 * np.pi**2 * (k1**2 + k2**2)
    return -np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y) / denom


def gradient_magnitude_u(x, y, k1, k2):
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
        self.u_val = None

    def gradient_subdivide_u(self, k1, k2, grad_threshold=0.3):
        if self.level >= self.max_level:
            return
        cx = self.x + self.size / 2
        cy = self.y + self.size / 2
        force_split = self.level < self.min_level
        grad_mag = gradient_magnitude_u(cx, cy, k1, k2)
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

    def collect_leaves_u(self, leaves_list, k1, k2):
        if not self.children:
            cx = self.x + self.size / 2
            cy = self.y + self.size / 2
            self.u_val = u_function(cx, cy, k1, k2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves_u(leaves_list, k1, k2)


def generate_u_sample(noise_std=NOISE_STD, grad_threshold=0.3):
    k1 = random.randint(1, 5)
    k2 = random.randint(1, 5)
    denom = 4 * np.pi**2 * (k1**2 + k2**2)

    root = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root.gradient_subdivide_u(k1, k2, grad_threshold=grad_threshold)
    u_leaves = []
    root.collect_leaves_u(u_leaves, k1, k2)

    u_leaf_keys = [[] for _ in range(MAX_LEVEL + 1)]
    u_vals_raw  = [[] for _ in range(MAX_LEVEL + 1)]
    for node in u_leaves:
        d = node.level
        res = 1 << d
        ix = max(0, min(res-1, int(node.x * res)))
        iy = max(0, min(res-1, int(node.y * res)))
        k  = Morton2D.xy2key(ix, iy)
        u_leaf_keys[d].append(int(k.item()))
        u_vals_raw[d].append([float(node.u_val) * denom])

    u_leaf_keys_by_depth  = []
    u_vals_clean_by_depth = []
    u_vals_noisy_by_depth = []
    for d in range(MAX_LEVEL + 1):
        if len(u_leaf_keys[d]) == 0:
            u_leaf_keys_by_depth.append(torch.empty((0,), dtype=torch.long))
            u_vals_clean_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
            u_vals_noisy_by_depth.append(torch.empty((0, 1), dtype=torch.float32))
        else:
            keys_t  = torch.tensor(u_leaf_keys[d], dtype=torch.long)
            u_clean = torch.tensor(u_vals_raw[d], dtype=torch.float32)
            u_noisy = u_clean + noise_std * torch.randn_like(u_clean)
            u_leaf_keys_by_depth.append(keys_t)
            u_vals_clean_by_depth.append(u_clean)
            u_vals_noisy_by_depth.append(u_noisy)

    return u_leaf_keys_by_depth, u_vals_clean_by_depth, u_vals_noisy_by_depth, u_leaves, k1, k2, denom


# ==========================================
# 6. Loss
# ==========================================

def compute_loss(model, qt, device, mse, bce, split_weight=0.5):
    split_logits, val_pred, z = model(qt)

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

    # L2 on z to keep latent bounded for GN convergence
    z_reg = 1e-4 * (z ** 2).mean()

    L_total = L_val + split_weight * L_split + z_reg
    return L_total, L_val, L_split, z


# ==========================================
# 7. Training
# ==========================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    print(f"Option C: Cross-Attention Bottleneck AE  |  latent dim = {LATENT_TOTAL}")
    print(f"  num_queries={NUM_QUERIES}, query_dim={QUERY_DIM}")

    model = CrossAttnAE(in_c=1, hidden=128, max_depth=MAX_LEVEL, pos_freqs=6,
                        num_queries=NUM_QUERIES, query_dim=QUERY_DIM,
                        latent_total=LATENT_TOTAL).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    split_weight = 0.5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"plots/optionC_crossattn_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}\n")

    num_steps = 3000
    patience  = 150

    history = {'val': [], 'split': [], 'total': [], 'z_norm': []}
    best = {'loss': float('inf'), 'step': 0, 'counter': 0}
    best_path = f'{output_dir}/best_model.pt'
    sample_buffer = []

    for step in range(num_steps):
        u_keys, u_clean, u_noisy, u_leaves, k1, k2, denom = generate_u_sample()

        qt = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt.build_from_leaves(u_keys, u_noisy, u_clean)

        optimizer.zero_grad()
        L_total, L_val, L_split, z = compute_loss(model, qt, device, mse, bce, split_weight)
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        z_norm = z.detach().norm().item()
        history['val'].append(L_val.item())
        history['split'].append(L_split.item())
        history['total'].append(L_total.item())
        history['z_norm'].append(z_norm)

        if L_total.item() < best['loss']:
            best.update({'loss': L_total.item(), 'step': step, 'counter': 0})
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': L_total.item(),
                'val_loss': L_val.item(),
                'split_loss': L_split.item(),
                'num_queries': NUM_QUERIES,
                'query_dim': QUERY_DIM,
                'latent_total': LATENT_TOTAL,
            }, best_path)
        else:
            best['counter'] += 1

        print(
            f"Step {step:4d}: val={L_val.item():.5f} split={L_split.item():.5f} "
            f"total={L_total.item():.5f}  |z|={z_norm:.3f}  "
            f"best={best['loss']:.5f}@{best['step']}  patience={best['counter']}/{patience}  "
            f"k1={k1} k2={k2} leaves={len(u_leaves)}"
        )

        if best['counter'] >= patience:
            print(f"\nEarly stopping at step {step}.")
            break

        if step % 100 == 0:
            sample_buffer.append({
                'qt': qt, 'u_leaves': u_leaves,
                'k1': k1, 'k2': k2, 'denom': denom, 'step': step,
                'z': z.detach().cpu()
            })
            if len(sample_buffer) > 3:
                sample_buffer.pop(0)

        if step % 100 == 0 or step == num_steps - 1:
            _plot_samples(sample_buffer, model, output_dir, step)
            _plot_losses(history, output_dir, step)

    print(f"\nTraining complete. Best: step={best['step']}, loss={best['loss']:.6f}")
    print(f"  Saved: {best_path}")

    validate(best_path, output_dir, num_samples=10, device=device)


# ==========================================
# 8. Plotting
# ==========================================

def _collect_preds(model, qt):
    with torch.no_grad():
        _, val_pred, z = model(qt)
    pred_dict = {}
    loss_dict = {}
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
            loss_dict[key] = (pr[i] - gt[i]) ** 2
    return pred_dict, loss_dict, z.detach().cpu()


def _draw_field(ax, leaves, val_fn, title, cmap, vmin=None, vmax=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    values = [val_fn(node) for node in leaves]
    if not values:
        ax.set_title(title, fontsize=9); return
    if vmin is None: vmin = min(values)
    if vmax is None: vmax = max(values)
    val_range = max(vmax - vmin, 1e-10)
    ax.set_title(f"{title}\n[{vmin:.4f}, {vmax:.4f}]", fontsize=9)
    for node, v in zip(leaves, values):
        color = cmap(np.clip((v - vmin) / val_range, 0, 1))
        rect = patches.Rectangle((node.x, node.y), node.size, node.size,
                                  linewidth=0.3, edgecolor='black', facecolor=color)
        ax.add_patch(rect)


def _draw_pred_field(ax, qt, pred_dict, title, cmap, d_range=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    vals = list(pred_dict.values())
    if not vals:
        ax.set_title(title, fontsize=9); return
    vmin, vmax = (min(vals), max(vals)) if d_range is None else d_range
    val_range = max(vmax - vmin, 1e-10)
    ax.set_title(f"{title}\n[{vmin:.4f}, {vmax:.4f}]", fontsize=9)
    for d in range(MAX_LEVEL + 1):
        if qt.keys[d].numel() == 0: continue
        mask = qt.leaf_mask[d]
        if mask is None or not mask.any(): continue
        kd = qt.keys[d][mask].cpu()
        ix, iy = Morton2D.key2xy(kd, depth=d)
        res = 1 << d
        sz  = 1.0 / res
        for i in range(len(kd)):
            key = (d, int(ix[i].item()), int(iy[i].item()))
            v = pred_dict.get(key, 0.0)
            color = cmap(np.clip((v - vmin) / val_range, 0, 1))
            rect = patches.Rectangle((ix[i].item() * sz, iy[i].item() * sz), sz, sz,
                                      linewidth=0.3, edgecolor='black', facecolor=color)
            ax.add_patch(rect)


def _plot_samples(sample_buffer, model, output_dir, step):
    if not sample_buffer:
        return
    n = len(sample_buffer)
    cmap = plt.get_cmap('viridis')
    loss_cmap = plt.get_cmap('hot')

    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, samp in enumerate(sample_buffer):
        qt     = samp['qt']
        leaves = samp['u_leaves']
        k1, k2 = samp['k1'], samp['k2']
        s      = samp['step']

        pred_dict, loss_dict, z = _collect_preds(model, qt)

        # Col 0: GT u
        _draw_field(axes[row, 0], leaves,
                    lambda node: node.u_val,
                    f"GT u (k1={k1},k2={k2}) step={s}", cmap)

        # Col 1: Pred u
        _draw_pred_field(axes[row, 1], qt, pred_dict, "Pred u (normalized)", cmap)

        # Col 2: Loss map
        max_loss = max(loss_dict.values()) if loss_dict else 1.0
        _draw_pred_field(axes[row, 2], qt,
                         {k: v / max(max_loss, 1e-10) for k, v in loss_dict.items()},
                         f"Loss map (max={max_loss:.5f})", loss_cmap, d_range=(0, 1))

        # Col 3: Latent z bar chart
        ax = axes[row, 3]
        z_np = z.numpy()
        ax.bar(range(len(z_np)), z_np, color='darkorange', alpha=0.7)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(f"Latent z (dim={len(z_np)})\nstd={z_np.std():.3f}", fontsize=9)
        ax.set_xlabel("dim"); ax.set_ylabel("value")

    plt.suptitle(f"Option C: Cross-Attention Bottleneck AE — step {step}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/samples_step{step:04d}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {output_dir}/samples_step{step:04d}.png")


def _plot_losses(history, output_dir, step):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, key, color, label in zip(
        axes,
        ['total', 'val', 'split', 'z_norm'],
        ['darkred', 'tomato', 'darkorange', 'green'],
        ['Total Loss', 'Val Loss (MSE)', 'Split Loss (BCE)', '|z| (latent norm)']
    ):
        ax.plot(history[key], linewidth=1, alpha=0.8, color=color)
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
        if len(history[key]) > 0:
            ax.set_title(f"{label}\nfinal={history[key][-1]:.5f}", fontsize=9)

    plt.suptitle(f'Option C Training Losses — step {step}  (latent_dim={LATENT_TOTAL})', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_curve_step{step:04d}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {output_dir}/loss_curve_step{step:04d}.png")


def validate(model_path, output_dir, num_samples=10, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Validation: Option C Cross-Attention Bottleneck AE")
    print(f"{'='*60}")

    ckpt = torch.load(model_path, map_location=device)
    model = CrossAttnAE(in_c=1, hidden=128, max_depth=MAX_LEVEL, pos_freqs=6,
                        num_queries=NUM_QUERIES, query_dim=QUERY_DIM,
                        latent_total=LATENT_TOTAL).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(val_dir, exist_ok=True)

    mse_fn = nn.MSELoss()
    all_z = []
    per_depth_losses = {d: [] for d in range(MAX_LEVEL + 1)}

    for i in range(num_samples):
        u_keys, u_clean, u_noisy, u_leaves, k1, k2, denom = generate_u_sample()
        qt = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt.build_from_leaves(u_keys, u_noisy, u_clean)

        with torch.no_grad():
            _, val_pred, z = model(qt)

        all_z.append(z.cpu().numpy())

        for d in range(MAX_LEVEL + 1):
            mask = qt.leaf_mask[d]
            if mask is None or mask.numel() == 0 or not mask.any():
                continue
            loss_d = mse_fn(val_pred[d][mask], qt.values_gt[d][mask]).item()
            per_depth_losses[d].append(loss_d)

        print(f"  Sample {i+1}/{num_samples}: k1={k1}, k2={k2}, leaves={len(u_leaves)}, |z|={z.norm():.3f}")

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    depths = [d for d in range(MAX_LEVEL + 1) if per_depth_losses[d]]
    avg_losses = [np.mean(per_depth_losses[d]) for d in depths]
    axes[0].bar(depths, avg_losses, color='darkorange', alpha=0.8)
    axes[0].set_xlabel('Depth'); axes[0].set_ylabel('Avg MSE')
    axes[0].set_title('Avg Loss by Depth'); axes[0].grid(True, alpha=0.3)

    all_z_arr = np.stack(all_z)
    axes[1].imshow(all_z_arr, aspect='auto', cmap='RdBu_r',
                   vmin=-all_z_arr.std()*3, vmax=all_z_arr.std()*3)
    axes[1].set_xlabel('Latent dim'); axes[1].set_ylabel('Sample')
    axes[1].set_title(f'Latent z across {num_samples} samples\n(red=+, blue=-)')

    z_norms = [np.linalg.norm(z) for z in all_z]
    axes[2].hist(z_norms, bins=10, color='darkorange', alpha=0.8, edgecolor='black')
    axes[2].set_xlabel('||z||'); axes[2].set_ylabel('Count')
    axes[2].set_title(f'Latent norm distribution\nmean={np.mean(z_norms):.3f}')

    plt.suptitle(f'Option C Validation Summary (latent_dim={LATENT_TOTAL})', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{val_dir}/validation_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  -> Saved {val_dir}/validation_summary.png")


if __name__ == '__main__':
    main()
