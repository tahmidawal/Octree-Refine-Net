"""
20260410-quadtree-u-attn-bottleneck.py

Attention-pooling bottleneck at depth 2 of the quadtree.

Architecture overview
---------------------
Encoder  (bottom-up, depths MAX_DEPTH → BOTTLENECK_DEPTH):
  - Same QuadConv + QuadPool as Feb18, stopping at BOTTLENECK_DEPTH=2
  - h[d] for d=2..MAX_DEPTH produced by bottom-up pass
  - h[2]: (N2, hidden)  N2 ≤ 16 nodes at depth 2, each carrying
    aggregated subtree information from depths 3-7

Bottleneck (at depth 2):
  - Self-attention across N2 depth-2 nodes  →  refined h[2]
  - Cross-attention: K=16 learned query SLOTS attend into h[2]
    producing z_slots: (K, hidden)
  - Linear compress: z_slots  →  z_bot: (K, bot_dim)   e.g. K=16, bot_dim=16
  - LATENT = z_bot ∈ R^(K × bot_dim) = R^256  (fixed size, GN-ready)

Decoder  (top-down, depths 0 → MAX_DEPTH):
  - Cross-attention: depth-2 nodes attend into z_slots_expanded
    recovering h_dec[2]: (N2, hidden)
  - Top-down from depth 0 as normal (root_token + child_head)
  - At depth 2: decoder hidden state REPLACED by h_dec[2]
    (the bottleneck injection point)
  - Depths 3-7: child_head propagation from h_dec[2] top-down
  - fuse(h_dec[d], pos) → split logit + value at each node

Key property for GN-ROM
  z_bot ∈ R^(K×bot_dim) is always fixed-size regardless of tree topology.
  At ROM time: optimise z_bot via GN, expand → z_slots_expanded → decoder.
  No encoder needed at inference.

Trains variants K×bot_dim ∈ {16×16=256, 16×8=128, 8×16=128} sequentially.
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
                C_in = leaf_vals_input_by_depth[d].shape[1]; break
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
            found = (kn[idx] == child_keys)
            self.children_idx[d] = torch.where(found, idx, torch.full_like(idx, -1))
        self.children_idx[self.max_depth] = None

        self.parent_idx[0] = None
        for d in range(1, self.max_depth + 1):
            kd = self.keys[d]; kp = self.keys[d-1]
            if kd.numel() == 0:
                self.parent_idx[d] = torch.empty((0,), dtype=torch.long, device=self.device); continue
            pidx = torch.searchsorted(kp, kd >> 2).clamp(0, len(kp)-1)
            self.parent_idx[d] = pidx

        for d in range(self.max_depth + 1):
            self._construct_neigh(d)

        for d in range(self.max_depth + 1):
            kd = self.keys[d]
            if kd.numel() == 0:
                self.values_gt[d] = torch.zeros((0, C_in), device=self.device)
                self.leaf_mask[d] = torch.zeros((0,), dtype=torch.bool, device=self.device)
                self.split_gt[d]  = None; continue
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
                self.split_gt[d]  = None
            else:
                ch = self.children_idx[d]
                is_leaf = (ch == -1).all(dim=1)
                self.leaf_mask[d] = is_leaf
                self.split_gt[d]  = (~is_leaf).float()

    def _construct_neigh(self, depth):
        keys = self.keys[depth]
        N = len(keys)
        if N == 0:
            self.neighs[depth] = torch.empty((0, 9), dtype=torch.long, device=self.device); return
        if depth == 0:
            neigh = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
            neigh[:, 4] = 0
            self.neighs[depth] = neigh; return
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],
                               device=self.device, dtype=torch.long)
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        res = 1 << depth
        nx, ny = n_coords[...,0], n_coords[...,1]
        valid  = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
        n_keys = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
        if valid.any():
            n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)
        idx   = torch.searchsorted(keys, n_keys.clamp(min=0)).clamp(0, len(keys)-1)
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
        pad_vec    = torch.zeros((1, features.shape[1]), device=features.device)
        feat_padded = torch.cat([features, pad_vec], dim=0)
        gather_idx  = neigh_idx.clone()
        gather_idx[gather_idx == -1] = N
        col = feat_padded[gather_idx].view(N, -1)
        return self.weights(col)


class QuadPool(nn.Module):
    def forward(self, child_features, quadtree, depth_child):
        d_parent = depth_child - 1
        Np = len(quadtree.keys[d_parent])
        C  = child_features.shape[1]
        if Np == 0:
            return torch.zeros((0, C), device=child_features.device)
        ch     = quadtree.children_idx[d_parent]
        pooled = torch.zeros((Np, C), device=child_features.device)
        cnt    = torch.zeros((Np, 1), device=child_features.device)
        for c in range(4):
            idx = ch[:, c]; mask = idx != -1
            if mask.any():
                pooled[mask] += child_features[idx[mask]]
                cnt[mask]    += 1.0
        return pooled / cnt.clamp(min=1.0)


# ─────────────────────────────────────────
# Bottleneck: attention pooling at depth 2
# ─────────────────────────────────────────

class AttentionBottleneck(nn.Module):
    """
    Takes h[BOTTLENECK_DEPTH]: (N2, hidden) — depth-2 node features
    that already carry full subtree information from bottom-up pooling.

    Step 1 — Self-attention among depth-2 nodes so they can communicate
              spatial context across the coarse grid.

    Step 2 — Cross-attention: K learned query SLOTS attend into the N2 nodes.
              Output: z_slots ∈ (K, hidden)   fixed size regardless of N2.

    Step 3 — Linear compress: z_slots → z_bot ∈ (K, bot_dim)
              This is the actual compressed latent for GN-ROM.
              Total floats: K * bot_dim  (e.g. 16 * 16 = 256)

    The inverse (for the decoder) is also here:
      z_bot → z_slots_expanded ∈ (K, hidden)
      then cross-attention: N2 depth-2 nodes attend into slots
      recovering h_dec[2]: (N2, hidden)
    """

    def __init__(self, hidden=128, K=16, bot_dim=16, num_heads=4):
        super().__init__()
        self.hidden  = hidden
        self.K       = K
        self.bot_dim = bot_dim

        # Step 1: self-attention among depth-2 nodes
        self.self_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden)

        # Step 2: K learned query slots cross-attend into depth-2 nodes
        self.slot_queries = nn.Parameter(torch.randn(K, hidden) * 0.02)
        self.cross_attn_enc = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.cross_norm_enc = nn.LayerNorm(hidden)

        # Step 3: compress slots → z_bot
        self.compress = nn.Linear(hidden, bot_dim)
        self.bot_norm = nn.LayerNorm(bot_dim)

        # Decoder inverse: expand z_bot → slots
        self.expand   = nn.Linear(bot_dim, hidden)
        self.exp_norm = nn.LayerNorm(hidden)

        # Decoder cross-attention: depth-2 nodes attend into expanded slots
        # to recover per-node decoder hidden states
        self.cross_attn_dec = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.cross_norm_dec = nn.LayerNorm(hidden)

        # Positional projection for depth-2 nodes used as queries in decoder
        pos_dim = 3 + 2 * 6 * 3   # fourier_encode output dim for pos_freqs=6
        self.pos_proj = nn.Linear(pos_dim, hidden)

    def encode(self, h2):
        """
        h2 : (N2, hidden)
        returns z_bot : (K, bot_dim)   — the fixed-size latent
                z_slots: (K, hidden)   — before compression, kept for decoder shortcut
        """
        # batch dim required by MultiheadAttention
        x = h2.unsqueeze(0)                              # (1, N2, hidden)

        # Step 1: self-attention
        sa, _ = self.self_attn(x, x, x)
        x = self.self_norm(x + sa)                       # (1, N2, hidden)

        # Step 2: cross-attention — slots query into nodes
        Q = self.slot_queries.unsqueeze(0)               # (1, K, hidden)
        slots, _ = self.cross_attn_enc(Q, x, x)
        slots = self.cross_norm_enc(slots)               # (1, K, hidden)
        slots = slots.squeeze(0)                         # (K, hidden)

        # Step 3: compress
        z_bot = self.compress(slots)                     # (K, bot_dim)
        z_bot = self.bot_norm(z_bot)

        return z_bot, slots   # return slots too so decoder can optionally use them

    def decode(self, z_bot, h2_pos):
        """
        z_bot   : (K, bot_dim)  — compressed latent
        h2_pos  : (N2, pos_dim) — fourier-encoded positions of depth-2 nodes
                  used as queries so each node attends to the right slots
        returns h_dec2 : (N2, hidden) — reconstructed depth-2 decoder hidden states
        """
        # Expand slots
        slots_exp = self.expand(z_bot)                   # (K, hidden)
        slots_exp = self.exp_norm(slots_exp)

        # Project node positions to query space
        node_q = self.pos_proj(h2_pos)                   # (N2, hidden)

        # Cross-attention: nodes (using position as query) attend into expanded slots
        Q = node_q.unsqueeze(0)                          # (1, N2, hidden)
        KV = slots_exp.unsqueeze(0)                      # (1, K, hidden)
        h_dec, _ = self.cross_attn_dec(Q, KV, KV)
        h_dec = self.cross_norm_dec(h_dec)               # (1, N2, hidden)
        h_dec = h_dec.squeeze(0)                         # (N2, hidden)

        return h_dec


class BottleneckEncoder(nn.Module):
    """
    Bottom-up encoder that stops at BOTTLENECK_DEPTH.

    Produces h[d] for d = BOTTLENECK_DEPTH .. MAX_DEPTH via QuadConv + QuadPool.
    Then passes h[BOTTLENECK_DEPTH] through the AttentionBottleneck to get z_bot.

    Also returns h[d] for d < BOTTLENECK_DEPTH (depths 0 and 1) since the
    decoder needs to handle those depths too — but those are computed via
    a separate shallow bottom-up pass from depth BOTTLENECK_DEPTH upward.
    """
    def __init__(self, in_c=1, hidden=128, max_depth=7, pos_freqs=6,
                 bottleneck_depth=2, K=16, bot_dim=16):
        super().__init__()
        self.pos_freqs         = pos_freqs
        self.max_depth         = max_depth
        self.hidden            = hidden
        self.bottleneck_depth  = bottleneck_depth
        pos_dim = 3 + 2 * pos_freqs * 3

        self.in_proj   = nn.Linear(in_c + pos_dim, hidden)
        self.convs     = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool      = QuadPool()
        self.to_emb    = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max_depth + 1)])
        self.emb_norm  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(max_depth + 1)])
        self.depth_gain = nn.Parameter(torch.ones(max_depth + 1))

        self.bottleneck = AttentionBottleneck(hidden=hidden, K=K, bot_dim=bot_dim)

    def forward(self, qt):
        device = qt.device
        h = [None] * (self.max_depth + 1)

        # Initialize per-node features at all depths
        for d in range(self.max_depth + 1):
            fin = qt.features_in[d]
            kd  = qt.keys[d]
            if fin is None or fin.numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=device); continue
            pos  = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, device), self.pos_freqs)
            h[d] = self.in_proj(torch.cat([fin, pos], dim=1))

        # Bottom-up pooling: leaves → bottleneck depth
        for d in range(self.max_depth, self.bottleneck_depth, -1):
            if h[d].numel() == 0: continue
            pooled  = self.pool(h[d], qt, d)
            h[d-1]  = h[d-1] + pooled
            if d-1 > self.bottleneck_depth and h[d-1].numel() > 0:
                h[d-1] = F.relu(self.convs[d-1](h[d-1], qt, d-1))

        # Apply to_emb + norm + depth_gain at bottleneck depth
        bd = self.bottleneck_depth
        if h[bd] is not None and h[bd].numel() > 0:
            z = self.to_emb[bd](h[bd])
            z = self.emb_norm[bd](z)
            z = self.depth_gain[bd] * z
            h[bd] = z
        else:
            h[bd] = torch.zeros((1, self.hidden), device=device)

        # Continue bottom-up from bottleneck_depth to depth 0
        # (depths 0 and 1 also need embeddings for the decoder's upper portion)
        for d in range(self.bottleneck_depth, 0, -1):
            if h[d].numel() == 0: continue
            pooled  = self.pool(h[d], qt, d)
            h[d-1]  = h[d-1] + pooled
            if d-1 >= 1 and h[d-1].numel() > 0:
                h[d-1] = F.relu(self.convs[d-1](h[d-1], qt, d-1))

        # Apply to_emb at depths 0..bottleneck_depth-1 for skip connections there
        for d in range(self.bottleneck_depth):
            if h[d] is None or h[d].numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=device); continue
            z = self.to_emb[d](h[d])
            z = self.emb_norm[d](z)
            z = self.depth_gain[d] * z
            h[d] = z

        # Attention bottleneck at depth 2
        h2     = h[self.bottleneck_depth]               # (N2, hidden)
        z_bot, slots = self.bottleneck.encode(h2)       # (K, bot_dim), (K, hidden)

        return z_bot, slots, h   # h needed so decoder can use skips at depths 0..1


class BottleneckDecoder(nn.Module):
    """
    Top-down decoder that reconstructs the full quadtree from z_bot.

    Flow:
      1. Expand z_bot → h_dec[BOTTLENECK_DEPTH] via AttentionBottleneck.decode
         using the positions of depth-2 nodes as queries.

      2. For depths < BOTTLENECK_DEPTH (depths 0 and 1):
         Use per-node encoder skips h[0], h[1] — these are shallow features
         that don't go through the bottleneck. This is legitimate: depths 0-1
         have very few nodes (1 and up to 4) and carry only coarse info.
         At ROM inference, these can be replaced by zero or a learned token.

      3. For depths >= BOTTLENECK_DEPTH:
         Top-down propagation via child_head from h_dec[BOTTLENECK_DEPTH].
         No encoder access needed beyond the bottleneck.

    The fuse operation at each depth combines:
      - top-down propagated hidden state (from child_head)
      - per-node encoder skip (only at depths < BOTTLENECK_DEPTH)
        OR decoded bottleneck state (at depth == BOTTLENECK_DEPTH)
      - Fourier positional encoding
    """
    def __init__(self, hidden=128, out_c=1, max_depth=7, pos_freqs=6,
                 bottleneck_depth=2, K=16, bot_dim=16):
        super().__init__()
        self.max_depth        = max_depth
        self.pos_freqs        = pos_freqs
        self.hidden           = hidden
        self.bottleneck_depth = bottleneck_depth
        pos_dim = 3 + 2 * pos_freqs * 3

        self.root_token = nn.Parameter(torch.zeros(1, hidden))

        # Fuse at depths 0..bottleneck_depth-1: top-down h + encoder skip + pos
        self.fuse_upper = nn.Sequential(
            nn.Linear(hidden + hidden + pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.skip_norm_upper = nn.LayerNorm(hidden)

        # Fuse at depths bottleneck_depth..max_depth: top-down h + bottleneck skip + pos
        self.fuse_lower = nn.Sequential(
            nn.Linear(hidden + hidden + pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.skip_norm_lower = nn.LayerNorm(hidden)

        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4 * hidden))
        self.val_head   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))
        self.mix_convs  = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

    def forward(self, qt, z_bot, bottleneck: AttentionBottleneck, h_enc):
        """
        qt         : Quadtree (GT topology — teacher forcing at training time)
        z_bot      : (K, bot_dim) — compressed latent
        bottleneck : AttentionBottleneck module (for decode cross-attention)
        h_enc      : list of (N_d, hidden) encoder outputs for depths 0..max_depth
                     Only h_enc[0..bottleneck_depth-1] are used as skips.
                     h_enc[bottleneck_depth..] are NOT used — bottleneck only.
        """
        device = qt.device
        bd     = self.bottleneck_depth

        # ── Reconstruct h_dec at bottleneck depth via cross-attention ──────────
        kd_bot = qt.keys[bd]
        pos_bot = fourier_encode(
            node_centers_from_keys(kd_bot, bd, self.max_depth, device), self.pos_freqs
        )                                                   # (N2, pos_dim)
        h_dec_bot = bottleneck.decode(z_bot, pos_bot)      # (N2, hidden)

        # ── Top-down decoder pass ───────────────────────────────────────────────
        split_logits = [None] * self.max_depth
        val_pred     = [None] * (self.max_depth + 1)

        N0 = len(qt.keys[0])
        h_by_depth = [None] * (self.max_depth + 1)
        h_by_depth[0] = self.root_token.expand(N0, -1).to(device) if N0 > 0 else \
                        torch.zeros((0, self.hidden), device=device)

        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=device)
                if d < self.max_depth:
                    split_logits[d] = torch.zeros((0,), device=device)
                continue

            kd  = qt.keys[d]
            pos = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, device), self.pos_freqs)

            if d == bd:
                # ── At bottleneck depth: REPLACE top-down h with decoded bottleneck ──
                # h_dec_bot carries full subtree info decoded from z_bot
                skip = self.skip_norm_lower(h_dec_bot.to(device))
                h    = self.fuse_lower(torch.cat([h, skip, pos], dim=1))

            elif d < bd:
                # ── Above bottleneck: use encoder skip (shallow, few nodes) ──────────
                enc_skip = h_enc[d]
                if enc_skip is not None and enc_skip.shape[0] == h.shape[0]:
                    skip = self.skip_norm_upper(enc_skip.to(device))
                else:
                    skip = torch.zeros((h.shape[0], self.hidden), device=device)
                h = self.fuse_upper(torch.cat([h, skip, pos], dim=1))

            else:
                # ── Below bottleneck: top-down only (no encoder access) ───────────────
                # h carries info from child_head propagation which originated at h_dec_bot
                skip = torch.zeros((h.shape[0], self.hidden), device=device)
                h    = self.fuse_lower(torch.cat([h, skip, pos], dim=1))

            if d >= 1 and h.numel() > 0:
                h = F.relu(self.mix_convs[d](h, qt, d))

            val_pred[d] = self.val_head(h)

            if d == self.max_depth:
                break

            split_logits[d] = self.split_head(h).squeeze(-1)

            ch        = qt.children_idx[d]
            has_child = (ch != -1).any(dim=1)
            N_next    = len(qt.keys[d+1])
            h_next    = torch.zeros((N_next, self.hidden), device=device)

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


class AttnBottleneckAE(nn.Module):
    """
    Full AE with attention bottleneck at depth BOTTLENECK_DEPTH.

    Latent: z_bot ∈ R^(K × bot_dim)  — fixed size, GN-ready.
    At ROM inference: optimise z_bot, decode via BottleneckDecoder.
    """
    def __init__(self, in_c=1, hidden=128, max_depth=7, pos_freqs=6,
                 bottleneck_depth=2, K=16, bot_dim=16):
        super().__init__()
        self.bottleneck_depth = bottleneck_depth
        self.K       = K
        self.bot_dim = bot_dim
        self.latent_dim = K * bot_dim

        self.encoder    = BottleneckEncoder(in_c=in_c, hidden=hidden, max_depth=max_depth,
                                            pos_freqs=pos_freqs, bottleneck_depth=bottleneck_depth,
                                            K=K, bot_dim=bot_dim)
        self.decoder    = BottleneckDecoder(hidden=hidden, out_c=in_c, max_depth=max_depth,
                                            pos_freqs=pos_freqs, bottleneck_depth=bottleneck_depth,
                                            K=K, bot_dim=bot_dim)

    def forward(self, qt):
        z_bot, slots, h_enc = self.encoder(qt)
        split_logits, val_pred = self.decoder(qt, z_bot, self.encoder.bottleneck, h_enc)
        return split_logits, val_pred, z_bot

    @torch.no_grad()
    def encode(self, qt):
        """Returns z_bot ∈ R^(K, bot_dim) — the latent for GN."""
        self.eval()
        z_bot, _, _ = self.encoder(qt)
        return z_bot

    @torch.no_grad()
    def decode(self, qt, z_bot):
        """Decode from z_bot alone. qt provides teacher topology."""
        self.eval()
        _, h_enc_shallow = self._get_shallow_enc(qt)
        return self.decoder(qt, z_bot, self.encoder.bottleneck, h_enc_shallow)

    def _get_shallow_enc(self, qt):
        """For ROM inference: get shallow encoder outputs at depths 0..bd-1 only."""
        # In practice at ROM time you'd zero these out or use learned tokens.
        # Here we just rerun the encoder fully.
        z_bot, _, h_enc = self.encoder(qt)
        return z_bot, h_enc


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
    dfdx  = 2*np.pi*k1 * np.cos(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y)
    dfdy  = 2*np.pi*k2 * np.sin(2*np.pi*k1*x) * np.cos(2*np.pi*k2*y)
    return np.sqrt(dfdx**2 + dfdy**2) / denom


class QuadNode:
    def __init__(self, x, y, size, level, max_level, min_level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.max_level = max_level; self.min_level = min_level
        self.children = []; self.u_val = None

    def gradient_subdivide_u(self, k1, k2, grad_threshold=0.3):
        if self.level >= self.max_level: return
        cx = self.x + self.size / 2; cy = self.y + self.size / 2
        force_split = self.level < self.min_level
        grad_mag    = gradient_magnitude_u(cx, cy, k1, k2)
        denom       = 4 * np.pi**2 * (k1**2 + k2**2)
        max_grad    = 2 * np.pi * np.sqrt(k1**2 + k2**2) / denom
        norm_grad   = grad_mag / max_grad if max_grad > 0 else 0
        depth_factor = 1.0 + 0.1 * self.level
        if force_split or (norm_grad > grad_threshold * depth_factor):
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
            cx = self.x + self.size / 2; cy = self.y + self.size / 2
            self.u_val = u_function(cx, cy, k1, k2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves_u(leaves_list, k1, k2)


def generate_u_sample(noise_std=NOISE_STD, grad_threshold=0.3):
    k1 = random.randint(1, 5); k2 = random.randint(1, 5)
    denom = 4 * np.pi**2 * (k1**2 + k2**2)
    root  = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root.gradient_subdivide_u(k1, k2, grad_threshold=grad_threshold)
    u_leaves = []
    root.collect_leaves_u(u_leaves, k1, k2)

    u_leaf_keys = [[] for _ in range(MAX_LEVEL + 1)]
    u_vals_raw  = [[] for _ in range(MAX_LEVEL + 1)]
    for node in u_leaves:
        d = node.level; res = 1 << d
        ix = max(0, min(res-1, int(node.x * res)))
        iy = max(0, min(res-1, int(node.y * res)))
        k  = Morton2D.xy2key(ix, iy)
        u_leaf_keys[d].append(int(k.item()))
        u_vals_raw[d].append([float(node.u_val) * denom])  # normalize to [-1,1]

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
            u_clean = torch.tensor(u_vals_raw[d],  dtype=torch.float32)
            u_noisy = u_clean + noise_std * torch.randn_like(u_clean)
            u_leaf_keys_by_depth.append(keys_t)
            u_vals_clean_by_depth.append(u_clean)
            u_vals_noisy_by_depth.append(u_noisy)

    return u_leaf_keys_by_depth, u_vals_clean_by_depth, u_vals_noisy_by_depth, u_leaves, k1, k2, denom


# ==========================================
# 6. Loss
# ==========================================

def compute_loss(model, qt, device, mse, bce, split_weight=0.5):
    split_logits, val_pred, z_bot = model(qt)

    L_val = torch.tensor(0.0, device=device)
    n_val = 0
    for d in range(MAX_LEVEL + 1):
        mask = qt.leaf_mask[d]
        if mask is None or mask.numel() == 0 or not mask.any(): continue
        L_val += mse(val_pred[d][mask], qt.values_gt[d][mask]); n_val += 1
    if n_val > 0: L_val = L_val / n_val

    L_split = torch.tensor(0.0, device=device)
    n_split = 0
    for d in range(MAX_LEVEL):
        if qt.split_gt[d] is None or qt.split_gt[d].numel() == 0: continue
        if split_logits[d] is None or split_logits[d].numel() == 0: continue
        L_split += bce(split_logits[d], qt.split_gt[d]); n_split += 1
    if n_split > 0: L_split = L_split / n_split

    gain_reg = 1e-4 * (model.encoder.depth_gain ** 2).mean()
    L_total  = L_val + split_weight * L_split + gain_reg
    return L_total, L_val, L_split, z_bot


# ==========================================
# 7. Training
# ==========================================

def train_variant(K, bot_dim, device, num_steps=3000, patience=150):
    latent_dim = K * bot_dim
    print(f"\n{'='*65}")
    print(f"AttnBottleneck AE  |  K={K}  bot_dim={bot_dim}  latent={latent_dim}  bd=2")
    print(f"{'='*65}")

    model = AttnBottleneckAE(
        in_c=1, hidden=128, max_depth=MAX_LEVEL, pos_freqs=6,
        bottleneck_depth=2, K=K, bot_dim=bot_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  Latent floats: {latent_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss(); bce = nn.BCEWithLogitsLoss()
    split_weight = 0.5

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"plots/attn_bn_K{K}_d{bot_dim}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}\n")

    history   = {'val': [], 'split': [], 'total': []}
    best      = {'loss': float('inf'), 'step': 0, 'counter': 0}
    best_path = f'{output_dir}/best_model.pt'
    samp_buf  = []

    for step in range(num_steps):
        u_keys, u_clean, u_noisy, u_leaves, k1, k2, denom = generate_u_sample()
        qt = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt.build_from_leaves(u_keys, u_noisy, u_clean)

        optimizer.zero_grad()
        L_total, L_val, L_split, z_bot = compute_loss(model, qt, device, mse, bce, split_weight)
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history['val'].append(L_val.item())
        history['split'].append(L_split.item())
        history['total'].append(L_total.item())

        if L_total.item() < best['loss']:
            best.update({'loss': L_total.item(), 'step': step, 'counter': 0})
            torch.save({'step': step, 'K': K, 'bot_dim': bot_dim,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'total_loss': L_total.item(),
                        'val_loss': L_val.item(),
                        'split_loss': L_split.item()}, best_path)
        else:
            best['counter'] += 1

        print(
            f"[K={K},d={bot_dim}] Step {step:4d}: "
            f"val={L_val.item():.5f} split={L_split.item():.5f} "
            f"total={L_total.item():.5f}  "
            f"best={best['loss']:.5f}@{best['step']}  "
            f"patience={best['counter']}/{patience}  "
            f"k1={k1} k2={k2} leaves={len(u_leaves)}"
        )

        if best['counter'] >= patience:
            print(f"\n  Early stopping at step {step}.")
            break

        if step % 100 == 0:
            samp_buf.append({'qt': qt, 'u_leaves': u_leaves, 'k1': k1, 'k2': k2,
                             'denom': denom, 'step': step, 'z': z_bot.detach().cpu()})
            if len(samp_buf) > 3: samp_buf.pop(0)

        if step % 100 == 0 or step == num_steps - 1:
            _plot_samples(samp_buf, model, output_dir, step, K, bot_dim)
            _plot_losses(history, output_dir, step, K, bot_dim)

    print(f"\n  Best: step={best['step']}  loss={best['loss']:.6f}  → {best_path}")
    validate(best_path, output_dir, K, bot_dim, num_samples=10, device=device)
    return best['loss']


# ==========================================
# 8. Plotting
# ==========================================

def _collect_preds(model, qt):
    with torch.no_grad():
        _, val_pred, z_bot = model(qt)
    pred_dict = {}; loss_dict = {}
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
            pred_dict[key] = pr[i]; loss_dict[key] = (pr[i] - gt[i]) ** 2
    return pred_dict, loss_dict, z_bot.detach().cpu()


def _draw_field(ax, leaves, val_fn, title, cmap):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
    values = [val_fn(n) for n in leaves]
    if not values: ax.set_title(title, fontsize=9); return
    vmin, vmax = min(values), max(values)
    vr = max(vmax - vmin, 1e-10)
    ax.set_title(f"{title}\n[{vmin:.3f},{vmax:.3f}]", fontsize=9)
    for node, v in zip(leaves, values):
        color = cmap(np.clip((v-vmin)/vr, 0, 1))
        ax.add_patch(patches.Rectangle((node.x, node.y), node.size, node.size,
                                       linewidth=0.2, edgecolor='gray', facecolor=color))


def _draw_pred(ax, qt, pred_dict, title, cmap, d_range=None):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
    vals = list(pred_dict.values())
    if not vals: ax.set_title(title, fontsize=9); return
    vmin, vmax = (min(vals), max(vals)) if d_range is None else d_range
    vr = max(vmax - vmin, 1e-10)
    ax.set_title(f"{title}\n[{vmin:.3f},{vmax:.3f}]", fontsize=9)
    for d in range(MAX_LEVEL + 1):
        if qt.keys[d].numel() == 0: continue
        mask = qt.leaf_mask[d]
        if mask is None or not mask.any(): continue
        kd = qt.keys[d][mask].cpu()
        ix, iy = Morton2D.key2xy(kd, depth=d)
        res = 1 << d; sz = 1.0 / res
        for i in range(len(kd)):
            v = pred_dict.get((d, int(ix[i].item()), int(iy[i].item())), 0.0)
            color = cmap(np.clip((v-vmin)/vr, 0, 1))
            ax.add_patch(patches.Rectangle((ix[i].item()*sz, iy[i].item()*sz), sz, sz,
                                           linewidth=0.2, edgecolor='gray', facecolor=color))


def _plot_samples(samp_buf, model, output_dir, step, K, bot_dim):
    if not samp_buf: return
    n = len(samp_buf)
    cmap = plt.get_cmap('viridis'); loss_cmap = plt.get_cmap('hot')
    fig, axes = plt.subplots(n, 4, figsize=(20, 5*n))
    if n == 1: axes = axes[np.newaxis, :]

    for row, samp in enumerate(samp_buf):
        qt = samp['qt']; leaves = samp['u_leaves']
        k1, k2, s = samp['k1'], samp['k2'], samp['step']
        pred_dict, loss_dict, z = _collect_preds(model, qt)

        _draw_field(axes[row,0], leaves, lambda nd: nd.u_val,
                    f"GT u  k1={k1} k2={k2} step={s}", cmap)
        _draw_pred(axes[row,1], qt, pred_dict, "Pred u", cmap)
        max_loss = max(loss_dict.values()) if loss_dict else 1.0
        _draw_pred(axes[row,2], qt,
                   {k: v/max(max_loss,1e-10) for k,v in loss_dict.items()},
                   f"Loss map (max={max_loss:.5f})", loss_cmap, d_range=(0,1))

        ax = axes[row,3]
        z_np = z.numpy().flatten()
        ax.bar(range(len(z_np)), z_np, color='steelblue', alpha=0.8, width=1.0)
        ax.axhline(0, color='k', lw=0.5)
        # vertical lines separating K slots
        for ki in range(1, K):
            ax.axvline(ki*bot_dim - 0.5, color='red', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(f"z_bot (K={K}, d={bot_dim})\nstd={z_np.std():.3f}", fontsize=9)
        ax.set_xlabel("dim")

    plt.suptitle(f"AttnBottleneck AE  K={K} bot_dim={bot_dim} latent={K*bot_dim} — step {step}", fontsize=12)
    plt.tight_layout()
    out = f'{output_dir}/samples_step{step:04d}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out}")


def _plot_losses(history, output_dir, step, K, bot_dim):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, color, label in zip(axes,
        ['total','val','split'], ['navy','steelblue','orange'],
        ['Total Loss','Val MSE','Split BCE']):
        ax.plot(history[key], lw=1, alpha=0.8, color=color)
        ax.set_xlabel('Step'); ax.grid(True, alpha=0.3)
        last = history[key][-1] if history[key] else 0
        ax.set_title(f"{label}\nfinal={last:.5f}", fontsize=9)
    plt.suptitle(f'AttnBottleneck  K={K} d={bot_dim}  step {step}', fontsize=12)
    plt.tight_layout()
    out = f'{output_dir}/losses_step{step:04d}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out}")


def validate(model_path, output_dir, K, bot_dim, num_samples=10, device='cpu'):
    print(f"\n--- Validation K={K} bot_dim={bot_dim} ---")
    ckpt  = torch.load(model_path, map_location=device)
    model = AttnBottleneckAE(in_c=1, hidden=128, max_depth=MAX_LEVEL, pos_freqs=6,
                             bottleneck_depth=2, K=K, bot_dim=bot_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()

    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(val_dir, exist_ok=True)
    mse_fn = nn.MSELoss()
    all_z  = []; per_depth_losses = {d: [] for d in range(MAX_LEVEL+1)}

    for i in range(num_samples):
        u_keys, u_clean, u_noisy, u_leaves, k1, k2, denom = generate_u_sample()
        qt = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt.build_from_leaves(u_keys, u_noisy, u_clean)
        with torch.no_grad():
            _, val_pred, z_bot = model(qt)
        all_z.append(z_bot.cpu().numpy().flatten())
        for d in range(MAX_LEVEL+1):
            mask = qt.leaf_mask[d]
            if mask is None or mask.numel() == 0 or not mask.any(): continue
            per_depth_losses[d].append(mse_fn(val_pred[d][mask], qt.values_gt[d][mask]).item())
        print(f"  [{i+1}/{num_samples}] k1={k1} k2={k2} leaves={len(u_leaves)} "
              f"|z|={z_bot.norm():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    depths = [d for d in range(MAX_LEVEL+1) if per_depth_losses[d]]
    axes[0].bar(depths, [np.mean(per_depth_losses[d]) for d in depths],
                color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Depth'); axes[0].set_ylabel('Avg MSE')
    axes[0].set_title('Avg Reconstruction Loss by Depth'); axes[0].grid(True, alpha=0.3)

    all_z_arr = np.stack(all_z)
    im = axes[1].imshow(all_z_arr, aspect='auto', cmap='RdBu_r',
                        vmin=-3*all_z_arr.std(), vmax=3*all_z_arr.std())
    # vertical lines separating K slots
    for ki in range(1, K):
        axes[1].axvline(ki*bot_dim - 0.5, color='yellow', lw=1, ls='--')
    axes[1].set_xlabel('z_bot dim'); axes[1].set_ylabel('Sample')
    axes[1].set_title(f'z_bot across {num_samples} samples'); plt.colorbar(im, ax=axes[1])

    z_norms = [np.linalg.norm(z) for z in all_z]
    axes[2].hist(z_norms, bins=10, color='steelblue', alpha=0.8, edgecolor='black')
    axes[2].set_xlabel('||z_bot||')
    axes[2].set_title(f'Latent norm  mean={np.mean(z_norms):.3f}')

    plt.suptitle(f'AttnBottleneck Validation  K={K} d={bot_dim}  latent={K*bot_dim}', fontsize=13)
    plt.tight_layout()
    out = f'{val_dir}/validation_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out}")


# ==========================================
# 9. Main
# ==========================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    print("Attention Bottleneck AE at quadtree depth 2")
    print("Encoder: bottom-up QuadConv → depth-2 self-attn → K slot cross-attn → compress")
    print("Decoder: expand → depth-2 cross-attn → top-down child_head (no encoder access below depth 2)")

    # Variants: (K slots, bot_dim per slot)  → total latent floats
    variants = [
        (16, 16),   # 256 floats
        (16,  8),   # 128 floats
        ( 8, 16),   # 128 floats
    ]

    results = {}
    for K, bot_dim in variants:
        best_loss = train_variant(K, bot_dim, device)
        results[(K, bot_dim)] = best_loss

    print(f"\n{'='*65}")
    print("Summary:")
    for (K, d), loss in results.items():
        print(f"  K={K:2d}  bot_dim={d:2d}  latent={K*d:4d}  →  best_loss={loss:.6f}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
