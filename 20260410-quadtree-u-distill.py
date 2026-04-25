"""
20260410-quadtree-u-distill.py

Knowledge distillation approach for compressed quadtree AE.

Problem with previous attempts:
  Removing per-node skip connections kills the learning signal — the decoder
  gets stuck predicting the mean (val_loss ~0.24 forever).

Solution — two-path distillation:
  Path A (Teacher): Full Feb18 architecture with per-node skips E[d].
                    Trains normally, reaches val_loss ~0.0005.
  Path B (Student): Attention bottleneck at depth 2 → z_bot (fixed size).
                    Student decoder must match teacher OUTPUT (not GT directly).
                    This forces z_bot to encode what the teacher learned.

Training loss:
  L_total = L_teacher + λ * L_student

  L_teacher = MSE(pred_A[leaf], gt[leaf]) + 0.5*BCE(split_A, split_gt)  [Feb18 loss]
  L_student  = MSE(pred_B[leaf], pred_A[leaf].detach())                  [distill loss]
               + 0.5 * BCE(split_B[d], split_A[d].detach().sigmoid().round())

  pred_A is detached so student training doesn't interfere with teacher.

At ROM inference:
  Only Path B is used: z_bot → student decoder → adaptive quadtree.
  z_bot ∈ R^(K × bot_dim) is the GN latent.

Architecture:
  Shared encoder (Feb18 style) → E[d] per-node embeddings
  Teacher decoder: fuse(top-down h, E[d] skip, pos) → val, split  [unchanged]
  Bottleneck: self-attn(E[2]) → K-slot cross-attn → compress → z_bot
  Student decoder: z_bot → expand → cross-attn → h_dec[2]
                   then top-down child_head (no encoder access)

Trains three variants: (K=16,d=16)=256 floats, (K=16,d=8)=128 floats, (K=8,d=16)=128 floats
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
# 3. Quadtree Data Structure  (unchanged from Feb18)
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
        keys = self.keys[depth]; N = len(keys)
        if N == 0:
            self.neighs[depth] = torch.empty((0, 9), dtype=torch.long, device=self.device); return
        if depth == 0:
            neigh = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
            neigh[:, 4] = 0; self.neighs[depth] = neigh; return
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],
                               device=self.device, dtype=torch.long)
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        res = 1 << depth; nx, ny = n_coords[...,0], n_coords[...,1]
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
        neigh_idx = quadtree.neighs[depth]; N = neigh_idx.shape[0]
        if N == 0:
            return torch.zeros((0, self.weights.out_features), device=features.device)
        pad_vec    = torch.zeros((1, features.shape[1]), device=features.device)
        feat_padded = torch.cat([features, pad_vec], dim=0)
        gather_idx  = neigh_idx.clone(); gather_idx[gather_idx == -1] = N
        return self.weights(feat_padded[gather_idx].view(N, -1))


class QuadPool(nn.Module):
    def forward(self, child_features, quadtree, depth_child):
        d_parent = depth_child - 1
        Np = len(quadtree.keys[d_parent]); C = child_features.shape[1]
        if Np == 0:
            return torch.zeros((0, C), device=child_features.device)
        ch = quadtree.children_idx[d_parent]
        pooled = torch.zeros((Np, C), device=child_features.device)
        cnt    = torch.zeros((Np, 1), device=child_features.device)
        for c in range(4):
            idx = ch[:, c]; mask = idx != -1
            if mask.any():
                pooled[mask] += child_features[idx[mask]]; cnt[mask] += 1.0
        return pooled / cnt.clamp(min=1.0)


# ── Shared Encoder (identical to Feb18) ──────────────────────────────────────

class SharedEncoder(nn.Module):
    """Feb18 encoder — produces per-node E[d] for all depths."""
    def __init__(self, in_c=1, hidden=128, max_depth=7, pos_freqs=6):
        super().__init__()
        self.pos_freqs = pos_freqs; self.max_depth = max_depth; self.hidden = hidden
        pos_dim = 3 + 2 * pos_freqs * 3
        self.in_proj    = nn.Linear(in_c + pos_dim, hidden)
        self.convs      = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool       = QuadPool()
        self.to_emb     = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max_depth + 1)])
        self.emb_norm   = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(max_depth + 1)])
        self.depth_gain = nn.Parameter(torch.ones(max_depth + 1))

    def forward(self, qt):
        device = qt.device
        h = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            fin = qt.features_in[d]; kd = qt.keys[d]
            if fin is None or fin.numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=device); continue
            pos  = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, device), self.pos_freqs)
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
                E[d] = torch.zeros((0, self.hidden), device=device)
            else:
                z = self.to_emb[d](h[d])
                z = self.emb_norm[d](z)
                E[d] = self.depth_gain[d] * z
        return E


# ── Teacher Decoder (identical to Feb18 TreeDecoder) ─────────────────────────

class TeacherDecoder(nn.Module):
    """Full per-node skip decoder — same as Feb18. Strong learning signal."""
    def __init__(self, hidden=128, out_c=1, max_depth=7, pos_freqs=6):
        super().__init__()
        self.max_depth = max_depth; self.pos_freqs = pos_freqs; self.hidden = hidden
        pos_dim = 3 + 2 * pos_freqs * 3
        self.root_token = nn.Parameter(torch.zeros(1, hidden))
        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden + pos_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.skip_norm  = nn.LayerNorm(hidden)
        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4*hidden))
        self.val_head   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))
        self.mix_convs  = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

    def forward(self, qt, E):
        device = qt.device
        N0 = len(qt.keys[0])
        h_by_depth = [None] * (self.max_depth + 1)
        h_by_depth[0] = self.root_token.expand(N0, -1).to(device) if N0 > 0 else \
                        torch.zeros((0, self.hidden), device=device)
        split_logits = [None] * self.max_depth
        val_pred     = [None] * (self.max_depth + 1)

        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=device)
                if d < self.max_depth: split_logits[d] = torch.zeros((0,), device=device)
                continue
            kd  = qt.keys[d]
            pos = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, device), self.pos_freqs)
            if E[d] is not None and E[d].shape[0] == h.shape[0]:
                skip = self.skip_norm(E[d])
            else:
                skip = torch.zeros((h.shape[0], self.hidden), device=device)
            h = self.fuse(torch.cat([h, skip, pos], dim=1))
            if d >= 1 and h.numel() > 0:
                h = F.relu(self.mix_convs[d](h, qt, d))
            val_pred[d] = self.val_head(h)
            if d == self.max_depth: break
            split_logits[d] = self.split_head(h).squeeze(-1)
            ch = qt.children_idx[d]; has_child = (ch != -1).any(dim=1)
            N_next = len(qt.keys[d+1])
            h_next = torch.zeros((N_next, self.hidden), device=device)
            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, self.hidden)
                parent_rows = torch.nonzero(has_child).squeeze(-1)
                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1: h_next[ci] = child_feats[t, c]
            h_by_depth[d+1] = h_next
        return split_logits, val_pred


# ── Attention Bottleneck ──────────────────────────────────────────────────────

class AttentionBottleneck(nn.Module):
    """
    Compresses E[bd] (per-node embeddings at bottleneck depth) into
    z_bot ∈ R^(K, bot_dim) via:
      1. Self-attention among bd nodes
      2. K learned slot queries cross-attend into bd nodes  → z_slots (K, hidden)
      3. Linear compress z_slots → z_bot (K, bot_dim)

    Decode:
      4. Expand z_bot → z_slots_exp (K, hidden)
      5. Node positions cross-attend into slots → h_dec_bd (N_bd, hidden)
    """
    def __init__(self, hidden=128, K=16, bot_dim=16, num_heads=4):
        super().__init__()
        self.hidden = hidden; self.K = K; self.bot_dim = bot_dim

        self.self_attn      = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.self_norm      = nn.LayerNorm(hidden)
        self.slot_queries   = nn.Parameter(torch.randn(K, hidden) * 0.02)
        self.cross_attn_enc = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.cross_norm_enc = nn.LayerNorm(hidden)
        self.compress       = nn.Linear(hidden, bot_dim)
        self.bot_norm       = nn.LayerNorm(bot_dim)

        self.expand         = nn.Linear(bot_dim, hidden)
        self.exp_norm       = nn.LayerNorm(hidden)
        pos_dim = 3 + 2 * 6 * 3
        self.pos_proj       = nn.Linear(pos_dim, hidden)
        self.cross_attn_dec = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.cross_norm_dec = nn.LayerNorm(hidden)

    def encode(self, h_bd):
        """h_bd: (N_bd, hidden) → z_bot: (K, bot_dim)"""
        x = h_bd.unsqueeze(0)                                   # (1, N_bd, hidden)
        sa, _ = self.self_attn(x, x, x)
        x = self.self_norm(x + sa)
        Q = self.slot_queries.unsqueeze(0)                      # (1, K, hidden)
        slots, _ = self.cross_attn_enc(Q, x, x)
        slots = self.cross_norm_enc(slots).squeeze(0)           # (K, hidden)
        z_bot = self.bot_norm(self.compress(slots))             # (K, bot_dim)
        return z_bot

    def decode(self, z_bot, pos_bd):
        """z_bot: (K, bot_dim), pos_bd: (N_bd, pos_dim) → h_dec: (N_bd, hidden)"""
        slots_exp = self.exp_norm(self.expand(z_bot))           # (K, hidden)
        node_q    = self.pos_proj(pos_bd)                       # (N_bd, hidden)
        Q  = node_q.unsqueeze(0); KV = slots_exp.unsqueeze(0)
        h, _ = self.cross_attn_dec(Q, KV, KV)
        return self.cross_norm_dec(h).squeeze(0)                # (N_bd, hidden)


# ── Student Decoder ───────────────────────────────────────────────────────────

class StudentDecoder(nn.Module):
    """
    Decodes from z_bot only — no encoder access.
    At bd: inject h_dec_bd from bottleneck decode.
    Above bd (depths 0..bd-1): root token + child_head propagation downward
      — these get fused with a zero skip (no encoder).
    Below bd (depths bd+1..max): child_head from h_dec_bd, zero skip.
    """
    def __init__(self, hidden=128, out_c=1, max_depth=7, pos_freqs=6, bottleneck_depth=2):
        super().__init__()
        self.max_depth        = max_depth
        self.pos_freqs        = pos_freqs
        self.hidden           = hidden
        self.bottleneck_depth = bottleneck_depth
        pos_dim = 3 + 2 * pos_freqs * 3

        self.root_token = nn.Parameter(torch.zeros(1, hidden))
        # single fuse used at all depths — skip is zeros for student
        self.fuse = nn.Sequential(
            nn.Linear(hidden + pos_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4*hidden))
        self.val_head   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))
        self.mix_convs  = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

    def forward(self, qt, h_dec_bd):
        """
        qt        : Quadtree (teacher topology)
        h_dec_bd  : (N_bd, hidden) — bottleneck-decoded hidden at bd
        """
        device = qt.device; bd = self.bottleneck_depth
        N0 = len(qt.keys[0])
        h_by_depth = [None] * (self.max_depth + 1)
        h_by_depth[0] = self.root_token.expand(N0, -1).to(device) if N0 > 0 else \
                        torch.zeros((0, self.hidden), device=device)

        split_logits = [None] * self.max_depth
        val_pred     = [None] * (self.max_depth + 1)

        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=device)
                if d < self.max_depth: split_logits[d] = torch.zeros((0,), device=device)
                continue

            if d == bd:
                # Inject bottleneck-decoded hidden — override top-down propagation
                h = h_dec_bd

            kd  = qt.keys[d]
            pos = fourier_encode(node_centers_from_keys(kd, d, self.max_depth, device), self.pos_freqs)
            h = self.fuse(torch.cat([h, pos], dim=1))
            if d >= 1 and h.numel() > 0:
                h = F.relu(self.mix_convs[d](h, qt, d))

            val_pred[d] = self.val_head(h)
            if d == self.max_depth: break
            split_logits[d] = self.split_head(h).squeeze(-1)

            ch = qt.children_idx[d]; has_child = (ch != -1).any(dim=1)
            N_next = len(qt.keys[d+1])
            h_next = torch.zeros((N_next, self.hidden), device=device)
            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, self.hidden)
                parent_rows = torch.nonzero(has_child).squeeze(-1)
                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1: h_next[ci] = child_feats[t, c]
            h_by_depth[d+1] = h_next

        return split_logits, val_pred


# ── Full Distillation Model ───────────────────────────────────────────────────

class DistillQuadAE(nn.Module):
    """
    Two-path distillation model.

    forward() returns outputs from both teacher and student for loss computation.
    At ROM inference, only encode() + student_decode() are needed.
    """
    def __init__(self, in_c=1, hidden=128, max_depth=7, pos_freqs=6,
                 bottleneck_depth=2, K=16, bot_dim=16):
        super().__init__()
        self.bottleneck_depth = bottleneck_depth
        self.K = K; self.bot_dim = bot_dim
        self.latent_dim = K * bot_dim

        self.encoder         = SharedEncoder(in_c=in_c, hidden=hidden,
                                             max_depth=max_depth, pos_freqs=pos_freqs)
        self.teacher_decoder = TeacherDecoder(hidden=hidden, out_c=in_c,
                                              max_depth=max_depth, pos_freqs=pos_freqs)
        self.bottleneck      = AttentionBottleneck(hidden=hidden, K=K, bot_dim=bot_dim)
        self.student_decoder = StudentDecoder(hidden=hidden, out_c=in_c,
                                              max_depth=max_depth, pos_freqs=pos_freqs,
                                              bottleneck_depth=bottleneck_depth)

    def forward(self, qt):
        device = qt.device; bd = self.bottleneck_depth

        # Shared encoder
        E = self.encoder(qt)

        # Teacher path (strong signal, per-node skips)
        split_T, val_T = self.teacher_decoder(qt, E)

        # Bottleneck: compress E[bd] → z_bot
        h_bd  = E[bd]                                          # (N_bd, hidden)
        z_bot = self.bottleneck.encode(h_bd)                   # (K, bot_dim)

        # Student: decode from z_bot
        kd_bd   = qt.keys[bd]
        pos_bd  = fourier_encode(
            node_centers_from_keys(kd_bd, bd, self.max_depth, device), self.pos_freqs)
        h_dec_bd = self.bottleneck.decode(z_bot, pos_bd)       # (N_bd, hidden)
        split_S, val_S = self.student_decoder(qt, h_dec_bd)

        return (split_T, val_T), (split_S, val_S), z_bot

    @property
    def max_depth(self):
        return self.encoder.max_depth

    @torch.no_grad()
    def encode(self, qt):
        """Returns z_bot ∈ R^(K, bot_dim) — hand to GN solver."""
        self.eval()
        E     = self.encoder(qt)
        z_bot = self.bottleneck.encode(E[self.bottleneck_depth])
        return z_bot

    @torch.no_grad()
    def student_decode(self, qt, z_bot):
        """Decode from z_bot alone. qt provides teacher topology."""
        self.eval()
        device  = qt.device; bd = self.bottleneck_depth
        kd_bd   = qt.keys[bd]
        pos_bd  = fourier_encode(
            node_centers_from_keys(kd_bd, bd, self.max_depth, device), self.pos_freqs)
        h_dec_bd = self.bottleneck.decode(z_bot, pos_bd)
        return self.student_decoder(qt, h_dec_bd)


# ==========================================
# 5. Poisson Data Generation (u only)
# ==========================================

MAX_LEVEL = 7
MIN_LEVEL = 6
NOISE_STD = 0.05


def u_function(x, y, k1, k2):
    denom = 4 * np.pi**2 * (k1**2 + k2**2)
    return -np.sin(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y) / denom


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
        cx = self.x + self.size/2; cy = self.y + self.size/2
        force_split = self.level < self.min_level
        grad_mag    = gradient_magnitude_u(cx, cy, k1, k2)
        denom       = 4*np.pi**2*(k1**2+k2**2)
        max_grad    = 2*np.pi*np.sqrt(k1**2+k2**2)/denom
        norm_grad   = grad_mag/max_grad if max_grad > 0 else 0
        if force_split or (norm_grad > grad_threshold*(1.0+0.1*self.level)):
            half = self.size/2
            self.children = [
                QuadNode(self.x,      self.y,      half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half, self.y,      half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x,      self.y+half, half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half, self.y+half, half, self.level+1, self.max_level, self.min_level),
            ]
            for child in self.children:
                child.gradient_subdivide_u(k1, k2, grad_threshold)

    def collect_leaves_u(self, leaves_list, k1, k2):
        if not self.children:
            cx = self.x+self.size/2; cy = self.y+self.size/2
            self.u_val = u_function(cx, cy, k1, k2); leaves_list.append(self)
        else:
            for child in self.children: child.collect_leaves_u(leaves_list, k1, k2)


def generate_u_sample(noise_std=NOISE_STD, grad_threshold=0.3):
    k1 = random.randint(1,5); k2 = random.randint(1,5)
    denom = 4*np.pi**2*(k1**2+k2**2)
    root = QuadNode(0,0,1.0,0,MAX_LEVEL,MIN_LEVEL)
    root.gradient_subdivide_u(k1,k2,grad_threshold=grad_threshold)
    u_leaves = []; root.collect_leaves_u(u_leaves,k1,k2)

    u_leaf_keys=[[] for _ in range(MAX_LEVEL+1)]
    u_vals_raw =[[] for _ in range(MAX_LEVEL+1)]
    for node in u_leaves:
        d=node.level; res=1<<d
        ix=max(0,min(res-1,int(node.x*res))); iy=max(0,min(res-1,int(node.y*res)))
        k=Morton2D.xy2key(ix,iy)
        u_leaf_keys[d].append(int(k.item()))
        u_vals_raw[d].append([float(node.u_val)*denom])

    u_leaf_keys_by_depth=[]; u_vals_clean_by_depth=[]; u_vals_noisy_by_depth=[]
    for d in range(MAX_LEVEL+1):
        if len(u_leaf_keys[d])==0:
            u_leaf_keys_by_depth.append(torch.empty((0,),dtype=torch.long))
            u_vals_clean_by_depth.append(torch.empty((0,1),dtype=torch.float32))
            u_vals_noisy_by_depth.append(torch.empty((0,1),dtype=torch.float32))
        else:
            keys_t  = torch.tensor(u_leaf_keys[d],dtype=torch.long)
            u_clean = torch.tensor(u_vals_raw[d],dtype=torch.float32)
            u_noisy = u_clean + noise_std*torch.randn_like(u_clean)
            u_leaf_keys_by_depth.append(keys_t)
            u_vals_clean_by_depth.append(u_clean)
            u_vals_noisy_by_depth.append(u_noisy)

    return u_leaf_keys_by_depth, u_vals_clean_by_depth, u_vals_noisy_by_depth, u_leaves, k1, k2, denom


# ==========================================
# 6. Loss
# ==========================================

def compute_loss(model, qt, device, mse, bce,
                 split_weight=0.5, distill_weight=1.0):
    (split_T, val_T), (split_S, val_S), z_bot = model(qt)

    # ── Teacher loss (identical to Feb18) ────────────────────────────────────
    L_val_T = torch.tensor(0.0, device=device); n_val = 0
    for d in range(MAX_LEVEL+1):
        mask = qt.leaf_mask[d]
        if mask is None or mask.numel()==0 or not mask.any(): continue
        L_val_T += mse(val_T[d][mask], qt.values_gt[d][mask]); n_val += 1
    if n_val > 0: L_val_T = L_val_T / n_val

    L_split_T = torch.tensor(0.0, device=device); n_split = 0
    for d in range(MAX_LEVEL):
        if qt.split_gt[d] is None or qt.split_gt[d].numel()==0: continue
        if split_T[d] is None or split_T[d].numel()==0: continue
        L_split_T += bce(split_T[d], qt.split_gt[d]); n_split += 1
    if n_split > 0: L_split_T = L_split_T / n_split

    gain_reg = 1e-4 * (model.encoder.depth_gain**2).mean()
    L_teacher = L_val_T + split_weight*L_split_T + gain_reg

    # ── Student distillation loss (match teacher output, not GT) ────────────
    # val: student matches teacher predictions at leaf nodes
    L_val_S = torch.tensor(0.0, device=device); n_val_s = 0
    for d in range(MAX_LEVEL+1):
        mask = qt.leaf_mask[d]
        if mask is None or mask.numel()==0 or not mask.any(): continue
        if val_T[d] is None or val_S[d] is None: continue
        if val_T[d].shape[0] != val_S[d].shape[0]: continue
        # detach teacher so student gradient doesn't affect teacher
        L_val_S += mse(val_S[d][mask], val_T[d][mask].detach()); n_val_s += 1
    if n_val_s > 0: L_val_S = L_val_S / n_val_s

    # split: student matches teacher split probabilities
    L_split_S = torch.tensor(0.0, device=device); n_split_s = 0
    for d in range(MAX_LEVEL):
        if split_T[d] is None or split_S[d] is None: continue
        if split_T[d].numel()==0 or split_S[d].numel()==0: continue
        if split_T[d].shape != split_S[d].shape: continue
        # soft targets from teacher sigmoid
        soft_targets = torch.sigmoid(split_T[d].detach())
        L_split_S += F.binary_cross_entropy(torch.sigmoid(split_S[d]), soft_targets); n_split_s += 1
    if n_split_s > 0: L_split_S = L_split_S / n_split_s

    L_student = L_val_S + split_weight*L_split_S

    L_total = L_teacher + distill_weight*L_student
    return L_total, L_teacher, L_student, L_val_T, L_val_S, z_bot


# ==========================================
# 7. Training
# ==========================================

def train_variant(K, bot_dim, device, num_steps=3000, patience=150, distill_weight=1.0):
    latent_dim = K * bot_dim
    print(f"\n{'='*65}")
    print(f"Distill QuadAE  K={K}  bot_dim={bot_dim}  latent={latent_dim}  λ={distill_weight}")
    print(f"{'='*65}")

    model = DistillQuadAE(in_c=1, hidden=128, max_depth=MAX_LEVEL, pos_freqs=6,
                          bottleneck_depth=2, K=K, bot_dim=bot_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  Latent floats: {latent_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss(); bce = nn.BCEWithLogitsLoss()
    split_weight = 0.5

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"plots/distill_K{K}_d{bot_dim}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}\n")

    history   = {'total':[], 'teacher':[], 'student':[], 'val_T':[], 'val_S':[]}
    best_T    = {'loss': float('inf'), 'step': 0, 'counter': 0}
    best_S    = {'loss': float('inf'), 'step': 0}
    best_path = f'{output_dir}/best_model.pt'
    samp_buf  = []

    for step in range(num_steps):
        u_keys, u_clean, u_noisy, u_leaves, k1, k2, denom = generate_u_sample()
        qt = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt.build_from_leaves(u_keys, u_noisy, u_clean)

        optimizer.zero_grad()
        L_total, L_T, L_S, L_val_T, L_val_S, z_bot = compute_loss(
            model, qt, device, mse, bce, split_weight, distill_weight)
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history['total'].append(L_total.item())
        history['teacher'].append(L_T.item())
        history['student'].append(L_S.item())
        history['val_T'].append(L_val_T.item())
        history['val_S'].append(L_val_S.item())

        if L_T.item() < best_T['loss']:
            best_T.update({'loss': L_T.item(), 'step': step, 'counter': 0})
            torch.save({
                'step': step, 'K': K, 'bot_dim': bot_dim,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'teacher_loss': L_T.item(),
                'student_loss': L_S.item(),
                'val_loss_teacher': L_val_T.item(),
                'val_loss_student': L_val_S.item(),
            }, best_path)
        else:
            best_T['counter'] += 1

        if L_val_S.item() < best_S['loss']:
            best_S.update({'loss': L_val_S.item(), 'step': step})

        print(
            f"[K={K},d={bot_dim}] Step {step:4d}: "
            f"T_val={L_val_T.item():.5f}  S_val={L_val_S.item():.5f}  "
            f"T_best={best_T['loss']:.5f}@{best_T['step']}  "
            f"S_best={best_S['loss']:.5f}@{best_S['step']}  "
            f"patience={best_T['counter']}/{patience}  "
            f"k1={k1} k2={k2}"
        )

        if best_T['counter'] >= patience:
            print(f"\n  Early stopping at step {step}.")
            break

        if step % 100 == 0:
            samp_buf.append({'qt': qt, 'u_leaves': u_leaves, 'k1': k1, 'k2': k2,
                             'denom': denom, 'step': step, 'z': z_bot.detach().cpu()})
            if len(samp_buf) > 3: samp_buf.pop(0)

        if step % 100 == 0 or step == num_steps-1:
            _plot_samples(samp_buf, model, output_dir, step, K, bot_dim)
            _plot_losses(history, output_dir, step, K, bot_dim)

    print(f"\n  Teacher best: step={best_T['step']} loss={best_T['loss']:.6f}")
    print(f"  Student best val: step={best_S['step']} loss={best_S['loss']:.6f}")
    validate(best_path, output_dir, K, bot_dim, num_samples=10, device=device)
    return best_T['loss'], best_S['loss']


# ==========================================
# 8. Plotting
# ==========================================

def _collect_both(model, qt):
    with torch.no_grad():
        (_, val_T), (_, val_S), z_bot = model(qt)
    pred_T={}; pred_S={}; loss_T={}; loss_S={}
    for d in range(MAX_LEVEL+1):
        if qt.keys[d].numel()==0: continue
        mask = qt.leaf_mask[d]
        if mask is None or mask.numel()==0 or not mask.any(): continue
        kd = qt.keys[d][mask].cpu()
        ix, iy = Morton2D.key2xy(kd, depth=d)
        gt = qt.values_gt[d][mask].cpu().numpy().flatten()
        pT = val_T[d][mask].cpu().numpy().flatten()
        pS = val_S[d][mask].cpu().numpy().flatten() if val_S[d].shape[0]==qt.keys[d].shape[0] \
             else np.zeros_like(pT)
        for i in range(len(kd)):
            key=(d,int(ix[i].item()),int(iy[i].item()))
            pred_T[key]=pT[i]; loss_T[key]=(pT[i]-gt[i])**2
            pred_S[key]=pS[i]; loss_S[key]=(pS[i]-gt[i])**2
    return pred_T, pred_S, loss_T, loss_S, z_bot.detach().cpu()


def _draw_field(ax, leaves, val_fn, title, cmap):
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_aspect('equal');ax.axis('off')
    values=[val_fn(n) for n in leaves]
    if not values: ax.set_title(title,fontsize=9); return
    vmin,vmax=min(values),max(values); vr=max(vmax-vmin,1e-10)
    ax.set_title(f"{title}\n[{vmin:.3f},{vmax:.3f}]",fontsize=9)
    for node,v in zip(leaves,values):
        ax.add_patch(patches.Rectangle((node.x,node.y),node.size,node.size,
            linewidth=0.2,edgecolor='gray',facecolor=cmap(np.clip((v-vmin)/vr,0,1))))


def _draw_pred(ax, qt, pred_dict, title, cmap, d_range=None):
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_aspect('equal');ax.axis('off')
    vals=list(pred_dict.values())
    if not vals: ax.set_title(title,fontsize=9); return
    vmin,vmax=(min(vals),max(vals)) if d_range is None else d_range
    vr=max(vmax-vmin,1e-10)
    ax.set_title(f"{title}\n[{vmin:.3f},{vmax:.3f}]",fontsize=9)
    for d in range(MAX_LEVEL+1):
        if qt.keys[d].numel()==0: continue
        mask=qt.leaf_mask[d]
        if mask is None or not mask.any(): continue
        kd=qt.keys[d][mask].cpu(); ix,iy=Morton2D.key2xy(kd,depth=d)
        res=1<<d; sz=1.0/res
        for i in range(len(kd)):
            v=pred_dict.get((d,int(ix[i].item()),int(iy[i].item())),0.0)
            ax.add_patch(patches.Rectangle((ix[i].item()*sz,iy[i].item()*sz),sz,sz,
                linewidth=0.2,edgecolor='gray',facecolor=cmap(np.clip((v-vmin)/vr,0,1))))


def _plot_samples(samp_buf, model, output_dir, step, K, bot_dim):
    if not samp_buf: return
    n=len(samp_buf); cmap=plt.get_cmap('viridis'); loss_cmap=plt.get_cmap('hot')
    # 6 cols: GT | Teacher pred | Teacher loss | Student pred | Student loss | z_bot
    fig, axes = plt.subplots(n, 6, figsize=(28, 5*n))
    if n==1: axes=axes[np.newaxis,:]

    for row, samp in enumerate(samp_buf):
        qt=samp['qt']; leaves=samp['u_leaves']
        k1,k2,s=samp['k1'],samp['k2'],samp['step']
        pred_T,pred_S,loss_T,loss_S,z=_collect_both(model,qt)

        _draw_field(axes[row,0],leaves,lambda nd:nd.u_val,f"GT u k1={k1}k2={k2} s={s}",cmap)
        _draw_pred(axes[row,1],qt,pred_T,"Teacher pred",cmap)
        maxLT=max(loss_T.values()) if loss_T else 1.0
        _draw_pred(axes[row,2],qt,{k:v/max(maxLT,1e-10) for k,v in loss_T.items()},
                   f"Teacher loss\nmax={maxLT:.5f}",loss_cmap,d_range=(0,1))
        _draw_pred(axes[row,3],qt,pred_S,"Student pred",cmap)
        maxLS=max(loss_S.values()) if loss_S else 1.0
        _draw_pred(axes[row,4],qt,{k:v/max(maxLS,1e-10) for k,v in loss_S.items()},
                   f"Student loss\nmax={maxLS:.5f}",loss_cmap,d_range=(0,1))
        ax=axes[row,5]
        z_np=z.numpy().flatten()
        ax.bar(range(len(z_np)),z_np,color='steelblue',alpha=0.8,width=1.0)
        ax.axhline(0,color='k',lw=0.5)
        for ki in range(1,K): ax.axvline(ki*bot_dim-0.5,color='red',lw=0.8,ls='--',alpha=0.5)
        ax.set_title(f"z_bot K={K},d={bot_dim}\nstd={z_np.std():.3f}",fontsize=9)

    plt.suptitle(f"Distill AE  K={K} d={bot_dim} latent={K*bot_dim} — step {step}",fontsize=12)
    plt.tight_layout()
    out=f'{output_dir}/samples_step{step:04d}.png'
    plt.savefig(out,dpi=130,bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out}")


def _plot_losses(history, output_dir, step, K, bot_dim):
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    for ax,(key,color,label) in zip(axes,[
        ('total','navy','Total'),('teacher','steelblue','Teacher Loss'),
        ('student','orange','Student Distill'),
        ('val_T','green','Teacher Val MSE'),('val_S','red','Student Val MSE')]):
        ax.plot(history[key],lw=1,alpha=0.8,color=color)
        ax.set_xlabel('Step'); ax.grid(True,alpha=0.3)
        last=history[key][-1] if history[key] else 0
        ax.set_title(f"{label}\nfinal={last:.5f}",fontsize=9)
    plt.suptitle(f'Distill Losses  K={K} d={bot_dim}  step {step}',fontsize=12)
    plt.tight_layout()
    out=f'{output_dir}/losses_step{step:04d}.png'
    plt.savefig(out,dpi=130,bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out}")


def validate(model_path, output_dir, K, bot_dim, num_samples=10, device='cpu'):
    print(f"\n--- Validation K={K} bot_dim={bot_dim} ---")
    ckpt  = torch.load(model_path, map_location=device)
    model = DistillQuadAE(in_c=1, hidden=128, max_depth=MAX_LEVEL, pos_freqs=6,
                          bottleneck_depth=2, K=K, bot_dim=bot_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()

    val_dir=os.path.join(output_dir,'validation'); os.makedirs(val_dir,exist_ok=True)
    mse_fn=nn.MSELoss(); all_z=[]; depth_loss_T={d:[] for d in range(MAX_LEVEL+1)}
    depth_loss_S={d:[] for d in range(MAX_LEVEL+1)}

    for i in range(num_samples):
        u_keys,u_clean,u_noisy,u_leaves,k1,k2,denom=generate_u_sample()
        qt=Quadtree(max_depth=MAX_LEVEL,device=device)
        qt.build_from_leaves(u_keys,u_noisy,u_clean)
        with torch.no_grad():
            (_, val_T),(_, val_S),z_bot=model(qt)
        all_z.append(z_bot.cpu().numpy().flatten())
        for d in range(MAX_LEVEL+1):
            mask=qt.leaf_mask[d]
            if mask is None or mask.numel()==0 or not mask.any(): continue
            depth_loss_T[d].append(mse_fn(val_T[d][mask],qt.values_gt[d][mask]).item())
            if val_S[d].shape[0]==qt.keys[d].shape[0]:
                depth_loss_S[d].append(mse_fn(val_S[d][mask],qt.values_gt[d][mask]).item())
        print(f"  [{i+1}/{num_samples}] k1={k1} k2={k2} leaves={len(u_leaves)} |z|={z_bot.norm():.3f}")

    depths=[d for d in range(MAX_LEVEL+1) if depth_loss_T[d]]
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    axes[0].bar([d-0.2 for d in depths],[np.mean(depth_loss_T[d]) for d in depths],
                0.4,label='Teacher',color='steelblue',alpha=0.8)
    axes[0].bar([d+0.2 for d in depths],[np.mean(depth_loss_S[d]) for d in depths if depth_loss_S[d]],
                0.4,label='Student',color='orange',alpha=0.8)
    axes[0].set_xlabel('Depth'); axes[0].set_ylabel('Avg MSE'); axes[0].legend()
    axes[0].set_title('Teacher vs Student Loss by Depth'); axes[0].grid(True,alpha=0.3)

    all_z_arr=np.stack(all_z)
    im=axes[1].imshow(all_z_arr,aspect='auto',cmap='RdBu_r',
                      vmin=-3*all_z_arr.std(),vmax=3*all_z_arr.std())
    for ki in range(1,K): axes[1].axvline(ki*bot_dim-0.5,color='yellow',lw=1,ls='--')
    axes[1].set_xlabel('z_bot dim'); axes[1].set_ylabel('Sample')
    axes[1].set_title(f'z_bot across {num_samples} samples'); plt.colorbar(im,ax=axes[1])

    z_norms=[np.linalg.norm(z) for z in all_z]
    axes[2].hist(z_norms,bins=10,color='steelblue',alpha=0.8,edgecolor='black')
    axes[2].set_xlabel('||z_bot||'); axes[2].set_title(f'Latent norm mean={np.mean(z_norms):.3f}')

    plt.suptitle(f'Distill Validation  K={K} d={bot_dim}  latent={K*bot_dim}',fontsize=13)
    plt.tight_layout()
    out=f'{val_dir}/validation_summary.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out}")


# ==========================================
# 9. Main
# ==========================================

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    print("Distillation QuadAE: teacher (per-node skips) + student (attn bottleneck)")
    print("Student matches teacher output — not GT directly.")

    variants = [(16,16), (16,8), (8,16)]   # (K, bot_dim)
    results  = {}
    for K, bot_dim in variants:
        loss_T, loss_S = train_variant(K, bot_dim, device, distill_weight=1.0)
        results[(K,bot_dim)] = (loss_T, loss_S)

    print(f"\n{'='*65}")
    print("Summary:")
    for (K,d),(lT,lS) in results.items():
        print(f"  K={K:2d} d={d:2d} latent={K*d:4d}  teacher={lT:.6f}  student={lS:.6f}")
    print(f"{'='*65}")


if __name__=='__main__':
    main()
