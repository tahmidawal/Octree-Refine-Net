"""
save_embeddings.py

Loads trained model_f and model_u, generates N Poisson pairs,
encodes each with its respective AE, and saves the paired embeddings.

Output structure (one .pt file per sample):
  {
    'E_f': list of (N_d, emb_dim) tensors  -- f embeddings per depth
    'E_u': list of (N_d, emb_dim) tensors  -- u embeddings per depth
    'keys': list of (N_d,) long tensors    -- Morton keys per depth (same for f and u)
    'k1': int, 'k2': int                   -- frequency params
    'n_leaves': int                        -- total number of leaves
    'max_depth': int
  }

Also saves a manifest.json listing all files + summary stats.

Usage:
    python save_embeddings.py \
        --model_f plots/poisson_XYZ/best_model_f.pt \
        --model_u plots/poisson_XYZ/best_model_u.pt \
        --n_samples 2000 \
        --out_dir embeddings/poisson_XYZ
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

# ==========================================
# Morton / Positional Encoding (same as training)
# ==========================================

class Morton2D:
    @staticmethod
    def _interleave_bits(x):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.long)
        x = x.long() & 0x0000FFFF
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        return x

    @staticmethod
    def _deinterleave_bits(x):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.long)
        x = x.long() & 0x55555555
        x = (x | (x >> 1)) & 0x33333333
        x = (x | (x >> 2)) & 0x0F0F0F0F
        x = (x | (x >> 4)) & 0x00FF00FF
        x = (x | (x >> 8)) & 0x0000FFFF
        return x

    @staticmethod
    def xy2key(x, y, depth=16):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.long)
        if not torch.is_tensor(y): y = torch.tensor(y, dtype=torch.long)
        kx = Morton2D._interleave_bits(x.long())
        ky = Morton2D._interleave_bits(y.long())
        return (kx | (ky << 1)).long()

    @staticmethod
    def key2xy(key, depth=16):
        if not torch.is_tensor(key): key = torch.tensor(key, dtype=torch.long)
        key = key.long()
        return Morton2D._deinterleave_bits(key).long(), Morton2D._deinterleave_bits(key >> 1).long()


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
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(pos.shape[0], -1)
    return torch.cat([pos, enc], dim=1)


# ==========================================
# Quadtree
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

        C_in = next((leaf_vals_input_by_depth[d].shape[1]
                     for d in range(self.max_depth + 1)
                     if leaf_vals_input_by_depth[d].numel() > 0), 1)

        for d in range(self.max_depth + 1):
            lk = leaf_keys_by_depth[d].to(self.device).long()
            lv = leaf_vals_input_by_depth[d].to(self.device).float()
            if lk.numel() == 0:
                self.keys[d] = torch.empty((0,), dtype=torch.long, device=self.device)
                self.features_in[d] = torch.zeros((0, C_in), device=self.device)
            else:
                lk_u, inv = torch.unique(lk, sorted=True, return_inverse=True)
                self.keys[d] = lk_u
                feat = torch.zeros((len(lk_u), C_in), device=self.device)
                cnt  = torch.zeros((len(lk_u), 1), device=self.device)
                feat.index_add_(0, inv, lv)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=self.device))
                self.features_in[d] = feat / cnt.clamp(min=1)

        if self.keys[0].numel() == 0:
            self.keys[0] = torch.tensor([0], dtype=torch.long, device=self.device)
            self.features_in[0] = torch.zeros((1, C_in), device=self.device)

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
                lk_u, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled = torch.zeros((len(lk_u), C_in), device=self.device)
                cnt = torch.zeros((len(lk_u), 1), device=self.device)
                pooled.index_add_(0, inv, lv)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=self.device))
                pooled = pooled / cnt.clamp(min=1)
                idx = torch.searchsorted(kd, lk_u).clamp(0, len(kd)-1)
                found = kd[idx] == lk_u
                feat[idx[found]] = pooled[found]
            self.features_in[d] = feat

        for d in range(self.max_depth):
            kd = self.keys[d]; kn = self.keys[d+1]
            if kd.numel() == 0:
                self.children_idx[d] = torch.empty((0,4), dtype=torch.long, device=self.device); continue
            if kn.numel() == 0:
                self.children_idx[d] = torch.full((len(kd),4), -1, dtype=torch.long, device=self.device); continue
            ck = (kd.unsqueeze(1) << 2) + torch.arange(4, device=self.device).view(1,4)
            idx = torch.searchsorted(kn, ck).clamp(0, len(kn)-1)
            found = kn[idx] == ck
            self.children_idx[d] = torch.where(found, idx, torch.full_like(idx, -1))
        self.children_idx[self.max_depth] = None

        self.parent_idx[0] = None
        for d in range(1, self.max_depth + 1):
            kd = self.keys[d]; kp = self.keys[d-1]
            if kd.numel() == 0:
                self.parent_idx[d] = torch.empty((0,), dtype=torch.long, device=self.device); continue
            self.parent_idx[d] = torch.searchsorted(kp, kd >> 2).clamp(0, len(kp)-1)

        for d in range(self.max_depth + 1):
            self._build_neigh(d)

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
                lk_u, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled = torch.zeros((len(lk_u), C_in), device=self.device)
                cnt = torch.zeros((len(lk_u), 1), device=self.device)
                pooled.index_add_(0, inv, lv_t)
                cnt.index_add_(0, inv, torch.ones((len(inv), 1), device=self.device))
                pooled = pooled / cnt.clamp(min=1)
                idx = torch.searchsorted(kd, lk_u).clamp(0, len(kd)-1)
                found = kd[idx] == lk_u
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

    def _build_neigh(self, depth):
        keys = self.keys[depth]; N = len(keys)
        if N == 0:
            self.neighs[depth] = torch.empty((0,9), dtype=torch.long, device=self.device); return
        if depth == 0:
            neigh = torch.full((N,9), -1, dtype=torch.long, device=self.device)
            neigh[:, 4] = 0; self.neighs[depth] = neigh; return
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor(
            [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],
            device=self.device, dtype=torch.long)
        nc = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        res = 1 << depth
        nx, ny = nc[...,0], nc[...,1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
        n_keys = torch.full((N,9), -1, dtype=torch.long, device=self.device)
        if valid.any():
            n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)
        idx = torch.searchsorted(keys, n_keys.clamp(min=0)).clamp(0, len(keys)-1)
        found = valid & (keys[idx] == n_keys)
        idx[~found] = -1
        self.neighs[depth] = idx


# ==========================================
# Network (same architecture as training)
# ==========================================

class QuadConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.weights = nn.Linear(9 * in_c, out_c)
    def forward(self, features, qt, depth):
        neigh_idx = qt.neighs[depth]; N = neigh_idx.shape[0]
        if N == 0: return torch.zeros((0, self.weights.out_features), device=features.device)
        pad = torch.zeros((1, features.shape[1]), device=features.device)
        fp = torch.cat([features, pad], dim=0)
        gi = neigh_idx.clone(); gi[gi == -1] = N
        return self.weights(fp[gi].view(N, -1))

class QuadPool(nn.Module):
    def forward(self, cf, qt, dc):
        dp = dc - 1; Np = len(qt.keys[dp]); C = cf.shape[1]
        if Np == 0: return torch.zeros((0, C), device=cf.device)
        ch = qt.children_idx[dp]
        pooled = torch.zeros((Np, C), device=cf.device)
        cnt    = torch.zeros((Np, 1), device=cf.device)
        for c in range(4):
            idx = ch[:, c]; mask = idx != -1
            if mask.any(): pooled[mask] += cf[idx[mask]]; cnt[mask] += 1.0
        return pooled / cnt.clamp(min=1.0)

class TreeEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=128, emb_dim=128, max_depth=7, pos_freqs=6):
        super().__init__()
        self.pos_freqs = pos_freqs; self.max_depth = max_depth
        self.hidden = hidden; self.emb_dim = emb_dim
        pos_dim = 3 + 2 * pos_freqs * 3
        self.in_proj  = nn.Linear(in_c + pos_dim, hidden)
        self.convs    = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool     = QuadPool()
        self.to_emb   = nn.ModuleList([nn.Linear(hidden, emb_dim) for _ in range(max_depth + 1)])
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
            h[d-1] = h[d-1] + self.pool(h[d], qt, d)
            if d-1 >= 1 and h[d-1].numel() > 0:
                h[d-1] = F.relu(self.convs[d-1](h[d-1], qt, d-1))
        E = []
        for d in range(self.max_depth + 1):
            if h[d] is None or h[d].numel() == 0:
                E.append(torch.zeros((0, self.emb_dim), device=qt.device))
            else:
                z = self.emb_norm[d](self.to_emb[d](h[d]))
                E.append(self.depth_gain[d] * z)
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
        N0 = len(qt.keys[0])
        h_by_depth = [None] * (self.max_depth + 1)
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
        return *self.decoder(qt, E), E


# ==========================================
# Poisson data generation (same as training)
# ==========================================

MAX_LEVEL = 7
MIN_LEVEL = 6
NOISE_STD = 0.05


def f_function(x, y, k1, k2):
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)

def u_function(x, y, k1, k2):
    return f_function(x, y, k1, k2) / (4 * np.pi**2 * (k1**2 + k2**2))

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
        cx, cy = self.x + self.size/2, self.y + self.size/2
        force_split = self.level < self.min_level
        grad_mag = gradient_magnitude_f(cx, cy, k1, k2)
        max_grad = 2 * np.pi * np.sqrt(k1**2 + k2**2)
        depth_factor = 1.0 + 0.1 * self.level
        if force_split or (grad_mag / max_grad > grad_threshold * depth_factor):
            half = self.size / 2
            self.children = [
                QuadNode(self.x,      self.y,      half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half, self.y,      half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x,      self.y+half, half, self.level+1, self.max_level, self.min_level),
                QuadNode(self.x+half, self.y+half, half, self.level+1, self.max_level, self.min_level),
            ]
            for child in self.children:
                child.gradient_subdivide(k1, k2, grad_threshold)

    def collect_leaves(self, leaves_list, k1, k2):
        if not self.children:
            cx, cy = self.x + self.size/2, self.y + self.size/2
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
            f_vals_clean_by_depth.append(torch.empty((0,1), dtype=torch.float32))
            f_vals_noisy_by_depth.append(torch.empty((0,1), dtype=torch.float32))
            u_vals_clean_by_depth.append(torch.empty((0,1), dtype=torch.float32))
            u_vals_noisy_by_depth.append(torch.empty((0,1), dtype=torch.float32))
        else:
            keys_t = torch.tensor(leaf_keys[d], dtype=torch.long)
            f_c = torch.tensor(f_vals_raw[d], dtype=torch.float32)
            u_c = torch.tensor(u_vals_raw[d], dtype=torch.float32)
            leaf_keys_by_depth.append(keys_t)
            f_vals_clean_by_depth.append(f_c)
            f_vals_noisy_by_depth.append(f_c + noise_std * torch.randn_like(f_c))
            u_vals_clean_by_depth.append(u_c)
            u_vals_noisy_by_depth.append(u_c + noise_std * torch.randn_like(u_c))

    return (leaf_keys_by_depth,
            f_vals_clean_by_depth, f_vals_noisy_by_depth,
            u_vals_clean_by_depth, u_vals_noisy_by_depth,
            leaves, k1, k2)


# ==========================================
# Main: encode and save
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_f',   required=True, help='Path to best_model_f.pt')
    parser.add_argument('--model_u',   required=True, help='Path to best_model_u.pt')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--out_dir',   default='embeddings/poisson')
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model_f encoder
    ckpt_f = torch.load(args.model_f, map_location=device)
    model_f = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    model_f.load_state_dict(ckpt_f['model_state_dict'])
    model_f.eval()
    print(f"model_f loaded — step={ckpt_f['step']}, loss={ckpt_f['total_loss']:.6f}")

    # Load model_u encoder
    ckpt_u = torch.load(args.model_u, map_location=device)
    model_u = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    model_u.load_state_dict(ckpt_u['model_state_dict'])
    model_u.eval()
    print(f"model_u loaded — step={ckpt_u['step']}, loss={ckpt_u['total_loss']:.6f}")

    manifest = []
    n_saved = 0
    n_skipped = 0

    print(f"\nGenerating and encoding {args.n_samples} Poisson pairs...")

    with torch.no_grad():
        for i in tqdm(range(args.n_samples), desc="Encoding"):
            try:
                (leaf_keys, f_clean, f_noisy, u_clean, u_noisy,
                 leaves, k1, k2) = generate_poisson_pair()

                # Build quadtrees (use noisy input, clean target — matches AE training)
                qt_f = Quadtree(max_depth=MAX_LEVEL, device=device)
                qt_f.build_from_leaves(leaf_keys, f_noisy, f_clean)

                qt_u = Quadtree(max_depth=MAX_LEVEL, device=device)
                qt_u.build_from_leaves(leaf_keys, u_noisy, u_clean)

                # Encode
                E_f = model_f.encoder(qt_f)   # list of (N_d, 128) tensors
                E_u = model_u.encoder(qt_u)   # same shapes as E_f (shared topology)

                # Sanity check: shapes must match at every depth
                shapes_ok = all(
                    E_f[d].shape == E_u[d].shape
                    for d in range(MAX_LEVEL + 1)
                )
                if not shapes_ok:
                    print(f"  WARNING sample {i}: shape mismatch, skipping")
                    n_skipped += 1
                    continue

                # Keys are shared (same topology for f and u)
                keys_cpu = [qt_f.keys[d].cpu() for d in range(MAX_LEVEL + 1)]

                # Move embeddings to CPU before saving
                E_f_cpu = [e.cpu() for e in E_f]
                E_u_cpu = [e.cpu() for e in E_u]

                # Record node counts per depth
                n_nodes_per_depth = [int(qt_f.keys[d].shape[0]) for d in range(MAX_LEVEL + 1)]

                fname = f"sample_{i:06d}_k{k1}_{k2}.pt"
                fpath = os.path.join(args.out_dir, fname)

                torch.save({
                    'E_f':              E_f_cpu,           # list[MAX_LEVEL+1] of (N_d, 128)
                    'E_u':              E_u_cpu,           # list[MAX_LEVEL+1] of (N_d, 128)
                    'keys':             keys_cpu,          # list[MAX_LEVEL+1] of (N_d,) long
                    'k1':               k1,
                    'k2':               k2,
                    'n_leaves':         len(leaves),
                    'max_depth':        MAX_LEVEL,
                    'n_nodes_per_depth': n_nodes_per_depth,
                }, fpath)

                manifest.append({
                    'file':              fname,
                    'k1':                k1,
                    'k2':                k2,
                    'n_leaves':          len(leaves),
                    'n_nodes_per_depth': n_nodes_per_depth,
                })
                n_saved += 1

            except Exception as e:
                print(f"  ERROR sample {i}: {e}")
                n_skipped += 1
                continue

    # Save manifest
    manifest_data = {
        'n_saved':      n_saved,
        'n_skipped':    n_skipped,
        'max_depth':    MAX_LEVEL,
        'emb_dim':      128,
        'model_f_path': args.model_f,
        'model_u_path': args.model_u,
        'model_f_step': int(ckpt_f['step']),
        'model_u_step': int(ckpt_u['step']),
        'created':      datetime.now().isoformat(),
        'samples':      manifest,
    }
    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)

    # Print summary stats from a few samples
    print(f"\n{'='*60}")
    print(f"Done. Saved {n_saved} samples, skipped {n_skipped}")
    print(f"Output dir:  {args.out_dir}")
    print(f"Manifest:    {manifest_path}")
    print(f"\nNode count summary (from manifest):")
    for d in range(MAX_LEVEL + 1):
        counts = [s['n_nodes_per_depth'][d] for s in manifest]
        if counts:
            print(f"  depth {d}: mean={np.mean(counts):.1f}  "
                  f"min={np.min(counts)}  max={np.max(counts)}")
    print(f"\nk1,k2 distribution:")
    from collections import Counter
    kk = Counter([(s['k1'], s['k2']) for s in manifest])
    for (k1, k2), cnt in sorted(kk.items()):
        print(f"  k1={k1} k2={k2}: {cnt} samples")


if __name__ == '__main__':
    main()
