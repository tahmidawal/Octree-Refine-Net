"""
train_bottleneck_f.py

Trains a TRUE bottleneck Quadtree Autoencoder on the forcing function f.
PDE: Laplacian(u) = f  on [0,1]^2
f(x,y) = sin(2*pi*k1*x) * sin(2*pi*k2*y)

Compression Strategy: Truncated Latent Tree
- Encoder processes the full tree bottom-up.
- Latent embeddings are ONLY extracted at depths <= LATENT_DEPTH (e.g., 2).
- Decoder must reconstruct depths 3 through MAX_LEVEL entirely from the macro-state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# Hyperparameters
# ==========================================
MAX_LEVEL = 6
MIN_LEVEL = 2
LATENT_DEPTH = 2  # The Bottleneck: Only depths 0, 1, 2 get embeddings!
HIDDEN_DIM = 64
EMB_DIM = 64
NOISE_STD = 0.05
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. Core Utilities (Morton & Positional)
# ==========================================
class Morton2D:
    @staticmethod
    def _interleave_bits(x):
        x = x.long()
        x = x & 0x0000FFFF
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        return x

    @staticmethod
    def _deinterleave_bits(x):
        x = x.long()
        x = x & 0x55555555
        x = (x | (x >> 1)) & 0x33333333
        x = (x | (x >> 2)) & 0x0F0F0F0F
        x = (x | (x >> 4)) & 0x00FF00FF
        x = (x | (x >> 8)) & 0x0000FFFF
        return x

    @staticmethod
    def xy2key(x, y):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.long)
        if not torch.is_tensor(y): y = torch.tensor(y, dtype=torch.long)
        return (Morton2D._interleave_bits(x) | (Morton2D._interleave_bits(y) << 1)).long()

    @staticmethod
    def key2xy(key):
        if not torch.is_tensor(key): key = torch.tensor(key, dtype=torch.long)
        x = Morton2D._deinterleave_bits(key)
        y = Morton2D._deinterleave_bits(key >> 1)
        return x.long(), y.long()

def node_centers_from_keys(keys, depth, max_depth, device):
    if keys.numel() == 0: return torch.zeros((0, 3), device=device)
    ix, iy = Morton2D.key2xy(keys)
    res = float(1 << depth)
    x = (ix.float() + 0.5) / res
    y = (iy.float() + 0.5) / res
    dnorm = torch.full_like(x, float(depth) / float(max_depth))
    return torch.stack([x, y, dnorm], dim=1).to(device)

def fourier_encode(pos, num_freqs=4):
    if pos.numel() == 0: return pos
    freqs = (2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)).view(1, 1, -1)
    x = pos.unsqueeze(-1) * np.pi * 2.0 * freqs
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(pos.shape[0], -1)
    return torch.cat([pos, enc], dim=1)

# ==========================================
# 2. Quadtree Data Structure
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
        if leaf_vals_target_by_depth is None: leaf_vals_target_by_depth = leaf_vals_input_by_depth
        C_in = 1

        # Init keys and features
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
                feat.index_add_(0, inv, lv)
                cnt = torch.zeros((len(lk_unique), 1), device=self.device).index_add_(0, inv, torch.ones_like(lv))
                self.features_in[d] = feat / cnt.clamp(min=1)

        if self.keys[0].numel() == 0:
            self.keys[0] = torch.tensor([0], dtype=torch.long, device=self.device)
            self.features_in[0] = torch.zeros((1, C_in), dtype=torch.float, device=self.device)

        # Save pre-closure keys to remap features after expansion
        pre_keys = [self.keys[d].clone() for d in range(self.max_depth + 1)]

        # Ancestor closure
        for d in range(self.max_depth, 0, -1):
            if self.keys[d].numel() > 0:
                parents = torch.unique(self.keys[d] >> 2, sorted=True)
                self.keys[d-1] = torch.unique(torch.cat([self.keys[d-1], parents]), sorted=True)

        # Realign features_in with (possibly expanded) keys
        for d in range(self.max_depth + 1):
            old_feat = self.features_in[d]
            old_k = pre_keys[d]
            Nd = len(self.keys[d])
            if old_k.shape[0] == Nd:
                continue
            new_feat = torch.zeros((Nd, C_in), device=self.device)
            if old_k.numel() > 0:
                idx = torch.searchsorted(self.keys[d], old_k).clamp(0, Nd - 1)
                found = self.keys[d][idx] == old_k
                new_feat[idx[found]] = old_feat[found]
            self.features_in[d] = new_feat

        # Children and Parents
        self.parent_idx[0] = None
        for d in range(self.max_depth):
            kd, kn = self.keys[d], self.keys[d+1]
            if kd.numel() == 0:
                self.children_idx[d] = torch.empty((0, 4), dtype=torch.long, device=self.device)
            elif kn.numel() == 0:
                self.children_idx[d] = torch.full((len(kd), 4), -1, dtype=torch.long, device=self.device)
            else:
                child_keys = (kd.unsqueeze(1) << 2) + torch.arange(4, device=self.device).view(1, 4)
                idx = torch.searchsorted(kn, child_keys).clamp(0, len(kn)-1)
                self.children_idx[d] = torch.where(kn[idx] == child_keys, idx, -1)
            
            if d > 0 and kd.numel() > 0:
                self.parent_idx[d] = torch.searchsorted(self.keys[d-1], kd >> 2).clamp(0, len(self.keys[d-1])-1)

        self.children_idx[self.max_depth] = None
        self.parent_idx[self.max_depth] = torch.searchsorted(self.keys[self.max_depth-1], self.keys[self.max_depth] >> 2) if self.keys[self.max_depth].numel() > 0 else None

        # Build GT Labels
        for d in range(self.max_depth + 1):
            kd = self.keys[d]
            if kd.numel() == 0:
                self.values_gt[d] = torch.zeros((0, C_in), device=self.device)
                self.leaf_mask[d] = torch.zeros((0,), dtype=torch.bool, device=self.device)
                self.split_gt[d] = None
                continue

            target_feat = torch.zeros((len(kd), C_in), device=self.device)
            lk = leaf_keys_by_depth[d].to(self.device).long()
            lv_tgt = leaf_vals_target_by_depth[d].to(self.device).float()
            if lk.numel() > 0:
                lk_unique, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled = torch.zeros((len(lk_unique), C_in), device=self.device).index_add_(0, inv, lv_tgt)
                cnt = torch.zeros((len(lk_unique), 1), device=self.device).index_add_(0, inv, torch.ones_like(lv_tgt))
                idx = torch.searchsorted(kd, lk_unique).clamp(0, len(kd)-1)
                found = kd[idx] == lk_unique
                target_feat[idx[found]] = (pooled / cnt.clamp(min=1))[found]
            self.values_gt[d] = target_feat

            if d == self.max_depth:
                self.leaf_mask[d] = torch.ones((len(kd),), dtype=torch.bool, device=self.device)
                self.split_gt[d] = None
            else:
                is_leaf = (self.children_idx[d] == -1).all(dim=1)
                self.leaf_mask[d] = is_leaf
                self.split_gt[d] = (~is_leaf).float()

# ==========================================
# 3. Network Architecture (The Bottleneck)
# ==========================================
class QuadPool(nn.Module):
    def forward(self, child_features, qt, depth_child):
        d_parent = depth_child - 1
        Np = len(qt.keys[d_parent])
        if Np == 0: return torch.zeros((0, child_features.shape[1]), device=child_features.device)
        ch = qt.children_idx[d_parent]
        pooled = torch.zeros((Np, child_features.shape[1]), device=child_features.device)
        cnt = torch.zeros((Np, 1), device=child_features.device)
        for c in range(4):
            idx = ch[:, c]; mask = idx != -1
            if mask.any():
                pooled[mask] += child_features[idx[mask]]
                cnt[mask] += 1.0
        return pooled / cnt.clamp(min=1.0)

class BottleneckEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=64, emb_dim=64, max_depth=6, latent_depth=2, pos_freqs=4):
        super().__init__()
        self.max_depth = max_depth
        self.latent_depth = latent_depth
        pos_dim = 3 + 2 * pos_freqs * 3
        
        self.in_proj = nn.Sequential(nn.Linear(in_c + pos_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.pool = QuadPool()
        
        # We only need embedding projections for depths <= latent_depth
        self.to_emb = nn.ModuleList([nn.Linear(hidden, emb_dim) if d <= latent_depth else nn.Identity() for d in range(max_depth + 1)])
        self.emb_norm = nn.ModuleList([nn.LayerNorm(emb_dim) if d <= latent_depth else nn.Identity() for d in range(max_depth + 1)])

    def forward(self, qt):
        hidden = self.in_proj[0].out_features
        h = [None] * (self.max_depth + 1)
        # 1. Project inputs — initialize h to match qt.keys size at each depth
        for d in range(self.max_depth + 1):
            Nd = len(qt.keys[d])
            if qt.features_in[d] is not None and qt.features_in[d].numel() > 0:
                pos = fourier_encode(node_centers_from_keys(qt.keys[d], d, self.max_depth, qt.device))
                h[d] = self.in_proj(torch.cat([qt.features_in[d], pos], dim=1))
            else:
                h[d] = torch.zeros((Nd, hidden), device=qt.device)

        # 2. Bottom-up pooling
        for d in range(self.max_depth, 0, -1):
            if h[d].numel() > 0:
                pooled = self.pool(h[d], qt, d)
                h[d-1] = h[d-1] + pooled

        # 3. Extract Bottleneck Embeddings
        E = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            if d <= self.latent_depth and h[d].numel() > 0:
                E[d] = self.emb_norm[d](self.to_emb[d](h[d]))
        return E

class BottleneckDecoder(nn.Module):
    def __init__(self, hidden=64, emb_dim=64, out_c=1, max_depth=6, latent_depth=2, pos_freqs=4):
        super().__init__()
        self.max_depth = max_depth
        self.latent_depth = latent_depth
        pos_dim = 3 + 2 * pos_freqs * 3
        
        self.root_token = nn.Parameter(torch.zeros(1, hidden))
        self.skip_norm = nn.LayerNorm(emb_dim)
        
        # More capacity here because it has to generate structure without skips!
        self.fuse = nn.Sequential(
            nn.Linear(hidden + emb_dim + pos_dim, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden)
        )
        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, 4 * hidden))
        self.val_head   = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_c))

    def forward(self, qt, emb_list):
        h_by_depth = [None] * (self.max_depth + 1)
        N0 = len(qt.keys[0])
        h_by_depth[0] = self.root_token.expand(N0, -1).to(qt.device) if N0 > 0 else torch.zeros((0, self.fuse[0].in_features), device=qt.device)

        split_logits = [None] * self.max_depth
        val_pred = [None] * (self.max_depth + 1)

        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=qt.device)
                if d < self.max_depth: split_logits[d] = torch.zeros((0,), device=qt.device)
                continue

            pos = fourier_encode(node_centers_from_keys(qt.keys[d], d, self.max_depth, qt.device))

            # Apply skip connection ONLY if we are at or above the latent depth
            if d <= self.latent_depth and emb_list[d] is not None and emb_list[d].shape[0] == h.shape[0]:
                skip = self.skip_norm(emb_list[d])
            else:
                skip = torch.zeros((h.shape[0], self.skip_norm.weight.shape[0]), device=h.device)

            h = self.fuse(torch.cat([h, skip, pos], dim=1))
            val_pred[d] = self.val_head(h)

            if d == self.max_depth: break

            split_logits[d] = self.split_head(h).squeeze(-1)

            # Top-down child generation
            ch = qt.children_idx[d]
            has_child = (ch != -1).any(dim=1)
            N_next = len(qt.keys[d+1])
            h_next = torch.zeros((N_next, h.shape[1]), device=h.device)

            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, h.shape[1])
                # Vectorized scatter: map parent children indices to child features
                ch_sub = ch[has_child]                        # (N_parents, 4)
                valid = ch_sub != -1                          # (N_parents, 4)
                dst = ch_sub[valid]                           # flat destination indices
                src = child_feats[valid]                      # flat source features
                h_next[dst] = src

            h_by_depth[d+1] = h_next

        return split_logits, val_pred

class BottleneckTreeAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BottleneckEncoder(hidden=HIDDEN_DIM, emb_dim=EMB_DIM, max_depth=MAX_LEVEL, latent_depth=LATENT_DEPTH)
        self.decoder = BottleneckDecoder(hidden=HIDDEN_DIM, emb_dim=EMB_DIM, max_depth=MAX_LEVEL, latent_depth=LATENT_DEPTH)

    def forward(self, qt):
        E = self.encoder(qt)
        split_logits, val_pred = self.decoder(qt, E)
        return split_logits, val_pred, E

# ==========================================
# 4. Data Generation
# ==========================================
def gradient_magnitude_f(x, y, k1, k2):
    dfdx = 2*np.pi*k1 * np.cos(2*np.pi*k1*x) * np.sin(2*np.pi*k2*y)
    dfdy = 2*np.pi*k2 * np.sin(2*np.pi*k1*x) * np.cos(2*np.pi*k2*y)
    return np.sqrt(dfdx**2 + dfdy**2)

class QuadNode:
    def __init__(self, x, y, size, level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.children = []
        self.f_val = None

    def subdivide(self, k1, k2, threshold=0.3):
        if self.level >= MAX_LEVEL: return
        cx, cy = self.x + self.size/2, self.y + self.size/2
        grad = gradient_magnitude_f(cx, cy, k1, k2) / (2 * np.pi * np.sqrt(k1**2 + k2**2))
        
        if self.level < MIN_LEVEL or grad > (threshold * (1.0 + 0.1 * self.level)):
            half = self.size / 2
            self.children = [
                QuadNode(self.x, self.y, half, self.level+1),
                QuadNode(self.x+half, self.y, half, self.level+1),
                QuadNode(self.x, self.y+half, half, self.level+1),
                QuadNode(self.x+half, self.y+half, half, self.level+1)
            ]
            for c in self.children: c.subdivide(k1, k2, threshold)

    def collect(self, k1, k2, leaves):
        if not self.children:
            self.f_val = np.sin(2*np.pi*k1*(self.x + self.size/2)) * np.sin(2*np.pi*k2*(self.y + self.size/2))
            leaves.append(self)
        else:
            for c in self.children: c.collect(k1, k2, leaves)

def generate_f_sample():
    k1, k2 = random.randint(1, 4), random.randint(1, 4)
    root = QuadNode(0, 0, 1.0, 0)
    root.subdivide(k1, k2)
    leaves = []
    root.collect(k1, k2, leaves)
    
    keys = [[] for _ in range(MAX_LEVEL + 1)]
    vals = [[] for _ in range(MAX_LEVEL + 1)]
    for nd in leaves:
        d, res = nd.level, 1 << nd.level
        ix, iy = int(nd.x * res), int(nd.y * res)
        keys[d].append(int(Morton2D.xy2key(ix, iy).item()))
        vals[d].append([nd.f_val])
        
    keys_t, clean_t, noisy_t = [], [], []
    for d in range(MAX_LEVEL + 1):
        if not keys[d]:
            keys_t.append(torch.empty((0,), dtype=torch.long))
            clean_t.append(torch.empty((0, 1), dtype=torch.float32))
            noisy_t.append(torch.empty((0, 1), dtype=torch.float32))
        else:
            keys_t.append(torch.tensor(keys[d], dtype=torch.long))
            c = torch.tensor(vals[d], dtype=torch.float32)
            clean_t.append(c)
            noisy_t.append(c + NOISE_STD * torch.randn_like(c))
            
    return keys_t, clean_t, noisy_t

# ==========================================
# 5. Training Loop
# ==========================================
def main():
    print(f"Training True Bottleneck Autoencoder on {DEVICE}")
    print(f"Max Depth: {MAX_LEVEL} | Latent Bottleneck Depth: {LATENT_DEPTH}")
    
    model = BottleneckTreeAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    
    steps = 1000
    for step in range(steps):
        keys, clean, noisy = generate_f_sample()
        qt = Quadtree(MAX_LEVEL, DEVICE)
        qt.build_from_leaves(keys, noisy, clean)
        
        opt.zero_grad()
        split_logits, val_pred, E = model(qt)
        
        # Calculate compressed size
        latent_nodes = sum(e.shape[0] for e in E if e is not None)
        total_nodes = sum(len(qt.keys[d]) for d in range(MAX_LEVEL + 1))
        
        L_val = torch.tensor(0.0, device=DEVICE)
        n_val = 0
        for d in range(MAX_LEVEL + 1):
            mask = qt.leaf_mask[d]
            if mask is not None and mask.any():
                L_val += mse(val_pred[d][mask], qt.values_gt[d][mask])
                n_val += 1
        if n_val > 0: L_val /= n_val

        L_split = torch.tensor(0.0, device=DEVICE)
        n_split = 0
        for d in range(MAX_LEVEL):
            if qt.split_gt[d] is not None and qt.split_gt[d].numel() > 0:
                L_split += bce(split_logits[d], qt.split_gt[d])
                n_split += 1
        if n_split > 0: L_split /= n_split

        L_total = L_val + 0.5 * L_split
        L_total.backward()
        opt.step()
        
        if step % 50 == 0:
            print(f"Step {step:4d} | Total Loss: {L_total.item():.5f} | MSE: {L_val.item():.5f} | "
                  f"Compression: {latent_nodes} latent nodes / {total_nodes} total nodes")

if __name__ == '__main__':
    main()