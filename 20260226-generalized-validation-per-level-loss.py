#!/usr/bin/env python3
"""
Validation script for Generalized-Split model (uses regular QuadConv).
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
from sklearn.decomposition import PCA
import seaborn as sns

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
# 2. Positional Encoding
# ==========================================

def node_centers_from_keys(keys: torch.Tensor, depth: int, max_depth: int, device='cpu'):
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
# 3. Quadtree Data Structure
# ==========================================

class Quadtree:
    def __init__(self, max_depth=8, device='cpu'):
        self.max_depth = max_depth
        self.device = device
        self.keys = [None] * (max_depth + 1)
        self.children_idx = [None] * (max_depth + 1)
        self.parent_idx = [None] * (max_depth + 1)
        self.neighs = [None] * (max_depth + 1)
        self.features_in = [None] * (max_depth + 1)
        self.split_gt = [None] * max_depth
        self.leaf_mask = [None] * (max_depth + 1)
        self.values_gt = [None] * (max_depth + 1)

    def build_from_leaves(self, leaf_keys_by_depth, leaf_vals_noisy_by_depth, leaf_vals_clean_by_depth):
        all_keys = [set() for _ in range(self.max_depth + 1)]
        leaf_set = [set() for _ in range(self.max_depth + 1)]

        for d in range(self.max_depth + 1):
            if leaf_keys_by_depth[d] is not None and len(leaf_keys_by_depth[d]) > 0:
                for k in leaf_keys_by_depth[d].tolist():
                    all_keys[d].add(k)
                    leaf_set[d].add(k)

        for d in range(self.max_depth, 0, -1):
            for k in all_keys[d]:
                parent_key = k >> 2
                all_keys[d - 1].add(parent_key)

        for d in range(self.max_depth + 1):
            sorted_keys = sorted(all_keys[d])
            self.keys[d] = torch.tensor(sorted_keys, dtype=torch.long, device=self.device)

        for d in range(self.max_depth + 1):
            N = len(self.keys[d])
            self.features_in[d] = torch.zeros((N, 1), device=self.device)
            self.values_gt[d] = torch.zeros((N, 1), device=self.device)
            self.leaf_mask[d] = torch.zeros((N,), dtype=torch.bool, device=self.device)

            if leaf_keys_by_depth[d] is not None and len(leaf_keys_by_depth[d]) > 0:
                leaf_keys_t = leaf_keys_by_depth[d].to(self.device)
                noisy_vals = leaf_vals_noisy_by_depth[d].to(self.device)
                clean_vals = leaf_vals_clean_by_depth[d].to(self.device)

                for i, lk in enumerate(leaf_keys_t.tolist()):
                    idx = (self.keys[d] == lk).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        idx = idx[0]
                        self.features_in[d][idx] = noisy_vals[i]
                        self.values_gt[d][idx] = clean_vals[i]
                        self.leaf_mask[d][idx] = True

        for d in range(self.max_depth):
            N = len(self.keys[d])
            self.children_idx[d] = torch.full((N, 4), -1, dtype=torch.long, device=self.device)
            self.split_gt[d] = torch.zeros((N,), dtype=torch.float32, device=self.device)

            if N == 0:
                continue

            child_keys = self.keys[d + 1]
            if child_keys.numel() == 0:
                continue

            parent_of_child = child_keys >> 2
            child_quad = child_keys & 3

            for ci, (pk, cq) in enumerate(zip(parent_of_child.tolist(), child_quad.tolist())):
                pidx = (self.keys[d] == pk).nonzero(as_tuple=True)[0]
                if len(pidx) > 0:
                    pidx = pidx[0]
                    self.children_idx[d][pidx, cq] = ci
                    self.split_gt[d][pidx] = 1.0

        for d in range(1, self.max_depth + 1):
            N = len(self.keys[d])
            self.parent_idx[d] = torch.full((N,), -1, dtype=torch.long, device=self.device)
            if N == 0:
                continue
            parent_keys = self.keys[d] >> 2
            for i, pk in enumerate(parent_keys.tolist()):
                pidx = (self.keys[d - 1] == pk).nonzero(as_tuple=True)[0]
                if len(pidx) > 0:
                    self.parent_idx[d][i] = pidx[0]

        for d in range(self.max_depth + 1):
            self._construct_neighs(d)

    def _construct_neighs(self, depth):
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
        offsets = torch.tensor([
            [-1, -1], [0, -1], [1, -1],
            [-1, 0], [0, 0], [1, 0],
            [-1, 1], [0, 1], [1, 1]
        ], device=self.device, dtype=torch.long)

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
# 4. Network Modules (Regular QuadConv)
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


class QuadUnpool(nn.Module):
    def forward(self, parent_features, quadtree, depth_child):
        assert depth_child >= 1
        pidx = quadtree.parent_idx[depth_child]
        if pidx.numel() == 0:
            return torch.zeros((0, parent_features.shape[1]), device=parent_features.device)
        return parent_features[pidx]


class TreeEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=64, emb_dim=None, max_depth=8, pos_freqs=6):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.max_depth = max_depth
        self.hidden = hidden
        self.emb_dim = hidden if emb_dim is None else emb_dim

        pos_dim = 3 + 2 * pos_freqs * 3
        self.in_proj = nn.Linear(in_c + pos_dim, hidden)
        self.convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool = QuadPool()

        # Explicit "embedding head" per depth (store E[d])
        self.to_emb = nn.ModuleList([nn.Linear(hidden, self.emb_dim) for _ in range(max_depth + 1)])

        # Per-depth normalization + learned depth gain for stable embeddings
        self.emb_norm = nn.ModuleList([nn.LayerNorm(self.emb_dim) for _ in range(max_depth + 1)])
        self.depth_gain = nn.Parameter(torch.ones(max_depth + 1))

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
            if h[d - 1].shape[0] != pooled.shape[0]:
                raise RuntimeError("Pool produced mismatched parent count")

            h[d - 1] = h[d - 1] + pooled

            if d - 1 >= 1 and h[d - 1].numel() > 0:
                h[d - 1] = F.relu(self.convs[d - 1](h[d - 1], qt, d - 1))

        # Produce explicit embeddings (what you store)
        E = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            if h[d] is None or h[d].numel() == 0:
                E[d] = torch.zeros((0, self.emb_dim), device=qt.device)
            else:
                z = self.to_emb[d](h[d])       # (N_d, emb_dim)
                z = self.emb_norm[d](z)        # normalize per node
                z = self.depth_gain[d] * z     # learned depth scaling
                E[d] = z

        return E


class TreeDecoderTeacherForced(nn.Module):
    def __init__(self, hidden=64, emb_dim=None, out_c=1, max_depth=8, pos_freqs=6):
        super().__init__()
        self.max_depth = max_depth
        self.pos_freqs = pos_freqs
        self.hidden = hidden
        self.emb_dim = hidden if emb_dim is None else emb_dim

        pos_dim = 3 + 2 * pos_freqs * 3

        # Learned root token: the ONLY non-embedding "start"
        self.root_token = nn.Parameter(torch.zeros(1, hidden))

        # Fuse decoder state h with stored embedding E and position
        self.fuse = nn.Sequential(
            nn.Linear(hidden + self.emb_dim + pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # Normalize incoming skip embeddings for robustness
        self.skip_norm = nn.LayerNorm(self.emb_dim)

        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4 * hidden))
        self.val_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))

        self.mix_convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

    def forward(self, qt: Quadtree, emb_list):
        h_by_depth = [None] * (self.max_depth + 1)

        # Start from learned root token (broadcast to number of root nodes, usually 1)
        N0 = len(qt.keys[0])
        if N0 == 0:
            h_by_depth[0] = torch.zeros((0, self.hidden), device=qt.device)
        else:
            h_by_depth[0] = self.root_token.expand(N0, -1).to(qt.device)

        split_logits = [None] * self.max_depth
        val_pred = [None] * (self.max_depth + 1)

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

            # Stored embedding skip (explicit)
            if emb_list is None or emb_list[d] is None or emb_list[d].shape[0] != h.shape[0]:
                skip = torch.zeros((h.shape[0], self.emb_dim), device=h.device)
            else:
                skip = emb_list[d].to(h.device)

            skip = self.skip_norm(skip)
            h = self.fuse(torch.cat([h, skip, pos], dim=1))

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
    def __init__(self, in_c=1, hidden=64, emb_dim=None, max_depth=8, pos_freqs=6):
        super().__init__()
        self.encoder = TreeEncoder(in_c=in_c, hidden=hidden, emb_dim=emb_dim, max_depth=max_depth, pos_freqs=pos_freqs)
        self.decoder = TreeDecoderTeacherForced(hidden=hidden, emb_dim=(hidden if emb_dim is None else emb_dim),
                                                out_c=in_c, max_depth=max_depth, pos_freqs=pos_freqs)

    def forward(self, qt: Quadtree):
        E = self.encoder(qt)
        split_logits, val_pred = self.decoder(qt, E)
        return split_logits, val_pred, E


# ==========================================
# 5. Data Generation
# ==========================================

MAX_LEVEL = 8
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
                QuadNode(self.x, self.y, half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y, half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x, self.y + half, half, self.level + 1, self.max_level, self.min_level),
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

    root = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root.gradient_subdivide(k1, k2, grad_threshold=grad_threshold)

    leaves = []
    root.collect_leaves(leaves, k1, k2)

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


# ==========================================
# 6. Embedding Visualization Tools
# ==========================================

class EmbeddingVisualizer:
    @staticmethod
    def get_pca_colors(embeddings):
        """
        Projects (N, Hidden) -> (N, 3) for RGB visualization.
        """
        if embeddings.shape[0] < 3:
            # Not enough points for PCA, return grey
            return np.ones((embeddings.shape[0], 3)) * 0.5
            
        # 1. PCA projection to 3 components
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(embeddings.detach().cpu().numpy())
        
        # 2. Normalize to [0, 1] for RGB
        p_min = pca_features.min(axis=0)
        p_max = pca_features.max(axis=0)
        rgb = (pca_features - p_min) / (p_max - p_min + 1e-6)
        return rgb

    @staticmethod
    def plot_embedding_spatial(ax, qt, keys, embeddings, depth):
        """Plots the spatial quadtree colored by PCA of embeddings."""
        rgb_values = EmbeddingVisualizer.get_pca_colors(embeddings)
        
        # Get coordinates
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
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Embedding PCA (Depth {depth})\n(Color = Feature Similarity)", fontsize=8)

    @staticmethod
    def plot_numerical_stats(ax_hist, ax_vec, embeddings):
        """
        ax_hist: Plots histogram of ALL activations in this layer.
        ax_vec: Plots the raw vector values of a SINGLE random node.
        """
        data = embeddings.detach().cpu().numpy()
        
        # 1. Histogram of all numbers in the latent space
        ax_hist.hist(data.flatten(), bins=50, color='purple', alpha=0.7)
        ax_hist.set_title("Histogram of Latent Activations", fontsize=8)
        ax_hist.set_yscale('log')
        ax_hist.grid(True, alpha=0.3)
        
        # 2. Bar chart of ONE specific node (e.g., the middle one)
        mid_idx = len(data) // 2
        single_vec = data[mid_idx]
        
        ax_vec.bar(range(len(single_vec)), single_vec, color='teal')
        ax_vec.set_title(f"Raw Vector @ Node {mid_idx} (Dim={len(single_vec)})", fontsize=8)
        ax_vec.set_ylim(single_vec.min(), single_vec.max())
        ax_vec.grid(True, axis='y', alpha=0.3)

    @staticmethod
    def save_embeddings_text(filepath, qt, enc_h_list, max_depth, num_nodes_to_show=10):
        """
        Save raw embedding vectors to a text file so you can see exactly what the decoder receives.
        Shows embeddings at each depth level with node coordinates.
        """
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RAW EMBEDDING VECTORS (What the Decoder Sees)\n")
            f.write("=" * 80 + "\n\n")
            
            for d in range(max_depth + 1):
                emb = enc_h_list[d]
                keys = qt.keys[d]
                
                if emb is None or len(emb) == 0:
                    f.write(f"Depth {d}: No embeddings\n\n")
                    continue
                
                data = emb.detach().cpu().numpy()
                ix, iy = Morton2D.key2xy(keys.cpu(), depth=d)
                
                f.write(f"{'='*60}\n")
                f.write(f"DEPTH {d}: {len(data)} nodes, {data.shape[1]} dimensions\n")
                f.write(f"{'='*60}\n")
                
                # Stats for this depth
                f.write(f"  Stats: min={data.min():.4f}, max={data.max():.4f}, ")
                f.write(f"mean={data.mean():.4f}, std={data.std():.4f}\n")
                f.write(f"  Non-zero ratio: {(np.abs(data) > 1e-6).mean()*100:.1f}%\n\n")
                
                # Show first N nodes
                n_show = min(num_nodes_to_show, len(data))
                f.write(f"  First {n_show} node embeddings:\n")
                f.write(f"  {'-'*56}\n")
                
                for node_idx in range(n_show):
                    x_coord = ix[node_idx].item()
                    y_coord = iy[node_idx].item()
                    vec = data[node_idx]
                    
                    f.write(f"\n  Node {node_idx} @ ({x_coord}, {y_coord}):\n")
                    
                    # Print vector in chunks of 16 for readability
                    for chunk_start in range(0, len(vec), 16):
                        chunk_end = min(chunk_start + 16, len(vec))
                        chunk = vec[chunk_start:chunk_end]
                        formatted = " ".join([f"{v:+.3f}" for v in chunk])
                        f.write(f"    [{chunk_start:3d}-{chunk_end-1:3d}]: {formatted}\n")
                    
                    # Summary stats for this node
                    f.write(f"    -> L2 norm: {np.linalg.norm(vec):.4f}, ")
                    f.write(f"active dims (|v|>0.1): {(np.abs(vec) > 0.1).sum()}/{len(vec)}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF EMBEDDING DUMP\n")
            f.write("=" * 80 + "\n")


# ==========================================
# 6b. Validation Function
# ==========================================

def validate_model(model_path, output_dir, num_samples=5, device='cpu'):
    """Load a model checkpoint and generate validation plots with embedding visualization."""
    
    print(f"\n{'='*60}")
    print(f"Validating model with Embedding Analysis: {model_path}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint loaded:")
    print(f"  - Step: {checkpoint['step']}")
    print(f"  - Total loss: {checkpoint['total_loss']:.6f}")
    print(f"  - Val loss: {checkpoint['val_loss']:.6f}")
    print(f"  - Split loss: {checkpoint['split_loss']:.6f}")
    
    model = TreeAE_DirectionB(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully")
    
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.dirname(model_path))
    
    # Visualize embeddings at a specific 'interesting' depth (second to last has rich structural info)
    viz_depth = 6
    
    cmap = plt.get_cmap('viridis')
    loss_cmap = plt.get_cmap('hot')
    
    # Collect avg loss by depth across ALL samples
    all_samples_loss_by_depth = {d: [] for d in range(MAX_LEVEL + 1)}
    
    with torch.no_grad():
        for i in range(num_samples):
            leaf_keys_by_depth, leaf_vals_noisy, leaf_vals_clean, leaves, k1, k2 = generate_user_data()
            qt = Quadtree(max_depth=MAX_LEVEL, device=device)
            qt.build_from_leaves(leaf_keys_by_depth, leaf_vals_noisy, leaf_vals_clean)
            
            # 1. Run Encoder explicitly to get embeddings
            enc_h_list = model.encoder(qt)
            
            # Log per-depth embedding stats to verify normalization
            print(f"    Embedding stats per depth:")
            for d in range(MAX_LEVEL + 1):
                emb = enc_h_list[d]
                if emb is None or emb.numel() == 0:
                    continue
                m = emb.mean().item()
                s = emb.std().item()
                mn = emb.min().item()
                mx = emb.max().item()
                print(f"      d={d}: mean={m:+.4f} std={s:.4f} min={mn:+.3f} max={mx:+.3f} N={emb.shape[0]}")
            
            # 2. Run Decoder
            split_logits, val_pred_viz = model.decoder(qt, enc_h_list)
            
            # Collect predictions and losses
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
                pr = val_pred_viz[d][mask].cpu().numpy().flatten()
                gt = qt.values_gt[d][mask].cpu().numpy().flatten()
                
                for j in range(len(kd)):
                    key = (d, int(ix[j].item()), int(iy[j].item()))
                    pred_dict[key] = pr[j]
                    leaf_loss = (pr[j] - gt[j]) ** 2
                    loss_dict[key] = leaf_loss
                    loss_by_depth[d].append(leaf_loss)
            
            print(f"  Generated sample {i+1}/{num_samples}: k1={k1}, k2={k2}, leaves={len(leaves)}")
            
            # ==========================================
            # Create 8-panel plot for this sample (2x4)
            # ==========================================
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            
            all_losses = list(loss_dict.values())
            max_loss = max(all_losses) if all_losses else 1.0
            max_loss = max(max_loss, 1e-6)
            
            # Row 0, Col 0: Ground Truth
            ax1 = axes[0, 0]
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_aspect('equal')
            ax1.axis('off')
            ax1.set_title(f"GT (Clean): k1={k1}, k2={k2}", fontsize=10)
            
            for node in leaves:
                normalized_color = (node.val + 1) / 2
                color = cmap(normalized_color)
                rect = patches.Rectangle(
                    (node.x, node.y), node.size, node.size,
                    linewidth=0.5, edgecolor='black', facecolor=color
                )
                ax1.add_patch(rect)
            
            # Row 0, Col 1: Predicted
            ax2 = axes[0, 1]
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_aspect('equal')
            ax2.axis('off')
            ax2.set_title(f"Predicted (Denoised)", fontsize=10)
            
            for node in leaves:
                d = node.level
                res = 1 << d
                ix = int(node.x * res)
                iy = int(node.y * res)
                ix = max(0, min(res - 1, ix))
                iy = max(0, min(res - 1, iy))
                
                pred_val = pred_dict.get((d, ix, iy), 0.0)
                normalized_color = (pred_val + 1) / 2
                normalized_color = np.clip(normalized_color, 0, 1)
                color = cmap(normalized_color)
                rect = patches.Rectangle(
                    (node.x, node.y), node.size, node.size,
                    linewidth=0.5, edgecolor='black', facecolor=color
                )
                ax2.add_patch(rect)
            
            # Row 0, Col 2: Loss Map
            ax3 = axes[0, 2]
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_aspect('equal')
            ax3.axis('off')
            ax3.set_title(f"Per-Leaf MSE Loss", fontsize=10)
            
            for node in leaves:
                d = node.level
                res = 1 << d
                ix = int(node.x * res)
                iy = int(node.y * res)
                ix = max(0, min(res - 1, ix))
                iy = max(0, min(res - 1, iy))
                
                leaf_loss = loss_dict.get((d, ix, iy), 0.0)
                normalized_loss = np.clip(leaf_loss / max_loss, 0, 1)
                color = loss_cmap(normalized_loss)
                rect = patches.Rectangle(
                    (node.x, node.y), node.size, node.size,
                    linewidth=0.5, edgecolor='black', facecolor=color
                )
                ax3.add_patch(rect)
            
            sm = plt.cm.ScalarMappable(cmap=loss_cmap, norm=plt.Normalize(0, max_loss))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
            cbar.set_label('MSE Loss', fontsize=8)
            
            # Row 1, Col 0: Embedding PCA Spatial
            emb = enc_h_list[viz_depth]
            keys = qt.keys[viz_depth]
            
            if emb is not None and len(emb) > 0:
                EmbeddingVisualizer.plot_embedding_spatial(axes[1, 0], qt, keys, emb, viz_depth)
                
                # Row 1, Col 1: Histogram of Latent Activations
                # Row 1, Col 2: Single Vector Barcode
                EmbeddingVisualizer.plot_numerical_stats(axes[1, 1], axes[1, 2], emb)
            else:
                axes[1, 0].text(0.5, 0.5, f"No Embeddings\nat depth {viz_depth}", ha='center', va='center')
                axes[1, 0].axis('off')
                axes[1, 1].text(0.5, 0.5, "N/A", ha='center', va='center')
                axes[1, 1].axis('off')
                axes[1, 2].text(0.5, 0.5, "N/A", ha='center', va='center')
                axes[1, 2].axis('off')
            
            # Row 0, Col 3: Avg Loss by Depth (bar chart)
            ax_depth = axes[0, 3]
            depths, avg_losses = [], []
            for d in range(MAX_LEVEL + 1):
                if loss_by_depth[d]:
                    depths.append(d)
                    avg_losses.append(np.mean(loss_by_depth[d]))
            
            if depths:
                bars = ax_depth.bar(depths, avg_losses, color='steelblue', edgecolor='black')
                ax_depth.set_xlabel('Depth Level', fontsize=9)
                ax_depth.set_ylabel('Avg MSE Loss', fontsize=9)
                ax_depth.set_title('Avg Loss by Depth', fontsize=10)
                ax_depth.set_xticks(range(MAX_LEVEL + 1))
                ax_depth.grid(True, axis='y', alpha=0.3)
                for bar, val in zip(bars, avg_losses):
                    ax_depth.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  f'{val:.4f}', ha='center', va='bottom', fontsize=7)
            else:
                ax_depth.text(0.5, 0.5, "No loss data", ha='center', va='center')
                ax_depth.axis('off')
            
            # Row 1, Col 3: Loss histogram (distribution of all per-leaf losses)
            ax_loss_hist = axes[1, 3]
            if all_losses:
                ax_loss_hist.hist(all_losses, bins=50, color='coral', edgecolor='black', alpha=0.7)
                ax_loss_hist.set_xlabel('MSE Loss', fontsize=9)
                ax_loss_hist.set_ylabel('Count', fontsize=9)
                ax_loss_hist.set_title('Loss Distribution (All Leaves)', fontsize=10)
                ax_loss_hist.grid(True, alpha=0.3)
                ax_loss_hist.axvline(np.mean(all_losses), color='red', linestyle='--', label=f'Mean: {np.mean(all_losses):.4f}')
                ax_loss_hist.legend(fontsize=8)
            else:
                ax_loss_hist.text(0.5, 0.5, "No loss data", ha='center', va='center')
                ax_loss_hist.axis('off')
            
            plt.suptitle(f'Sample {i+1}: {model_name} (Step {checkpoint["step"]}, Loss={checkpoint["total_loss"]:.6f})', fontsize=12)
            plt.tight_layout()
            
            output_path = f'{output_dir}/sample_{i+1:02d}_k{k1}_{k2}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    -> Saved {output_path}")
            
            # Save raw embedding vectors to text file
            text_path = f'{output_dir}/sample_{i+1:02d}_k{k1}_{k2}_embeddings.txt'
            EmbeddingVisualizer.save_embeddings_text(text_path, qt, enc_h_list, MAX_LEVEL, num_nodes_to_show=10)
            print(f"    -> Saved {text_path}")
            
            # Accumulate avg loss per depth for this sample
            for d in range(MAX_LEVEL + 1):
                if loss_by_depth[d]:
                    all_samples_loss_by_depth[d].append(np.mean(loss_by_depth[d]))
    
    # Create summary plot: Average of average losses across all samples by depth
    fig_summary, ax_summary = plt.subplots(1, 1, figsize=(10, 6))
    
    depths = []
    avg_of_avg_losses = []
    std_of_avg_losses = []
    
    for d in range(MAX_LEVEL + 1):
        if all_samples_loss_by_depth[d]:
            depths.append(d)
            avg_of_avg_losses.append(np.mean(all_samples_loss_by_depth[d]))
            std_of_avg_losses.append(np.std(all_samples_loss_by_depth[d]))
    
    if depths:
        bars = ax_summary.bar(depths, avg_of_avg_losses, yerr=std_of_avg_losses, 
                              color='steelblue', edgecolor='black', capsize=5, alpha=0.8)
        ax_summary.set_xlabel('Depth Level', fontsize=12)
        ax_summary.set_ylabel('Average MSE Loss', fontsize=12)
        ax_summary.set_title(f'Average Loss by Depth (Across {num_samples} Samples)\n'
                             f'Model: {model_name} (Step {checkpoint["step"]})', fontsize=14)
        ax_summary.set_xticks(range(MAX_LEVEL + 1))
        ax_summary.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val, std in zip(bars, avg_of_avg_losses, std_of_avg_losses):
            ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.0001,
                            f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    summary_path = f'{output_dir}/summary_avg_loss_by_depth.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig_summary)
    print(f"\n  -> Saved summary plot: {summary_path}")
    
    return checkpoint


# ==========================================
# 7. Main
# ==========================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    # Models to validate (Generalized-Split architecture with regular QuadConv)
    model_paths = [
        '/home/tahmid/Development/OctreeRefineNet/plots/20260226_205831/best_model.pt',
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'/home/tahmid/Development/OctreeRefineNet/plots/validation_generalized_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    results = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            ckpt = validate_model(model_path, output_dir, num_samples=10, device=device)
            results.append({
                'path': model_path,
                'step': ckpt['step'],
                'total_loss': ckpt['total_loss'],
                'val_loss': ckpt['val_loss'],
                'split_loss': ckpt['split_loss'],
            })
        else:
            print(f"Model not found: {model_path}")
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"\n{os.path.basename(os.path.dirname(r['path']))}:")
        print(f"  Step: {r['step']}")
        print(f"  Total Loss: {r['total_loss']:.6f}")
        print(f"  Val Loss: {r['val_loss']:.6f}")
        print(f"  Split Loss: {r['split_loss']:.6f}")
    
    print(f"\nAll validation plots saved to: {output_dir}")
