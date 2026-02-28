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
# 1. Morton Codes & Positional Encoding
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
        if leaf_vals_target_by_depth is None:
            leaf_vals_target_by_depth = leaf_vals_input_by_depth

        C_in = 1
        for d in range(self.max_depth + 1):
            if leaf_vals_input_by_depth[d].numel() > 0:
                C_in = leaf_vals_input_by_depth[d].shape[1]
                break

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

        for d in range(self.max_depth, 0, -1):
            if self.keys[d].numel() == 0: continue
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
                idx = torch.searchsorted(kd, lk_unique).clamp(0, len(kd) - 1)
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
            idx = torch.searchsorted(kn, child_keys).clamp(0, len(kn) - 1)
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
            pidx = torch.searchsorted(kp, parent_keys).clamp(0, len(kp) - 1)
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
        offsets = torch.tensor([[-1,-1],[-1,0],[-1,1],[ 0,-1],[ 0,0],[ 0,1],[ 1,-1],[ 1,0],[ 1,1]], device=self.device, dtype=torch.long)
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0) 

        res = 1 << depth
        nx = n_coords[..., 0]
        ny = n_coords[..., 1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)

        n_keys = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
        if valid.any(): n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)

        idx = torch.searchsorted(keys, n_keys.clamp(min=0)).clamp(0, len(keys) - 1)
        found = valid & (keys[idx] == n_keys)
        idx[~found] = -1
        self.neighs[depth] = idx

# ==========================================
# 3. Model Architecture
# ==========================================
class QuadConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Linear(9 * in_channels, out_channels)

    def forward(self, features, quadtree, depth):
        neigh_idx = quadtree.neighs[depth] 
        N = neigh_idx.shape[0]
        if N == 0: return torch.zeros((0, self.weights.out_features), device=features.device)

        pad_vec = torch.zeros((1, features.shape[1]), device=features.device)
        feat_padded = torch.cat([features, pad_vec], dim=0)
        gather_idx = neigh_idx.clone()
        gather_idx[gather_idx == -1] = N 
        col_flat = feat_padded[gather_idx].view(N, -1)
        return self.weights(col_flat)

class QuadPool(nn.Module):
    def forward(self, child_features, quadtree, depth_child):
        d_parent = depth_child - 1
        parent_keys = quadtree.keys[d_parent]
        Np, C = len(parent_keys), child_features.shape[1]
        if Np == 0: return torch.zeros((0, C), device=child_features.device)

        ch = quadtree.children_idx[d_parent] 
        pooled = torch.zeros((Np, C), device=child_features.device)
        cnt = torch.zeros((Np, 1), device=child_features.device)

        for c in range(4):
            idx = ch[:, c]
            mask = (idx != -1)
            if mask.any():
                pooled[mask] += child_features[idx[mask]]
                cnt[mask] += 1.0
        return pooled / cnt.clamp(min=1.0)

class TreeEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=64, emb_dim=64, max_depth=8, pos_freqs=6):
        super().__init__()
        self.max_depth = max_depth
        self.hidden = hidden
        self.emb_dim = emb_dim
        self.pos_freqs = pos_freqs
        
        pos_dim = 3 + 2 * pos_freqs * 3
        self.in_proj = nn.Linear(in_c + pos_dim, hidden)
        self.convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])
        self.pool = QuadPool()
        self.to_emb = nn.ModuleList([nn.Linear(hidden, self.emb_dim) for _ in range(max_depth + 1)])
        self.emb_norm = nn.ModuleList([nn.LayerNorm(self.emb_dim) for _ in range(max_depth + 1)])
        self.depth_gain = nn.Parameter(torch.ones(max_depth + 1))

    def forward(self, qt):
        h = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            if qt.features_in[d] is None or qt.features_in[d].numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=qt.device)
                continue
            pos = fourier_encode(node_centers_from_keys(qt.keys[d], d, self.max_depth, qt.device), num_freqs=self.pos_freqs)
            h[d] = self.in_proj(torch.cat([qt.features_in[d], pos], dim=1))

        for d in range(self.max_depth, 0, -1):
            if h[d].numel() == 0: continue
            pooled = self.pool(h[d], qt, d) 
            h[d - 1] = h[d - 1] + pooled
            if d - 1 >= 1 and h[d - 1].numel() > 0:
                h[d - 1] = F.relu(self.convs[d - 1](h[d - 1], qt, d - 1))

        E = [None] * (self.max_depth + 1)
        for d in range(self.max_depth + 1):
            if h[d].numel() == 0:
                E[d] = torch.zeros((0, self.emb_dim), device=qt.device)
            else:
                z = self.emb_norm[d](self.to_emb[d](h[d]))
                E[d] = self.depth_gain[d] * z
        return E 

class TreeDecoderTeacherForced(nn.Module):
    def __init__(self, hidden=64, emb_dim=64, out_c=1, max_depth=8, pos_freqs=6):
        super().__init__()
        self.max_depth = max_depth
        self.hidden = hidden
        self.emb_dim = emb_dim
        self.pos_freqs = pos_freqs

        pos_dim = 3 + 2 * pos_freqs * 3
        self.root_token = nn.Parameter(torch.zeros(1, hidden))
        self.fuse = nn.Sequential(nn.Linear(hidden + emb_dim + pos_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.skip_norm = nn.LayerNorm(emb_dim)
        
        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4 * hidden))
        self.val_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))
        self.mix_convs = nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth + 1)])

    def forward(self, qt, emb_list):
        h_by_depth = [None] * (self.max_depth + 1)
        N0 = len(qt.keys[0])
        h_by_depth[0] = self.root_token.expand(N0, -1).to(qt.device) if N0 > 0 else torch.zeros((0, self.hidden), device=qt.device)

        split_logits, val_pred = [None] * self.max_depth, [None] * (self.max_depth + 1)

        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=qt.device)
                if d < self.max_depth: split_logits[d] = torch.zeros((0,), device=qt.device)
                continue

            pos = fourier_encode(node_centers_from_keys(qt.keys[d], d, self.max_depth, qt.device), num_freqs=self.pos_freqs)
            
            # Using aligned embeddings ensuring shapes match
            skip = emb_list[d].to(h.device)
            skip = self.skip_norm(skip)
            
            h = self.fuse(torch.cat([h, skip, pos], dim=1))
            if h.numel() > 0 and d >= 1: h = F.relu(self.mix_convs[d](h, qt, d))
            
            val_pred[d] = self.val_head(h)
            if d == self.max_depth: break

            split_logits[d] = self.split_head(h).squeeze(-1)

            ch = qt.children_idx[d] 
            has_child = (ch != -1).any(dim=1)
            h_next = torch.zeros((len(qt.keys[d + 1]), self.hidden), device=h.device)

            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, self.hidden)
                parent_rows = torch.nonzero(has_child).squeeze(-1)
                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1: h_next[ci] = child_feats[t, c]
            h_by_depth[d + 1] = h_next

        return split_logits, val_pred

# ==========================================
# 4. Latent Space Alignment (The Crucial Bridge)
# ==========================================
def align_embeddings(qt_in, E_in, qt_out, emb_dim, max_depth):
    """
    Spatially resamples embeddings from the Input Topology to the Target Topology.
    This solves the skip-connection crash when the encoder and decoder have different mesh structures.
    """
    device = qt_in.device
    res = 1 << max_depth
    grid = torch.zeros((1, emb_dim, res, res), device=device)

    # 1. Splat `qt_in` embeddings to a dense spatial grid
    for d in range(max_depth + 1):
        if qt_in.keys[d].numel() == 0: continue
        ix, iy = Morton2D.key2xy(qt_in.keys[d], depth=d)
        scale = 1 << (max_depth - d)
        
        dense_d = torch.zeros((1, emb_dim, 1 << d, 1 << d), device=device)
        dense_d[0, :, ix.long(), iy.long()] = E_in[d].T
        
        mask_d = torch.zeros((1, 1, 1 << d, 1 << d), device=device)
        mask_d[0, 0, ix.long(), iy.long()] = 1.0
        
        if scale > 1:
            dense_max = F.interpolate(dense_d, scale_factor=scale, mode='nearest')
            mask_max = F.interpolate(mask_d, scale_factor=scale, mode='nearest').bool()
        else:
            dense_max = dense_d
            mask_max = mask_d.bool()
            
        # Deeper layers overwrite shallower layers
        grid = torch.where(mask_max, dense_max, grid)

    # 2. Sample from the dense grid at `qt_out` node locations
    E_aligned = [None] * (max_depth + 1)
    grid_squeeze = grid.squeeze(0) # (emb_dim, res, res)
    
    for d in range(max_depth + 1):
        if qt_out.keys[d].numel() == 0:
            E_aligned[d] = torch.zeros((0, emb_dim), device=device)
            continue
            
        ix, iy = Morton2D.key2xy(qt_out.keys[d], depth=d)
        scale = 1 << (max_depth - d)
        
        # Sample at the center of the target quad
        ix_max = (ix.long() * scale + (scale // 2)).clamp(0, res - 1)
        iy_max = (iy.long() * scale + (scale // 2)).clamp(0, res - 1)
        
        E_aligned[d] = grid_squeeze[:, ix_max, iy_max].T # Shape (N, emb_dim)
        
    return E_aligned

# ==========================================
# 5. Domain Generation (Indicator -> Continuous)
# ==========================================
def target_function(x, y, k1, k2):
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)

class QuadNode:
    def __init__(self, x, y, size, level, max_level, min_level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.max_level, self.min_level = max_level, min_level
        self.children, self.val = [], None

    def indicator_subdivide(self, cx, cy, radius, max_level):
        """Splits only if the boundary of the circle cuts through this node."""
        if self.level >= max_level: return
        center_x, center_y = self.x + self.size / 2, self.y + self.size / 2
        dist = np.sqrt((center_x - cx)**2 + (center_y - cy)**2)
        node_radius = (self.size / 2) * 1.414 # Distance to corner
        
        # If boundary falls inside this node's radius, we split
        intersects_boundary = (dist > radius - node_radius) and (dist < radius + node_radius)
        force_split = self.level < self.min_level
        
        if intersects_boundary or force_split:
            half = self.size / 2
            self.children = [
                QuadNode(self.x, self.y, half, self.level+1, max_level, self.min_level),
                QuadNode(self.x+half, self.y, half, self.level+1, max_level, self.min_level),
                QuadNode(self.x, self.y+half, half, self.level+1, max_level, self.min_level),
                QuadNode(self.x+half, self.y+half, half, self.level+1, max_level, self.min_level)
            ]
            for c in self.children: c.indicator_subdivide(cx, cy, radius, max_level)

    def custom_patch_subdivide(self, k1, k2, cx, cy, radius, base_level, mid_level, max_level):
        """Creates continuous patches based on distance from center."""
        node_cx, node_cy = self.x + self.size / 2, self.y + self.size / 2
        dist = np.sqrt((node_cx - cx)**2 + (node_cy - cy)**2)
        buffer = self.size / 2
        
        if dist <= radius + buffer: target_level = max_level
        else: target_level = base_level

        if self.level >= target_level: return
        
        half = self.size / 2
        self.children = [
            QuadNode(self.x, self.y, half, self.level+1, max_level, self.min_level),
            QuadNode(self.x+half, self.y, half, self.level+1, max_level, self.min_level),
            QuadNode(self.x, self.y+half, half, self.level+1, max_level, self.min_level),
            QuadNode(self.x+half, self.y+half, half, self.level+1, max_level, self.min_level)
        ]
        for c in self.children: c.custom_patch_subdivide(k1, k2, cx, cy, radius, base_level, mid_level, max_level)

    def collect_leaves_indicator(self, leaves_list, cx, cy, radius):
        if not self.children:
            center_x, center_y = self.x + self.size / 2, self.y + self.size / 2
            dist = np.sqrt((center_x - cx)**2 + (center_y - cy)**2)
            self.val = 1.0 if dist <= radius else -1.0 # Indicator function!
            leaves_list.append(self)
        else:
            for child in self.children: child.collect_leaves_indicator(leaves_list, cx, cy, radius)
            
    def collect_leaves_continuous(self, leaves_list, k1, k2):
        if not self.children:
            center_x, center_y = self.x + self.size / 2, self.y + self.size / 2
            self.val = target_function(center_x, center_y, k1, k2) # Continuous target!
            leaves_list.append(self)
        else:
            for child in self.children: child.collect_leaves_continuous(leaves_list, k1, k2)

def generate_translation_data(max_level=7, noise_std=0.05):
    k1, k2 = random.randint(1, 5), random.randint(1, 5)
    cx, cy = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
    radius = random.uniform(0.15, 0.3)
    
    # --- GENERATE INPUT (INDICATOR TOPOLOGY) ---
    root_in = QuadNode(0, 0, 1.0, 0, max_level, min_level=3)
    root_in.indicator_subdivide(cx, cy, radius, max_level)
    leaves_in = []
    root_in.collect_leaves_indicator(leaves_in, cx, cy, radius)
    
    keys_in, vals_in = [[] for _ in range(max_level + 1)], [[] for _ in range(max_level + 1)]
    for node in leaves_in:
        d = node.level
        res = 1 << d
        ix = max(0, min(res - 1, int(node.x * res)))
        iy = max(0, min(res - 1, int(node.y * res)))
        keys_in[d].append(int(Morton2D.xy2key(ix, iy).item()))
        vals_in[d].append([float(node.val)])

    # --- GENERATE OUTPUT (CONTINUOUS PATCH TOPOLOGY) ---
    root_out = QuadNode(0, 0, 1.0, 0, max_level, min_level=5)
    root_out.custom_patch_subdivide(k1, k2, cx, cy, radius, base_level=5, mid_level=6, max_level=7)
    leaves_out = []
    root_out.collect_leaves_continuous(leaves_out, k1, k2)
    
    keys_out, vals_out = [[] for _ in range(max_level + 1)], [[] for _ in range(max_level + 1)]
    for node in leaves_out:
        d = node.level
        res = 1 << d
        ix = max(0, min(res - 1, int(node.x * res)))
        iy = max(0, min(res - 1, int(node.y * res)))
        keys_out[d].append(int(Morton2D.xy2key(ix, iy).item()))
        vals_out[d].append([float(node.val)])
        
    def to_tensors(keys, vals):
        k_t, v_t = [], []
        for d in range(max_level + 1):
            if len(keys[d]) == 0:
                k_t.append(torch.empty((0,), dtype=torch.long))
                v_t.append(torch.empty((0, 1), dtype=torch.float32))
            else:
                k_t.append(torch.tensor(keys[d], dtype=torch.long))
                v_t.append(torch.tensor(vals[d], dtype=torch.float32))
        return k_t, v_t

    k_in_t, v_in_t = to_tensors(keys_in, vals_in)
    k_out_t, v_out_clean_t = to_tensors(keys_out, vals_out)
    v_out_noisy_t = [v + noise_std * torch.randn_like(v) if v.numel() > 0 else v for v in v_out_clean_t]

    return k_in_t, v_in_t, leaves_in, k_out_t, v_out_noisy_t, v_out_clean_t, leaves_out

# ==========================================
# 6. Main Execution (Domain Translation Training)
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_LEVEL = 7
    emb_dim = 128
    
    encoder = TreeEncoder(in_c=1, hidden=128, emb_dim=emb_dim, max_depth=MAX_LEVEL).to(device)
    decoder = TreeDecoderTeacherForced(hidden=128, emb_dim=emb_dim, out_c=1, max_depth=MAX_LEVEL).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    num_steps = 3000

    print("Training Translation Model (Indicator Quadtree -> Patch Quadtree)...")
    
    for step in range(num_steps):
        # Generate entirely different topological trees
        k_in, v_in, leaves_in, k_out, v_out_noisy, v_out_clean, leaves_out = generate_translation_data(max_level=MAX_LEVEL)

        qt_in = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_in.build_from_leaves(k_in, v_in, v_in) # Indicator input
        
        qt_out = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_out.build_from_leaves(k_out, v_out_noisy, v_out_clean) # Continuous target

        optimizer.zero_grad()
        
        # 1. Encode the Indicator topology
        E_in = encoder(qt_in)
        
        # 2. Resample Latent Space! Map Indicator spatial coords to Patch spatial coords
        E_aligned = align_embeddings(qt_in, E_in, qt_out, emb_dim=emb_dim, max_depth=MAX_LEVEL)
        
        # 3. Decode the Patch topology
        split_logits, val_pred = decoder(qt_out, E_aligned)

        # Calculate Loss against the Continuous Patch Tree
        L_val = torch.tensor(0.0, device=device)
        L_split = torch.tensor(0.0, device=device)
        n_val, n_split = 0, 0
        
        for d in range(MAX_LEVEL + 1):
            mask = qt_out.leaf_mask[d]
            if mask is not None and mask.any():
                L_val = L_val + mse(val_pred[d][mask], qt_out.values_gt[d][mask])
                n_val += 1
                
            if d < MAX_LEVEL and qt_out.split_gt[d] is not None and qt_out.split_gt[d].numel() > 0:
                L_split = L_split + bce(split_logits[d], qt_out.split_gt[d])
                n_split += 1
                
        if n_val > 0: L_val /= n_val
        if n_split > 0: L_split /= n_split
        
        L_total = L_val + 0.5 * L_split
        L_total.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:04d} | Total Loss: {L_total.item():.4f} | Val MSE: {L_val.item():.4f}")

if __name__ == "__main__":
    main()