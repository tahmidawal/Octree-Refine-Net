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
    """
    Handles 2D Morton Code (Z-Order Curve) encoding and decoding.
    Interleaves bits of X and Y.
    Works with Python ints or torch tensors (dtype long).
    """
    @staticmethod
    def _interleave_bits(x):
        # Ensure torch long
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
        # depth unused in this bit-interleaving version, kept for API parity
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
    """
    keys: (N,) morton keys at depth
    returns: (N, 3) -> [x_center, y_center, depth_norm]
    """
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
    """
    pos: (N, 3) in [0,1] for x,y and [0,1] for depth_norm
    returns: (N, 3 + 2*num_freqs*3)
    """
    if pos.numel() == 0:
        return pos
    freqs = (2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)).view(1, 1, -1)
    x = pos.unsqueeze(-1) * np.pi * 2.0 * freqs  # (N,3,F)
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (N,3,2F)
    enc = enc.view(pos.shape[0], -1)  # (N, 3*2F)
    return torch.cat([pos, enc], dim=1)


def fourier_encode_nd(x: torch.Tensor, num_freqs: int = 6):
    """
    Generic N-dimensional Fourier encoding.
    x: (N, D)
    returns: (N, D + 2*num_freqs*D)
    """
    if x.numel() == 0:
        return x
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)).view(1, 1, -1)  # (1,1,F)
    x_ = x.unsqueeze(-1) * (2.0 * np.pi) * freqs  # (N,D,F)
    enc = torch.cat([torch.sin(x_), torch.cos(x_)], dim=-1)  # (N,D,2F)
    enc = enc.reshape(x.shape[0], -1)  # (N, D*2F)
    return torch.cat([x, enc], dim=1)


def query_features_from_keys(keys: torch.Tensor, depth: int, max_depth: int,
                             qx: float, qy: float, qr: float, num_freqs: int = 4):
    """
    Per-node conditioning feature that tells the network where the 'refine patch' is.
    For each node, computes:
      - delta: (x - qx, y - qy) offset from patch center
      - dist: distance to patch center
      - r: patch radius
      - inside: 1 if node center is inside patch, 0 otherwise
      - dnorm: normalized depth
    Returns (N, q_dim) where q_dim = 6 + 2*num_freqs*6
    """
    if keys.numel() == 0:
        base_dim = 6
        q_dim = base_dim + 2 * num_freqs * base_dim
        return torch.zeros((0, q_dim), device=keys.device)

    ix, iy = Morton2D.key2xy(keys, depth=depth)
    res = float(1 << depth)
    x = (ix.float() + 0.5) / res
    y = (iy.float() + 0.5) / res

    q = torch.tensor([qx, qy], device=keys.device, dtype=x.dtype).view(1, 2)
    xy = torch.stack([x, y], dim=1)  # (N,2)
    delta = xy - q                   # (N,2)
    dist = torch.norm(delta, dim=1, keepdim=True)  # (N,1)

    r = torch.full_like(dist, float(qr))
    inside = (dist <= r).float()

    dnorm = torch.full_like(dist, float(depth) / float(max_depth))

    base = torch.cat([delta, dist, r, inside, dnorm], dim=1)  # (N, 2+1+1+1+1) = (N,6)
    return fourier_encode_nd(base, num_freqs=num_freqs)


# ==========================================
# 2. True Adaptive Quadtree Data Structure
# ==========================================

class Quadtree:
    """
    TRUE adaptive quadtree representation:
      - keys[d] sparse nodes per depth
      - children_idx[d]: (N_d,4) mapping to indices in keys[d+1], else -1
      - parent_idx[d]: (N_d,) mapping to index in keys[d-1]
      - neighs[d]: (N_d,9) same-depth neighbor indices, else -1
      - features_in[d]: (N_d,C) input values placed at leaf nodes, 0 elsewhere
      - split_gt[d], leaf_mask[d], values_gt[d]: targets for Direction B
    """
    def __init__(self, max_depth, device='cpu'):
        self.max_depth = max_depth
        self.device = device

        self.keys = [None] * (max_depth + 1)
        self.neighs = [None] * (max_depth + 1)
        self.features_in = [None] * (max_depth + 1)

        self.children_idx = [None] * (max_depth + 1)  # None at max_depth
        self.parent_idx   = [None] * (max_depth + 1)  # None at depth 0

        self.split_gt   = [None] * (max_depth + 1)    # None at max_depth
        self.leaf_mask  = [None] * (max_depth + 1)
        self.values_gt  = [None] * (max_depth + 1)

        # Cross-level neighbor info: for each node, store neighbor index and source depth
        self.cross_neigh_idx = [None] * (max_depth + 1)    # (N_d, 9) index into keys[source_depth]
        self.cross_neigh_depth = [None] * (max_depth + 1)  # (N_d, 9) which depth the neighbor is from

    def build_from_leaves(self, leaf_keys_by_depth, leaf_vals_input_by_depth, leaf_vals_target_by_depth=None):
        """
        leaf_keys_by_depth[d]: LongTensor (N_leaf_d,)
        leaf_vals_input_by_depth[d]: FloatTensor (N_leaf_d, C) - model input (can be noisy)
        leaf_vals_target_by_depth[d]: FloatTensor (N_leaf_d, C) - supervision target (clean)
                                      If None, uses input as target (no denoising)
        """
        if leaf_vals_target_by_depth is None:
            leaf_vals_target_by_depth = leaf_vals_input_by_depth

        # Infer C_in
        C_in = None
        for d in range(self.max_depth + 1):
            if leaf_vals_input_by_depth[d].numel() > 0:
                C_in = leaf_vals_input_by_depth[d].shape[1]
                break
        if C_in is None:
            C_in = 1

        # 1) Initialize keys/features with leaves (using INPUT values)
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

        # Ensure a root exists
        if self.keys[0].numel() == 0:
            self.keys[0] = torch.tensor([0], dtype=torch.long, device=self.device)
            self.features_in[0] = torch.zeros((1, C_in), dtype=torch.float, device=self.device)

        # 2) Internal closure: add all ancestors so parent keys always exist
        for d in range(self.max_depth, 0, -1):
            if self.keys[d].numel() == 0:
                continue
            parents = torch.unique(self.keys[d] >> 2, sorted=True)
            self.keys[d - 1] = torch.unique(torch.cat([self.keys[d - 1], parents]), sorted=True)

        # 3) Rebuild features_in[d] for expanded keys (keep leaf INPUT values, internal nodes = 0)
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

        # 4) Build children_idx for d=0..max_depth-1
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

        # 5) Build parent_idx for d=1..max_depth
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

        # 6) Build safe neighbors at each depth
        for d in range(self.max_depth + 1):
            self._construct_neigh_geometric_safe(d)

        # 6b) Build cross-level neighbors (fallback to ancestors when same-depth neighbor missing)
        for d in range(self.max_depth + 1):
            self._construct_cross_level_neighs(d)

        # 7) Build GT labels using TARGET values (clean for denoising AE)
        for d in range(self.max_depth + 1):
            kd = self.keys[d]
            if kd.numel() == 0:
                self.values_gt[d] = torch.zeros((0, C_in), device=self.device)
                self.leaf_mask[d] = torch.zeros((0,), dtype=torch.bool, device=self.device)
                self.split_gt[d] = None
                continue

            # Build values_gt from TARGET values (clean)
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
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)  # (N,9,2)

        res = 1 << depth
        nx = n_coords[..., 0]
        ny = n_coords[..., 1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)

        # build neighbor keys where valid
        n_keys = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
        if valid.any():
            n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)

        # search for valid keys
        idx = torch.searchsorted(keys, n_keys.clamp(min=0))
        idx = idx.clamp(0, len(keys) - 1)
        found = valid & (keys[idx] == n_keys)
        idx[~found] = -1
        self.neighs[depth] = idx

    def _construct_cross_level_neighs(self, depth):
        """
        For each node at `depth`, find neighbors. If a same-depth neighbor doesn't exist,
        walk up the tree to find the coarsest ancestor covering that spatial region.
        
        Stores:
          cross_neigh_idx[depth]: (N, 9) - index into keys[source_depth]
          cross_neigh_depth[depth]: (N, 9) - which depth the neighbor comes from
        """
        keys = self.keys[depth]
        N = len(keys)
        
        if N == 0:
            self.cross_neigh_idx[depth] = torch.empty((0, 9), dtype=torch.long, device=self.device)
            self.cross_neigh_depth[depth] = torch.empty((0, 9), dtype=torch.long, device=self.device)
            return
        
        if depth == 0:
            # Root has no real neighbors, just itself at center
            neigh_idx = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
            neigh_depth = torch.full((N, 9), 0, dtype=torch.long, device=self.device)
            neigh_idx[:, 4] = 0  # center is self
            self.cross_neigh_idx[depth] = neigh_idx
            self.cross_neigh_depth[depth] = neigh_depth
            return
        
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor(
            [[-1,-1],[-1,0],[-1,1],
             [ 0,-1],[ 0,0],[ 0,1],
             [ 1,-1],[ 1,0],[ 1,1]],
            device=self.device,
            dtype=torch.long
        )
        
        # Output tensors
        neigh_idx = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
        neigh_depth = torch.full((N, 9), depth, dtype=torch.long, device=self.device)
        
        for k in range(9):
            dx, dy = offsets[k]
            nx = x + dx
            ny = y + dy
            
            res = 1 << depth
            valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
            
            if not valid.any():
                continue
            
            valid_indices = torch.where(valid)[0]
            cur_nx = nx[valid].clone()
            cur_ny = ny[valid].clone()
            
            # Track which nodes still need a neighbor found
            remaining = torch.ones(len(valid_indices), dtype=torch.bool, device=self.device)
            
            # Walk up from depth to 0
            for d in range(depth, -1, -1):
                if not remaining.any():
                    break
                
                keys_d = self.keys[d]
                if keys_d.numel() == 0:
                    # No nodes at this depth, go coarser
                    cur_nx = cur_nx >> 1
                    cur_ny = cur_ny >> 1
                    continue
                
                # Compute Morton keys for remaining positions at depth d
                r_mask = remaining
                target_keys = Morton2D.xy2key(cur_nx[r_mask], cur_ny[r_mask], depth=d)
                
                # Search in sorted keys
                idx = torch.searchsorted(keys_d, target_keys)
                idx = idx.clamp(0, len(keys_d) - 1)
                found = (keys_d[idx] == target_keys)
                
                if found.any():
                    # Map back to original valid_indices
                    r_indices = torch.where(r_mask)[0][found]
                    orig_indices = valid_indices[r_indices]
                    found_idx = idx[found]
                    
                    neigh_idx[orig_indices, k] = found_idx
                    neigh_depth[orig_indices, k] = d
                    remaining[r_indices] = False
                
                # For those still not found, go to parent level
                cur_nx = cur_nx >> 1
                cur_ny = cur_ny >> 1
        
        self.cross_neigh_idx[depth] = neigh_idx
        self.cross_neigh_depth[depth] = neigh_depth


# ==========================================
# 3. Sparse QuadConv Layers (same idea)
# ==========================================

class QuadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.weights = nn.Linear(9 * in_channels, out_channels)

    def forward(self, features, quadtree, depth):
        neigh_idx = quadtree.neighs[depth]  # (N,9)
        N, K = neigh_idx.shape
        if N == 0:
            return torch.zeros((0, self.weights.out_features), device=features.device)

        pad_vec = torch.zeros((1, features.shape[1]), device=features.device)
        feat_padded = torch.cat([features, pad_vec], dim=0)

        gather_idx = neigh_idx.clone()
        gather_idx[gather_idx == -1] = N  # map missing to pad
        col = feat_padded[gather_idx]     # (N,9,C)
        col_flat = col.view(N, -1)
        return self.weights(col_flat)


class CrossLevelQuadConv(nn.Module):
    """
    Cross-level neighbor gathering convolution.
    When a same-depth neighbor doesn't exist, uses the coarser ancestor covering that region.
    Includes depth_delta channel so the network knows the resolution difference.
    """
    def __init__(self, in_channels, out_channels, max_depth, kernel_size=3):
        super().__init__()
        self.max_depth = max_depth
        # +1 per neighbor for normalized depth delta
        self.weights = nn.Linear(9 * (in_channels + 1), out_channels)
    
    def forward(self, all_features, quadtree, depth):
        """
        all_features: list of (N_d, C) tensors for each depth
        quadtree: the Quadtree object with cross_neigh_idx and cross_neigh_depth
        depth: which depth we're convolving at
        """
        neigh_idx = quadtree.cross_neigh_idx[depth]    # (N, 9)
        neigh_depth = quadtree.cross_neigh_depth[depth]  # (N, 9)
        
        if neigh_idx is None or neigh_idx.numel() == 0:
            return torch.zeros((0, self.weights.out_features), device=quadtree.device)
        
        N = neigh_idx.shape[0]
        C = all_features[depth].shape[1] if all_features[depth].numel() > 0 else 0
        
        if N == 0 or C == 0:
            return torch.zeros((0, self.weights.out_features), device=quadtree.device)
        
        # Gather features from different depths
        # Output: (N, 9, C+1) where last channel is normalized depth delta
        gathered = torch.zeros((N, 9, C + 1), device=quadtree.device)
        
        for k in range(9):
            idx_k = neigh_idx[:, k]      # (N,)
            depth_k = neigh_depth[:, k]  # (N,)
            
            # For each unique source depth, gather features
            for src_d in range(self.max_depth + 1):
                mask = (depth_k == src_d) & (idx_k != -1)
                if not mask.any():
                    continue
                
                src_feat = all_features[src_d]
                if src_feat is None or src_feat.numel() == 0:
                    continue
                
                # Gather features from source depth
                src_idx = idx_k[mask]
                # Clamp to valid range
                src_idx = src_idx.clamp(0, src_feat.shape[0] - 1)
                gathered[mask, k, :C] = src_feat[src_idx]
                
                # Normalized depth delta: how many levels up we went / max_depth
                delta = float(depth - src_d) / float(max(self.max_depth, 1))
                gathered[mask, k, C] = delta
        
        gathered_flat = gathered.view(N, -1)  # (N, 9*(C+1))
        return self.weights(gathered_flat)


class QuadPool(nn.Module):
    """
    Pools from depth d (children) -> depth d-1 (parents) using children_idx[d-1].
    Mean pool across existing children.
    """
    def forward(self, child_features, quadtree, depth_child):
        assert depth_child >= 1
        d_parent = depth_child - 1
        parent_keys = quadtree.keys[d_parent]
        Np = len(parent_keys)
        C = child_features.shape[1]

        if Np == 0:
            return torch.zeros((0, C), device=child_features.device)

        ch = quadtree.children_idx[d_parent]  # (Np,4)
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
    """
    Unpools from depth d-1 (parents) -> depth d (children) using parent_idx[d].
    """
    def forward(self, parent_features, quadtree, depth_child):
        assert depth_child >= 1
        pidx = quadtree.parent_idx[depth_child]
        if pidx.numel() == 0:
            return torch.zeros((0, parent_features.shape[1]), device=parent_features.device)
        return parent_features[pidx]


# ==========================================
# 4. Direction-B Autoencoder (Teacher Forced)
# ==========================================

class TreeEncoder(nn.Module):
    """
    Bottom-up encoder with positional encoding, PATCH CONDITIONING, and CROSS-LEVEL neighbor gathering:
      - projects input features + Fourier position encoding + query features at each depth to hidden
      - repeatedly pools depth d -> d-1 and mixes via cross-level neighbor conv
      - cross-level conv falls back to coarser ancestors when same-depth neighbors missing
    Returns ALL depth features for skip connections.
    """
    def __init__(self, in_c=1, hidden=64, max_depth=8, pos_freqs=6, query_freqs=4):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.query_freqs = query_freqs
        self.max_depth = max_depth
        self.hidden = hidden
        # pos dim = 3 + 2*pos_freqs*3
        pos_dim = 3 + 2 * pos_freqs * 3
        # query feature dim: base has 6 dims, Fourier adds 2*query_freqs*6
        self.q_dim = 6 + 2 * query_freqs * 6
        self.in_proj = nn.Linear(in_c + pos_dim + self.q_dim, hidden)
        # Use CrossLevelQuadConv for better boundary handling
        self.convs = nn.ModuleList([
            CrossLevelQuadConv(hidden, hidden, max_depth) for _ in range(max_depth + 1)
        ])
        self.pool = QuadPool()

    def forward(self, qt: Quadtree, patch=None):
        h = [None] * (self.max_depth + 1)

        for d in range(self.max_depth + 1):
            fin = qt.features_in[d]
            kd = qt.keys[d]
            if fin is None or fin.numel() == 0:
                h[d] = torch.zeros((0, self.hidden), device=qt.device)
                continue

            pos = node_centers_from_keys(kd, d, self.max_depth, device=qt.device)
            pos = fourier_encode(pos, num_freqs=self.pos_freqs)

            # Query/patch conditioning features
            if patch is None:
                qfeat = torch.zeros((fin.shape[0], self.q_dim), device=qt.device)
            else:
                qx, qy, qr = patch
                qfeat = query_features_from_keys(kd, d, self.max_depth, qx, qy, qr, num_freqs=self.query_freqs)

            x = torch.cat([fin, pos, qfeat], dim=1)
            h[d] = self.in_proj(x)

        # Bottom-up pooling
        for d in range(self.max_depth, 0, -1):
            if h[d].numel() == 0:
                continue

            pooled = self.pool(h[d], qt, d)  # -> depth d-1
            if h[d - 1].shape[0] != pooled.shape[0]:
                raise RuntimeError("Pool produced mismatched parent count; check tree closure/connectivity.")

            h[d - 1] = h[d - 1] + pooled

            # Mix via CROSS-LEVEL neighbor conv (skip depth 0)
            if d - 1 >= 1 and h[d - 1].numel() > 0:
                h[d - 1] = F.relu(self.convs[d - 1](h, qt, d - 1))

        return h  # Return ALL depths for skip connections


class TreeDecoderTeacherForced(nn.Module):
    """
    Teacher-forced top-down decoder with skip connections, positional encoding,
    PATCH CONDITIONING, and CROSS-LEVEL neighbor gathering:
      - fuses decoder state with encoder skip features, position encoding, and query features
      - applies cross-level neighborhood mixing at each depth
      - predicts split logits at depths < max_depth
      - predicts child hidden features for nodes that split in GT (teacher forcing)
      - predicts values for all nodes (supervise only GT leaves)
    """
    def __init__(self, hidden=64, out_c=1, max_depth=8, pos_freqs=6, query_freqs=4):
        super().__init__()
        self.max_depth = max_depth
        self.pos_freqs = pos_freqs
        self.query_freqs = query_freqs
        self.hidden = hidden
        pos_dim = 3 + 2 * pos_freqs * 3
        # query feature dim: base has 6 dims, Fourier adds 2*query_freqs*6
        self.q_dim = 6 + 2 * query_freqs * 6

        # Fuse decoder h with encoder skip, position, and query features
        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden + pos_dim + self.q_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.split_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.child_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4 * hidden))
        self.val_head   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_c))

        # Cross-level neighborhood mixing at each depth after expansion
        self.mix_convs = nn.ModuleList([
            CrossLevelQuadConv(hidden, hidden, max_depth) for _ in range(max_depth + 1)
        ])

    def forward(self, qt: Quadtree, enc_h_list, patch=None):
        # enc_h_list[d] is encoder hidden at depth d
        h_by_depth = [None] * (self.max_depth + 1)
        h_by_depth[0] = enc_h_list[0]  # Start from encoder root

        split_logits = [None] * self.max_depth        # 0..max_depth-1
        val_pred     = [None] * (self.max_depth + 1)  # 0..max_depth

        for d in range(self.max_depth + 1):
            h = h_by_depth[d]
            if h is None or h.numel() == 0:
                val_pred[d] = torch.zeros((0, 1), device=qt.device)
                if d < self.max_depth:
                    split_logits[d] = torch.zeros((0,), device=qt.device)
                continue

            # Fuse with encoder skip + positional encoding + query features
            kd = qt.keys[d]
            pos = node_centers_from_keys(kd, d, self.max_depth, device=qt.device)
            pos = fourier_encode(pos, num_freqs=self.pos_freqs)

            # Query/patch conditioning features
            if patch is None:
                qfeat = torch.zeros((h.shape[0], self.q_dim), device=qt.device)
            else:
                qx, qy, qr = patch
                qfeat = query_features_from_keys(kd, d, self.max_depth, qx, qy, qr, num_freqs=self.query_freqs)

            skip = enc_h_list[d] if (enc_h_list[d] is not None and enc_h_list[d].shape[0] == h.shape[0]) else torch.zeros_like(h)
            h = self.fuse(torch.cat([h, skip, pos, qfeat], dim=1))
            h_by_depth[d] = h  # Update for cross-level conv access

            # Cross-level neighborhood mixing (helps a lot for spatial coherence)
            if h.numel() > 0 and d >= 1:
                h = F.relu(self.mix_convs[d](h_by_depth, qt, d))
                h_by_depth[d] = h  # Update again after conv

            val_pred[d] = self.val_head(h)

            if d == self.max_depth:
                break

            split_logits[d] = self.split_head(h).squeeze(-1)

            # Teacher forcing expansion
            ch = qt.children_idx[d]  # (N_d,4)
            has_child = (ch != -1).any(dim=1)

            N_next = len(qt.keys[d + 1])
            h_next = torch.zeros((N_next, h.shape[1]), device=h.device)

            if has_child.any():
                child_feats = self.child_head(h[has_child]).view(-1, 4, h.shape[1])
                parent_rows = torch.nonzero(has_child).squeeze(-1)

                for t, p in enumerate(parent_rows):
                    for c in range(4):
                        ci = int(ch[p, c].item())
                        if ci != -1:
                            h_next[ci] = child_feats[t, c]

            h_by_depth[d + 1] = h_next

        return split_logits, val_pred


class TreeAE_DirectionB(nn.Module):
    def __init__(self, in_c=1, hidden=64, max_depth=8, pos_freqs=6, query_freqs=4):
        super().__init__()
        self.encoder = TreeEncoder(in_c=in_c, hidden=hidden, max_depth=max_depth, pos_freqs=pos_freqs, query_freqs=query_freqs)
        self.decoder = TreeDecoderTeacherForced(hidden=hidden, out_c=in_c, max_depth=max_depth, pos_freqs=pos_freqs, query_freqs=query_freqs)

    def forward(self, qt: Quadtree, patch=None):
        enc_h_list = self.encoder(qt, patch=patch)  # List of h[d] for all depths
        split_logits, val_pred = self.decoder(qt, enc_h_list, patch=patch)
        return split_logits, val_pred


# ==========================================
# 5. Generalized Data Logic (random K1,K2, denoising AE)
# ==========================================

MAX_LEVEL = 7
MIN_LEVEL = 6
NOISE_STD = 0.05  # Noise for denoising AE

def target_function(x, y, k1, k2):
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)

def gradient_magnitude(x, y, k1, k2):
    """
    Compute gradient magnitude of sin(2πk1*x)*sin(2πk2*y).
    df/dx = 2πk1 * cos(2πk1*x) * sin(2πk2*y)
    df/dy = 2πk2 * sin(2πk1*x) * cos(2πk2*y)
    |∇f| = sqrt((df/dx)^2 + (df/dy)^2)
    """
    dfdx = 2 * np.pi * k1 * np.cos(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)
    dfdy = 2 * np.pi * k2 * np.sin(2 * np.pi * k1 * x) * np.cos(2 * np.pi * k2 * y)
    return np.sqrt(dfdx**2 + dfdy**2)

class QuadNode:
    def __init__(self, x, y, size, level, max_level, min_level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.max_level = max_level
        self.min_level = min_level
        self.children, self.val = [], None

    def _intersects_circle(self, cx, cy, r):
        """
        Cell-circle intersection test (robust).
        Cell is [x, x+size] x [y, y+size].
        """
        x0, y0 = self.x, self.y
        x1, y1 = self.x + self.size, self.y + self.size

        # closest point in rectangle to circle center
        px = min(max(cx, x0), x1)
        py = min(max(cy, y0), y1)

        dx = px - cx
        dy = py - cy
        return (dx*dx + dy*dy) <= r*r

    def patch_subdivide(self, patch_cx, patch_cy, patch_r, patch_depth=None):
        """
        Split policy:
          - force split until min_level (coarse base grid)
          - inside patch: force split until patch_depth (or max_level if None)
          - outside patch: stop after min_level
        """
        if self.level >= self.max_level:
            return

        # Always build a baseline grid first
        if self.level < self.min_level:
            do_split = True
        else:
            inside = self._intersects_circle(patch_cx, patch_cy, patch_r)
            target_depth = self.max_level if patch_depth is None else patch_depth
            do_split = inside and (self.level < target_depth)

        if do_split:
            half = self.size / 2
            self.children = [
                QuadNode(self.x,        self.y,        half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y,        half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x,        self.y + half, half, self.level + 1, self.max_level, self.min_level),
                QuadNode(self.x + half, self.y + half, half, self.level + 1, self.max_level, self.min_level),
            ]
            for child in self.children:
                child.patch_subdivide(patch_cx, patch_cy, patch_r, patch_depth=patch_depth)

    def collect_leaves(self, leaves_list, k1, k2):
        if not self.children:
            center_x = self.x + self.size / 2
            center_y = self.y + self.size / 2
            self.val = target_function(center_x, center_y, k1, k2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves(leaves_list, k1, k2)

def generate_user_data(noise_std=NOISE_STD, patch_r_range=(0.12, 0.25), patch_depth=None):
    """
    PATCH-BASED data generation:
      - Random K1, K2 each call (1-5)
      - Random patch center (qx, qy) and radius qr
      - Tree refined only inside that patch (to patch_depth or max_level)
      - Returns noisy inputs + clean targets for denoising AE
      - Returns patch=(qx, qy, qr) so the network can learn splits
    """
    # Randomize frequencies each call
    k1 = random.randint(1, 5)
    k2 = random.randint(1, 5)

    # Random patch location and size
    qx = random.random()
    qy = random.random()
    qr = random.uniform(patch_r_range[0], patch_r_range[1])

    root = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root.patch_subdivide(qx, qy, qr, patch_depth=patch_depth)

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

    patch = (qx, qy, qr)
    return leaf_keys_by_depth, leaf_vals_noisy_by_depth, leaf_vals_clean_by_depth, leaves, k1, k2, patch


def generate_user_data_fixed_patch(qx, qy, qr, noise_std=NOISE_STD, patch_depth=None):
    """
    Generate data with a FIXED patch location (user-specified coordinate).
    Useful for inference when user wants to refine around a specific point.
    """
    k1 = random.randint(1, 5)
    k2 = random.randint(1, 5)

    root = QuadNode(0, 0, 1.0, 0, MAX_LEVEL, MIN_LEVEL)
    root.patch_subdivide(qx, qy, qr, patch_depth=patch_depth)

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

    patch = (qx, qy, qr)
    return leaf_keys_by_depth, leaf_vals_noisy_by_depth, leaf_vals_clean_by_depth, leaves, k1, k2, patch


# ==========================================
# 6. Main Execution (Generalized Denoising AE Training)
# ==========================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # Model (Denoising AE) - with positional encoding, patch conditioning, and skip connections
    model = TreeAE_DirectionB(in_c=1, hidden=128, max_depth=MAX_LEVEL, pos_freqs=6, query_freqs=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    mse = nn.MSELoss()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"plots/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # Training with PATCH-BASED splitting and SPLIT LOSS
    num_steps = 3000
    split_weight = 0.5  # Weight for split loss
    print(f"Training Denoising AE across random trees for {num_steps} steps...")
    print(f"  - Random K1,K2 in [1,5] each step")
    print(f"  - Noise std: {NOISE_STD}")
    print(f"  - PATCH-BASED splitting (random patch location)")
    print(f"  - Split loss: ENABLED (weight={split_weight})")
    print(f"  - Query conditioning: ENABLED (patch info fed to network)")
    
    val_losses = []
    split_losses = []
    total_losses = []

    # Store a few samples for plotting
    sample_trees = []

    bce = nn.BCEWithLogitsLoss()

    for step in range(num_steps):
        # Generate a NEW random tree each step with PATCH-BASED splitting
        leaf_keys_by_depth, leaf_vals_noisy, leaf_vals_clean, leaves, k1, k2, patch = generate_user_data()

        # Build quadtree with noisy input, clean target
        qt = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt.build_from_leaves(leaf_keys_by_depth, leaf_vals_noisy, leaf_vals_clean)

        optimizer.zero_grad()
        split_logits, val_pred = model(qt, patch=patch)

        # Value loss: supervise GT leaves with CLEAN values
        L_val = torch.tensor(0.0, device=device)
        n_val_terms = 0
        for d in range(MAX_LEVEL + 1):
            mask = qt.leaf_mask[d]
            if mask is None or mask.numel() == 0:
                continue
            if mask.any():
                L_val = L_val + mse(val_pred[d][mask], qt.values_gt[d][mask])
                n_val_terms += 1
        if n_val_terms > 0:
            L_val = L_val / n_val_terms

        # Split loss: supervise split decisions at each depth
        L_split = torch.tensor(0.0, device=device)
        n_split_terms = 0
        for d in range(MAX_LEVEL):  # No split at max_depth
            if qt.split_gt[d] is None or qt.split_gt[d].numel() == 0:
                continue
            if split_logits[d] is None or split_logits[d].numel() == 0:
                continue
            L_split = L_split + bce(split_logits[d], qt.split_gt[d])
            n_split_terms += 1
        if n_split_terms > 0:
            L_split = L_split / n_split_terms

        # Total loss
        L_total = L_val + split_weight * L_split

        L_total.backward()
        optimizer.step()

        val_losses.append(float(L_val.item()))
        split_losses.append(float(L_split.item()))
        total_losses.append(float(L_total.item()))

        # Print loss every step
        qx, qy, qr = patch
        print(f"Step {step}: val={L_val.item():.6f} split={L_split.item():.6f} total={L_total.item():.6f}  k1={k1} k2={k2}  patch=({qx:.2f},{qy:.2f},r={qr:.2f})  leaves={len(leaves)}")

        # Store sample for plotting
        if step % 100 == 0:
            sample_trees.append({
                'qt': qt,
                'leaves': leaves,
                'k1': k1,
                'k2': k2,
                'patch': patch,
                'step': step
            })
            # Keep only last 3 samples
            if len(sample_trees) > 3:
                sample_trees.pop(0)

        # Plot every 100 steps with multiple samples
        if step % 100 == 0 or step == num_steps - 1:
            # Get predictions and per-leaf losses for all stored samples
            with torch.no_grad():
                for sample in sample_trees:
                    _, val_pred_viz = model(sample['qt'], patch=sample['patch'])
                    sample['pred_dict'] = {}
                    sample['loss_dict'] = {}
                    sample['pred_by_depth'] = {d: [] for d in range(MAX_LEVEL + 1)}
                    sample['gt_by_depth'] = {d: [] for d in range(MAX_LEVEL + 1)}
                    
                    for d in range(MAX_LEVEL + 1):
                        if sample['qt'].keys[d].numel() == 0:
                            continue
                        mask = sample['qt'].leaf_mask[d]
                        if mask is None or mask.numel() == 0 or not mask.any():
                            continue
                        kd = sample['qt'].keys[d][mask].cpu()
                        ix, iy = Morton2D.key2xy(kd, depth=d)
                        pr = val_pred_viz[d][mask].cpu().numpy().flatten()
                        gt = sample['qt'].values_gt[d][mask].cpu().numpy().flatten()
                        
                        for i in range(len(kd)):
                            key = (d, int(ix[i].item()), int(iy[i].item()))
                            sample['pred_dict'][key] = pr[i]
                            leaf_loss = (pr[i] - gt[i]) ** 2  # MSE per leaf
                            sample['loss_dict'][key] = leaf_loss
                            sample['pred_by_depth'][d].append(pr[i])
                            sample['gt_by_depth'][d].append(gt[i])

            # Create plot with multiple samples: GT | Pred | Loss | Depth Histogram
            n_samples = len(sample_trees)
            if n_samples > 0:
                fig, axes = plt.subplots(n_samples, 4, figsize=(24, 6*n_samples))
                if n_samples == 1:
                    axes = axes.reshape(1, -1)
                
                cmap = plt.get_cmap('viridis')
                loss_cmap = plt.get_cmap('hot')

                for row, sample in enumerate(sample_trees):
                    # Compute max loss for normalization
                    all_losses = list(sample['loss_dict'].values())
                    max_loss = max(all_losses) if all_losses else 1.0
                    max_loss = max(max_loss, 1e-6)  # Avoid division by zero

                    # Plot 1: GT Quadtree with rectangles (clean values)
                    ax1 = axes[row, 0]
                    ax1.set_xlim(0, 1)
                    ax1.set_ylim(0, 1)
                    ax1.set_aspect('equal')
                    ax1.axis('off')
                    px, py, pr = sample['patch']
                    ax1.set_title(f"GT Step {sample['step']}: k1={sample['k1']}, k2={sample['k2']}, patch=({px:.2f},{py:.2f})", fontsize=10)
                    
                    # Draw patch circle
                    patch_circle = plt.Circle((px, py), pr, fill=False, edgecolor='red', linewidth=2, linestyle='--')
                    ax1.add_patch(patch_circle)

                    for node in sample['leaves']:
                        normalized_color = (node.val + 1) / 2
                        color = cmap(normalized_color)
                        rect = patches.Rectangle(
                            (node.x, node.y), node.size, node.size,
                            linewidth=0.5, edgecolor='black', facecolor=color
                        )
                        ax1.add_patch(rect)

                    # Plot 2: Predicted Quadtree with rectangles
                    ax2 = axes[row, 1]
                    ax2.set_xlim(0, 1)
                    ax2.set_ylim(0, 1)
                    ax2.set_aspect('equal')
                    ax2.axis('off')
                    ax2.set_title(f"Predicted (Denoised) Step {sample['step']}", fontsize=10)
                    
                    # Draw patch circle on predicted plot too
                    patch_circle2 = plt.Circle((px, py), pr, fill=False, edgecolor='red', linewidth=2, linestyle='--')
                    ax2.add_patch(patch_circle2)

                    for node in sample['leaves']:
                        d = node.level
                        res = 1 << d
                        ix = int(node.x * res)
                        iy = int(node.y * res)
                        ix = max(0, min(res - 1, ix))
                        iy = max(0, min(res - 1, iy))

                        pred_val = sample['pred_dict'].get((d, ix, iy), 0.0)
                        normalized_color = (pred_val + 1) / 2
                        normalized_color = np.clip(normalized_color, 0, 1)
                        color = cmap(normalized_color)
                        rect = patches.Rectangle(
                            (node.x, node.y), node.size, node.size,
                            linewidth=0.5, edgecolor='black', facecolor=color
                        )
                        ax2.add_patch(rect)

                    # Plot 3: Per-leaf loss heatmap
                    ax3 = axes[row, 2]
                    ax3.set_xlim(0, 1)
                    ax3.set_ylim(0, 1)
                    ax3.set_aspect('equal')
                    ax3.axis('off')
                    ax3.set_title(f"Per-Leaf Loss (MSE) Step {sample['step']}", fontsize=10)

                    for node in sample['leaves']:
                        d = node.level
                        res = 1 << d
                        ix = int(node.x * res)
                        iy = int(node.y * res)
                        ix = max(0, min(res - 1, ix))
                        iy = max(0, min(res - 1, iy))

                        leaf_loss = sample['loss_dict'].get((d, ix, iy), 0.0)
                        normalized_loss = np.clip(leaf_loss / max_loss, 0, 1)
                        color = loss_cmap(normalized_loss)
                        rect = patches.Rectangle(
                            (node.x, node.y), node.size, node.size,
                            linewidth=0.5, edgecolor='black', facecolor=color
                        )
                        ax3.add_patch(rect)

                    # Add colorbar for loss
                    sm = plt.cm.ScalarMappable(cmap=loss_cmap, norm=plt.Normalize(0, max_loss))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
                    cbar.set_label('MSE Loss', fontsize=8)

                    # Plot 4: Histogram of RELATIVE L2 loss per depth
                    # Relative L2 = ||pred - gt||_2 / ||gt||_2
                    ax4 = axes[row, 3]
                    depths = []
                    rel_l2_losses = []
                    for d in range(MAX_LEVEL + 1):
                        if sample['pred_by_depth'][d] and sample['gt_by_depth'][d]:
                            pr_arr = np.array(sample['pred_by_depth'][d])
                            gt_arr = np.array(sample['gt_by_depth'][d])
                            l2_error = np.linalg.norm(pr_arr - gt_arr)
                            l2_gt = np.linalg.norm(gt_arr)
                            rel_l2 = l2_error / max(l2_gt, 1e-8)  # Avoid division by zero
                            depths.append(d)
                            rel_l2_losses.append(rel_l2)
                    
                    if depths:
                        bars = ax4.bar(depths, rel_l2_losses, color='steelblue', edgecolor='black')
                        ax4.set_xlabel('Depth Level', fontsize=9)
                        ax4.set_ylabel('Relative L2 Loss', fontsize=9)
                        ax4.set_title(f"Relative L2 by Depth (Step {sample['step']})", fontsize=10)
                        ax4.set_xticks(range(MAX_LEVEL + 1))
                        # Add value labels on bars
                        for bar, val in zip(bars, rel_l2_losses):
                            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)

                plt.tight_layout()
                plt.savefig(f'{output_dir}/samples_step{step:04d}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  -> Saved {output_dir}/samples_step{step:04d}.png ({n_samples} samples)")

            # Save loss curves separately (all three losses)
            fig_loss, axes_loss = plt.subplots(1, 3, figsize=(18, 5))
            
            # Value loss
            ax1 = axes_loss[0]
            ax1.plot(val_losses, 'b-', linewidth=1, alpha=0.7)
            ax1.set_xlabel('Step', fontsize=11)
            ax1.set_ylabel('Value Loss (MSE)', fontsize=11)
            ax1.set_title('Value Loss', fontsize=12)
            ax1.grid(True, alpha=0.3)
            if len(val_losses) > 100:
                window = min(50, len(val_losses) // 10)
                smoothed = np.convolve(val_losses, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(val_losses)), smoothed, 'b-', linewidth=2, label=f'Smoothed')
                ax1.legend()
            
            # Split loss
            ax2 = axes_loss[1]
            ax2.plot(split_losses, 'g-', linewidth=1, alpha=0.7)
            ax2.set_xlabel('Step', fontsize=11)
            ax2.set_ylabel('Split Loss (BCE)', fontsize=11)
            ax2.set_title('Split Loss', fontsize=12)
            ax2.grid(True, alpha=0.3)
            if len(split_losses) > 100:
                window = min(50, len(split_losses) // 10)
                smoothed = np.convolve(split_losses, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(split_losses)), smoothed, 'g-', linewidth=2, label=f'Smoothed')
                ax2.legend()
            
            # Total loss
            ax3 = axes_loss[2]
            ax3.plot(total_losses, 'r-', linewidth=1, alpha=0.7)
            ax3.set_xlabel('Step', fontsize=11)
            ax3.set_ylabel('Total Loss', fontsize=11)
            ax3.set_title(f'Total Loss (val + {split_weight}*split)', fontsize=12)
            ax3.grid(True, alpha=0.3)
            if len(total_losses) > 100:
                window = min(50, len(total_losses) // 10)
                smoothed = np.convolve(total_losses, np.ones(window)/window, mode='valid')
                ax3.plot(range(window-1, len(total_losses)), smoothed, 'r-', linewidth=2, label=f'Smoothed')
                ax3.legend()
            
            plt.suptitle(f'Training Losses (Step {step})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/loss_curve_step{step:04d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig_loss)
            print(f"  -> Saved {output_dir}/loss_curve_step{step:04d}.png")

    print(f"Training complete. Processed {num_steps} random trees.")


if __name__ == "__main__":
    main()
