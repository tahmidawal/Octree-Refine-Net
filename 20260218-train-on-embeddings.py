"""
train_operator.py

Trains a multi-scale cross-attention operator that maps
  E_f (embeddings of f) --> E_u (embeddings of u)

then chains:
  encoder_f  -->  operator  -->  decoder_u

to solve the Poisson problem fully in embedding space.

Architecture: DepthwiseCrossAttentionOperator
  - For each depth d, computes:
      context = cat(E_f[d-1], E_f[d], E_f[d+1])   (multi-scale context)
      E_u_pred[d] = cross_attention(Q=E_f[d], K=context, V=context)
                  + depth-specific MLP refinement
  - Since f and u share topology, every E_f[d] token has a 1:1
    correspondence with E_u[d] — no spatial matching needed.
  - Cross-attention is done within each depth (each node attends to
    its 9 same-depth neighbors in f-space, plus its parent and children)
    using the precomputed neighbor/parent/children indices from the saved keys.

Loss:
  L = MSE(E_u_pred, E_u_target)
    + lambda_dec * MSE(decoder_u(E_u_pred), u_values_gt)  [end-to-end]

End-to-end inference pipeline:
  qt_f  -->  encoder_f  -->  E_f  -->  operator  -->  E_u_pred
        -->  decoder_u(qt_f_topology, E_u_pred)  -->  u_values

Usage:
    python train_operator.py \
        --emb_dir  embeddings/poisson_XYZ \
        --model_f  plots/poisson_XYZ/best_model_f.pt \
        --model_u  plots/poisson_XYZ/best_model_u.pt \
        --out_dir  operator/poisson_XYZ
"""

import argparse
import os
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ==========================================
# Morton / Positional (same as before)
# ==========================================

class Morton2D:
    @staticmethod
    def _interleave_bits(x):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.long)
        x = x.long() & 0x0000FFFF
        x = (x | (x << 8)) & 0x00FF00FF; x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333; x = (x | (x << 1)) & 0x55555555
        return x

    @staticmethod
    def _deinterleave_bits(x):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.long)
        x = x.long() & 0x55555555
        x = (x | (x >> 1)) & 0x33333333; x = (x | (x >> 2)) & 0x0F0F0F0F
        x = (x | (x >> 4)) & 0x00FF00FF; x = (x | (x >> 8)) & 0x0000FFFF
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
# Quadtree (same as before, needed for decoder)
# ==========================================

class Quadtree:
    def __init__(self, max_depth, device='cpu'):
        self.max_depth = max_depth; self.device = device
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
                feat.index_add_(0, inv, lv); cnt.index_add_(0, inv, torch.ones((len(inv),1), device=self.device))
                self.features_in[d] = feat / cnt.clamp(min=1)

        if self.keys[0].numel() == 0:
            self.keys[0] = torch.tensor([0], dtype=torch.long, device=self.device)
            self.features_in[0] = torch.zeros((1, C_in), device=self.device)

        for d in range(self.max_depth, 0, -1):
            if self.keys[d].numel() == 0: continue
            parents = torch.unique(self.keys[d] >> 2, sorted=True)
            self.keys[d-1] = torch.unique(torch.cat([self.keys[d-1], parents]), sorted=True)

        for d in range(self.max_depth + 1):
            kd = self.keys[d]; feat = torch.zeros((len(kd), C_in), device=self.device)
            lk = leaf_keys_by_depth[d].to(self.device).long()
            lv = leaf_vals_input_by_depth[d].to(self.device).float()
            if lk.numel() > 0:
                lk_u, inv = torch.unique(lk, sorted=True, return_inverse=True)
                pooled = torch.zeros((len(lk_u), C_in), device=self.device)
                cnt = torch.zeros((len(lk_u), 1), device=self.device)
                pooled.index_add_(0, inv, lv); cnt.index_add_(0, inv, torch.ones((len(inv),1), device=self.device))
                pooled = pooled / cnt.clamp(min=1)
                idx = torch.searchsorted(kd, lk_u).clamp(0, len(kd)-1)
                found = kd[idx] == lk_u; feat[idx[found]] = pooled[found]
            self.features_in[d] = feat

        for d in range(self.max_depth):
            kd = self.keys[d]; kn = self.keys[d+1]
            if kd.numel() == 0: self.children_idx[d] = torch.empty((0,4), dtype=torch.long, device=self.device); continue
            if kn.numel() == 0: self.children_idx[d] = torch.full((len(kd),4),-1, dtype=torch.long, device=self.device); continue
            ck = (kd.unsqueeze(1) << 2) + torch.arange(4, device=self.device).view(1,4)
            idx = torch.searchsorted(kn, ck).clamp(0, len(kn)-1); found = kn[idx] == ck
            self.children_idx[d] = torch.where(found, idx, torch.full_like(idx, -1))
        self.children_idx[self.max_depth] = None

        self.parent_idx[0] = None
        for d in range(1, self.max_depth + 1):
            kd = self.keys[d]; kp = self.keys[d-1]
            if kd.numel() == 0: self.parent_idx[d] = torch.empty((0,), dtype=torch.long, device=self.device); continue
            self.parent_idx[d] = torch.searchsorted(kp, kd >> 2).clamp(0, len(kp)-1)

        for d in range(self.max_depth + 1): self._build_neigh(d)

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
                pooled.index_add_(0, inv, lv_t); cnt.index_add_(0, inv, torch.ones((len(inv),1), device=self.device))
                pooled = pooled / cnt.clamp(min=1)
                idx = torch.searchsorted(kd, lk_u).clamp(0, len(kd)-1); found = kd[idx] == lk_u
                target_feat[idx[found]] = pooled[found]
            self.values_gt[d] = target_feat
            if d == self.max_depth:
                self.leaf_mask[d] = torch.ones((len(kd),), dtype=torch.bool, device=self.device); self.split_gt[d] = None
            else:
                ch = self.children_idx[d]; is_leaf = (ch == -1).all(dim=1)
                self.leaf_mask[d] = is_leaf; self.split_gt[d] = (~is_leaf).float()

    def _build_neigh(self, depth):
        keys = self.keys[depth]; N = len(keys)
        if N == 0: self.neighs[depth] = torch.empty((0,9), dtype=torch.long, device=self.device); return
        if depth == 0:
            neigh = torch.full((N,9), -1, dtype=torch.long, device=self.device); neigh[:, 4] = 0
            self.neighs[depth] = neigh; return
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],
                               device=self.device, dtype=torch.long)
        nc = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        res = 1 << depth; nx, ny = nc[...,0], nc[...,1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
        n_keys = torch.full((N,9), -1, dtype=torch.long, device=self.device)
        if valid.any(): n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)
        idx = torch.searchsorted(keys, n_keys.clamp(min=0)).clamp(0, len(keys)-1)
        found = valid & (keys[idx] == n_keys); idx[~found] = -1
        self.neighs[depth] = idx


# ==========================================
# AE modules (same as before — needed to load checkpoints)
# ==========================================

class QuadConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__(); self.weights = nn.Linear(9 * in_c, out_c)
    def forward(self, features, qt, depth):
        ni = qt.neighs[depth]; N = ni.shape[0]
        if N == 0: return torch.zeros((0, self.weights.out_features), device=features.device)
        pad = torch.zeros((1, features.shape[1]), device=features.device)
        fp = torch.cat([features, pad], dim=0); gi = ni.clone(); gi[gi==-1] = N
        return self.weights(fp[gi].view(N, -1))

class QuadPool(nn.Module):
    def forward(self, cf, qt, dc):
        dp = dc-1; Np = len(qt.keys[dp]); C = cf.shape[1]
        if Np == 0: return torch.zeros((0, C), device=cf.device)
        ch = qt.children_idx[dp]
        pooled = torch.zeros((Np, C), device=cf.device); cnt = torch.zeros((Np, 1), device=cf.device)
        for c in range(4):
            idx = ch[:, c]; mask = idx != -1
            if mask.any(): pooled[mask] += cf[idx[mask]]; cnt[mask] += 1.0
        return pooled / cnt.clamp(min=1.0)

class TreeEncoder(nn.Module):
    def __init__(self, in_c=1, hidden=128, emb_dim=128, max_depth=7, pos_freqs=6):
        super().__init__()
        self.pos_freqs=pos_freqs; self.max_depth=max_depth; self.hidden=hidden; self.emb_dim=emb_dim
        pos_dim = 3 + 2*pos_freqs*3
        self.in_proj=nn.Linear(in_c+pos_dim, hidden)
        self.convs=nn.ModuleList([QuadConv(hidden, hidden) for _ in range(max_depth+1)])
        self.pool=QuadPool()
        self.to_emb=nn.ModuleList([nn.Linear(hidden, emb_dim) for _ in range(max_depth+1)])
        self.emb_norm=nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(max_depth+1)])
        self.depth_gain=nn.Parameter(torch.ones(max_depth+1))
    def forward(self, qt):
        h = [None]*(self.max_depth+1)
        for d in range(self.max_depth+1):
            fin=qt.features_in[d]; kd=qt.keys[d]
            if fin is None or fin.numel()==0: h[d]=torch.zeros((0,self.hidden),device=qt.device); continue
            pos=fourier_encode(node_centers_from_keys(kd,d,self.max_depth,qt.device),self.pos_freqs)
            h[d]=self.in_proj(torch.cat([fin,pos],dim=1))
        for d in range(self.max_depth, 0, -1):
            if h[d].numel()==0: continue
            h[d-1]=h[d-1]+self.pool(h[d],qt,d)
            if d-1>=1 and h[d-1].numel()>0: h[d-1]=F.relu(self.convs[d-1](h[d-1],qt,d-1))
        E=[]
        for d in range(self.max_depth+1):
            if h[d] is None or h[d].numel()==0: E.append(torch.zeros((0,self.emb_dim),device=qt.device))
            else: z=self.emb_norm[d](self.to_emb[d](h[d])); E.append(self.depth_gain[d]*z)
        return E

class TreeDecoder(nn.Module):
    def __init__(self, hidden=128, emb_dim=128, out_c=1, max_depth=7, pos_freqs=6):
        super().__init__()
        self.max_depth=max_depth; self.pos_freqs=pos_freqs; self.hidden=hidden; self.emb_dim=emb_dim
        pos_dim=3+2*pos_freqs*3
        self.root_token=nn.Parameter(torch.zeros(1,hidden))
        self.fuse=nn.Sequential(nn.Linear(hidden+emb_dim+pos_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.skip_norm=nn.LayerNorm(emb_dim)
        self.split_head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.child_head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,4*hidden))
        self.val_head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,out_c))
        self.mix_convs=nn.ModuleList([QuadConv(hidden,hidden) for _ in range(max_depth+1)])
    def forward(self, qt, emb_list):
        N0=len(qt.keys[0])
        hbd=[None]*(self.max_depth+1)
        hbd[0]=self.root_token.expand(N0,-1).to(qt.device) if N0>0 else torch.zeros((0,self.hidden),device=qt.device)
        sl=[None]*self.max_depth; vp=[None]*(self.max_depth+1)
        for d in range(self.max_depth+1):
            h=hbd[d]
            if h is None or h.numel()==0:
                vp[d]=torch.zeros((0,1),device=qt.device)
                if d<self.max_depth: sl[d]=torch.zeros((0,),device=qt.device)
                continue
            kd=qt.keys[d]
            pos=fourier_encode(node_centers_from_keys(kd,d,self.max_depth,qt.device),self.pos_freqs)
            if emb_list is not None and emb_list[d] is not None and emb_list[d].shape[0]==h.shape[0]:
                skip=self.skip_norm(emb_list[d].to(h.device))
            else: skip=torch.zeros((h.shape[0],self.emb_dim),device=h.device)
            h=self.fuse(torch.cat([h,skip,pos],dim=1))
            if d>=1 and h.numel()>0: h=F.relu(self.mix_convs[d](h,qt,d))
            vp[d]=self.val_head(h)
            if d==self.max_depth: break
            sl[d]=self.split_head(h).squeeze(-1)
            ch=qt.children_idx[d]; has_child=(ch!=-1).any(dim=1)
            N_next=len(qt.keys[d+1]); hn=torch.zeros((N_next,self.hidden),device=h.device)
            if has_child.any():
                cf=self.child_head(h[has_child]).view(-1,4,self.hidden)
                pr=torch.nonzero(has_child).squeeze(-1)
                for t,p in enumerate(pr):
                    for c in range(4):
                        ci=int(ch[p,c].item())
                        if ci!=-1: hn[ci]=cf[t,c]
            hbd[d+1]=hn
        return sl, vp

class TreeAE(nn.Module):
    def __init__(self, in_c=1, hidden=128, emb_dim=128, max_depth=7, pos_freqs=6):
        super().__init__()
        self.encoder=TreeEncoder(in_c=in_c,hidden=hidden,emb_dim=emb_dim,max_depth=max_depth,pos_freqs=pos_freqs)
        self.decoder=TreeDecoder(hidden=hidden,emb_dim=emb_dim,out_c=in_c,max_depth=max_depth,pos_freqs=pos_freqs)
    def forward(self, qt):
        E=self.encoder(qt); return *self.decoder(qt,E), E


# ==========================================
# Dataset: loads saved embedding pairs
# ==========================================

class EmbeddingPairDataset(Dataset):
    """
    Loads (E_f, E_u) pairs from .pt files saved by save_embeddings.py.
    Returns them as lists of tensors (variable-length per depth).
    """
    def __init__(self, emb_dir, manifest_path=None):
        self.emb_dir = emb_dir
        if manifest_path is None:
            manifest_path = os.path.join(emb_dir, 'manifest.json')
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.samples = manifest['samples']
        self.max_depth = manifest['max_depth']
        self.emb_dim   = manifest['emb_dim']
        print(f"Dataset: {len(self.samples)} samples, "
              f"max_depth={self.max_depth}, emb_dim={self.emb_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]['file']
        data  = torch.load(os.path.join(self.emb_dir, fname), map_location='cpu')
        return {
            'E_f':   data['E_f'],    # list of (N_d, emb_dim)
            'E_u':   data['E_u'],    # list of (N_d, emb_dim)
            'keys':  data['keys'],   # list of (N_d,) long
            'k1':    data['k1'],
            'k2':    data['k2'],
            'n_nodes_per_depth': data['n_nodes_per_depth'],
        }


def collate_fn(batch):
    """
    Each sample has variable N_d per depth.
    We return a list-of-lists (no padding needed since we process
    each depth independently with no batching across samples).
    For simplicity, we return batch size 1 (no padding).
    """
    # With batch_size=1, just unwrap the single item
    assert len(batch) == 1, "Operator training uses batch_size=1 (variable tree sizes)"
    return batch[0]


# ==========================================
# Operator: multi-scale cross-attention E_f -> E_u
# ==========================================

class DepthwiseAttentionBlock(nn.Module):
    """
    For a single depth d:
      - E_f[d]:   (N_d, emb_dim)   query tokens (f embeddings at this depth)
      - context:  (N_d, K, emb_dim) per-node neighborhood context
                  (built from same-depth neighbors + parent + children)
      - outputs:  (N_d, emb_dim)   predicted u embeddings at this depth

    Uses multi-head attention where each node attends to its K context tokens.
    """
    def __init__(self, emb_dim, n_heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        ff_dim = ff_dim or emb_dim * 2

        assert emb_dim % n_heads == 0
        self.head_dim = emb_dim // n_heads

        # Q from f-node itself, K/V from context (also f)
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, context):
        """
        x:       (N, emb_dim)     — f embeddings, used as query
        context: (N, K, emb_dim)  — neighborhood, used as key/value
        returns: (N, emb_dim)     — predicted contribution to u embedding
        """
        N, K, C = context.shape
        residual = x

        # Multi-head attention
        Q = self.q_proj(x).view(N, self.n_heads, self.head_dim)            # (N, H, Dh)
        K_ = self.k_proj(context).view(N, K, self.n_heads, self.head_dim)  # (N, K, H, Dh)
        V_ = self.v_proj(context).view(N, K, self.n_heads, self.head_dim)  # (N, K, H, Dh)

        # Attention scores: (N, H, 1, K)
        Q = Q.unsqueeze(2)  # (N, H, 1, Dh)
        K_ = K_.permute(0, 2, 1, 3)  # (N, H, K, Dh)
        V_ = V_.permute(0, 2, 1, 3)  # (N, H, K, Dh)

        scores = (Q @ K_.transpose(-2, -1)) * self.scale  # (N, H, 1, K)
        attn   = torch.softmax(scores, dim=-1)             # (N, H, 1, K)
        out    = (attn @ V_).squeeze(2)                    # (N, H, Dh)
        out    = out.contiguous().view(N, C)               # (N, emb_dim)
        out    = self.out_proj(out)

        # Residual + norm
        x = self.norm1(residual + self.dropout(out))

        # Feed-forward
        x = self.norm2(x + self.ff(x))
        return x


def build_neighborhood_context(E_f_list, keys_list, neigh_idx_list,
                               parent_idx_list, children_idx_list,
                               depth, max_depth, device):
    """
    For each node at `depth`, gather:
      - its 9 same-depth neighbors (from neigh_idx)
      - its parent (from parent_idx)
      - its up to 4 children (from children_idx)
    All from E_f, zero-padding for missing entries.

    Returns: context (N_d, K, emb_dim)  where K = 9 + 1 + 4 = 14
    """
    E_d = E_f_list[depth]   # (N_d, C)
    N, C = E_d.shape

    if N == 0:
        return torch.zeros((0, 14, C), device=device)

    ctx_parts = []

    # ---- 9 same-depth neighbors ----
    ni = neigh_idx_list[depth].to(device)  # (N, 9)
    pad = torch.zeros((1, C), device=device)
    E_d_pad = torch.cat([E_d, pad], dim=0)
    gather = ni.clone(); gather[gather == -1] = N
    same_depth_ctx = E_d_pad[gather]       # (N, 9, C)
    ctx_parts.append(same_depth_ctx)

    # ---- parent (depth-1) ----
    if depth > 0 and parent_idx_list[depth] is not None:
        E_parent = E_f_list[depth - 1]      # (N_{d-1}, C)
        pidx = parent_idx_list[depth].to(device)  # (N_d,)
        if E_parent.numel() > 0:
            parent_ctx = E_parent[pidx].unsqueeze(1)  # (N, 1, C)
        else:
            parent_ctx = torch.zeros((N, 1, C), device=device)
    else:
        parent_ctx = torch.zeros((N, 1, C), device=device)
    ctx_parts.append(parent_ctx)

    # ---- children (depth+1), up to 4 ----
    if depth < max_depth and children_idx_list[depth] is not None:
        E_child = E_f_list[depth + 1]  # (N_{d+1}, C)
        ch = children_idx_list[depth].to(device)  # (N_d, 4)
        if E_child.numel() > 0:
            pad_c = torch.zeros((1, C), device=device)
            E_child_pad = torch.cat([E_child, pad_c], dim=0)
            gc = ch.clone(); gc[gc == -1] = E_child.shape[0]
            child_ctx = E_child_pad[gc]  # (N, 4, C)
        else:
            child_ctx = torch.zeros((N, 4, C), device=device)
    else:
        child_ctx = torch.zeros((N, 4, C), device=device)
    ctx_parts.append(child_ctx)

    # cat → (N, 14, C)
    return torch.cat(ctx_parts, dim=1)


class PoissonOperator(nn.Module):
    """
    Multi-scale operator: E_f --> E_u

    For each depth d:
      1. Build neighborhood context from E_f (same-depth neighbors + parent + children)
      2. Apply DepthwiseAttentionBlock to produce E_u_pred[d]
      3. Apply a residual depth-specific MLP to refine

    Depth-specific weights: each depth gets its own attention block + MLP,
    because the operator may have different character at coarse vs fine scales.
    """
    def __init__(self, emb_dim=128, n_heads=4, n_layers=2, max_depth=7, dropout=0.1):
        super().__init__()
        self.max_depth = max_depth
        self.emb_dim   = emb_dim

        # One stack of attention layers per depth
        self.depth_blocks = nn.ModuleList([
            nn.Sequential(*[
                DepthwiseAttentionBlock(emb_dim, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ])
            for _ in range(max_depth + 1)
        ])

        # Per-depth output projection (maps attended features to E_u space)
        self.out_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            )
            for _ in range(max_depth + 1)
        ])

    def forward(self, E_f_list, keys_list, neigh_idx_list,
                parent_idx_list, children_idx_list, device):
        """
        E_f_list: list of (N_d, emb_dim) — f embeddings per depth
        Returns:  list of (N_d, emb_dim) — predicted u embeddings per depth
        """
        E_u_pred = []
        for d in range(self.max_depth + 1):
            E_d = E_f_list[d]
            if E_d.numel() == 0:
                E_u_pred.append(torch.zeros_like(E_d))
                continue

            # Build multi-scale context
            ctx = build_neighborhood_context(
                E_f_list, keys_list, neigh_idx_list,
                parent_idx_list, children_idx_list,
                d, self.max_depth, device)   # (N_d, 14, emb_dim)

            # Apply depth-specific attention layers
            h = E_d
            for layer in self.depth_blocks[d]:
                h = layer(h, ctx)

            # Output projection
            E_u_d = self.out_proj[d](h)  # (N_d, emb_dim)
            E_u_pred.append(E_u_d)

        return E_u_pred


# ==========================================
# Poisson data generation (for end-to-end eval)
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
        self.x,self.y,self.size,self.level=x,y,size,level
        self.max_level=max_level; self.min_level=min_level
        self.children=[]; self.f_val=None; self.u_val=None
    def gradient_subdivide(self, k1, k2, grad_threshold=0.3):
        if self.level >= self.max_level: return
        cx,cy=self.x+self.size/2, self.y+self.size/2
        force_split = self.level < self.min_level
        grad_mag = gradient_magnitude_f(cx, cy, k1, k2)
        max_grad = 2*np.pi*np.sqrt(k1**2+k2**2)
        depth_factor = 1.0+0.1*self.level
        if force_split or (grad_mag/max_grad > grad_threshold*depth_factor):
            half=self.size/2
            self.children=[
                QuadNode(self.x,self.y,half,self.level+1,self.max_level,self.min_level),
                QuadNode(self.x+half,self.y,half,self.level+1,self.max_level,self.min_level),
                QuadNode(self.x,self.y+half,half,self.level+1,self.max_level,self.min_level),
                QuadNode(self.x+half,self.y+half,half,self.level+1,self.max_level,self.min_level),
            ]
            for c in self.children: c.gradient_subdivide(k1, k2, grad_threshold)
    def collect_leaves(self, leaves_list, k1, k2):
        if not self.children:
            cx,cy=self.x+self.size/2,self.y+self.size/2
            self.f_val=f_function(cx,cy,k1,k2); self.u_val=u_function(cx,cy,k1,k2)
            leaves_list.append(self)
        else:
            for c in self.children: c.collect_leaves(leaves_list, k1, k2)

def generate_poisson_pair(noise_std=NOISE_STD, grad_threshold=0.3):
    k1=random.randint(1,5); k2=random.randint(1,5)
    root=QuadNode(0,0,1.0,0,MAX_LEVEL,MIN_LEVEL)
    root.gradient_subdivide(k1,k2,grad_threshold=grad_threshold)
    leaves=[]; root.collect_leaves(leaves,k1,k2)
    leaf_keys=[[] for _ in range(MAX_LEVEL+1)]
    f_vals_raw=[[] for _ in range(MAX_LEVEL+1)]
    u_vals_raw=[[] for _ in range(MAX_LEVEL+1)]
    for node in leaves:
        d=node.level; res=1<<d
        ix=max(0,min(res-1,int(node.x*res))); iy=max(0,min(res-1,int(node.y*res)))
        k=Morton2D.xy2key(ix,iy)
        leaf_keys[d].append(int(k.item()))
        f_vals_raw[d].append([float(node.f_val)])
        u_vals_raw[d].append([float(node.u_val)])
    lkbd=[]; fcbd=[]; fnbd=[]; ucbd=[]; unbd=[]
    for d in range(MAX_LEVEL+1):
        if len(leaf_keys[d])==0:
            lkbd.append(torch.empty((0,),dtype=torch.long))
            fcbd.append(torch.empty((0,1),dtype=torch.float32)); fnbd.append(torch.empty((0,1),dtype=torch.float32))
            ucbd.append(torch.empty((0,1),dtype=torch.float32)); unbd.append(torch.empty((0,1),dtype=torch.float32))
        else:
            kt=torch.tensor(leaf_keys[d],dtype=torch.long)
            fc=torch.tensor(f_vals_raw[d],dtype=torch.float32)
            uc=torch.tensor(u_vals_raw[d],dtype=torch.float32)
            lkbd.append(kt); fcbd.append(fc); fnbd.append(fc+noise_std*torch.randn_like(fc))
            ucbd.append(uc); unbd.append(uc+noise_std*torch.randn_like(uc))
    return lkbd, fcbd, fnbd, ucbd, unbd, leaves, k1, k2


# ==========================================
# Training loop
# ==========================================

def train(args):
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Dataset ---
    dataset = EmbeddingPairDataset(args.emb_dir)
    n_train = int(0.9 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    print(f"Train: {n_train}  Val: {n_val}")

    # --- Load frozen AE models ---
    ckpt_f = torch.load(args.model_f, map_location=device)
    ae_f = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    ae_f.load_state_dict(ckpt_f['model_state_dict'])
    ae_f.eval()
    for p in ae_f.parameters(): p.requires_grad_(False)
    print(f"ae_f loaded (frozen) — step={ckpt_f['step']}")

    ckpt_u = torch.load(args.model_u, map_location=device)
    ae_u = TreeAE(in_c=1, hidden=128, emb_dim=128, max_depth=MAX_LEVEL, pos_freqs=6).to(device)
    ae_u.load_state_dict(ckpt_u['model_state_dict'])
    ae_u.eval()
    for p in ae_u.parameters(): p.requires_grad_(False)
    print(f"ae_u loaded (frozen) — step={ckpt_u['step']}")

    # --- Operator ---
    operator = PoissonOperator(
        emb_dim=128, n_heads=args.n_heads,
        n_layers=args.n_layers, max_depth=MAX_LEVEL,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in operator.parameters() if p.requires_grad)
    print(f"Operator parameters: {n_params:,}")

    optimizer = torch.optim.Adam(operator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss()

    best_val_loss = float('inf')
    best_path = os.path.join(args.out_dir, 'best_operator.pt')

    history = {'train_emb': [], 'val_emb': [], 'val_dec': []}

    print(f"\nTraining operator for {args.epochs} epochs")
    print(f"  emb loss weight={args.lambda_emb}, decoder loss weight={args.lambda_dec}")

    for epoch in range(args.epochs):
        # ---- Train ----
        operator.train()
        train_emb_losses = []

        for sample in train_loader:
            E_f  = [e.to(device) for e in sample['E_f']]
            E_u  = [e.to(device) for e in sample['E_u']]
            keys = [k.to(device) for k in sample['keys']]

            # Build neighbor/parent/children indices from saved keys
            # (We reconstruct these on the fly from keys — fast and avoids storing them)
            neigh_idx    = _build_neighs_from_keys(keys, MAX_LEVEL, device)
            parent_idx   = _build_parents_from_keys(keys, MAX_LEVEL, device)
            children_idx = _build_children_from_keys(keys, MAX_LEVEL, device)

            optimizer.zero_grad()

            # Operator forward: E_f --> E_u_pred
            E_u_pred = operator(E_f, keys, neigh_idx, parent_idx, children_idx, device)

            # Embedding loss: predicted E_u vs target E_u
            L_emb = torch.tensor(0.0, device=device)
            n_emb = 0
            for d in range(MAX_LEVEL + 1):
                if E_u[d].numel() == 0: continue
                L_emb = L_emb + mse(E_u_pred[d], E_u[d])
                n_emb += 1
            if n_emb > 0: L_emb = L_emb / n_emb

            L_total = args.lambda_emb * L_emb
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(operator.parameters(), 1.0)
            optimizer.step()
            train_emb_losses.append(L_emb.item())

        scheduler.step()

        # ---- Validate ----
        operator.eval()
        val_emb_losses = []
        val_dec_losses = []  # end-to-end: u_pred vs u_gt at leaves

        with torch.no_grad():
            for sample in val_loader:
                E_f  = [e.to(device) for e in sample['E_f']]
                E_u  = [e.to(device) for e in sample['E_u']]
                keys = [k.to(device) for k in sample['keys']]
                k1, k2 = int(sample['k1']), int(sample['k2'])

                neigh_idx    = _build_neighs_from_keys(keys, MAX_LEVEL, device)
                parent_idx   = _build_parents_from_keys(keys, MAX_LEVEL, device)
                children_idx = _build_children_from_keys(keys, MAX_LEVEL, device)

                E_u_pred = operator(E_f, keys, neigh_idx, parent_idx, children_idx, device)

                L_emb = torch.tensor(0.0, device=device)
                n_emb = 0
                for d in range(MAX_LEVEL + 1):
                    if E_u[d].numel() == 0: continue
                    L_emb = L_emb + mse(E_u_pred[d], E_u[d])
                    n_emb += 1
                if n_emb > 0: L_emb = L_emb / n_emb
                val_emb_losses.append(L_emb.item())

                # End-to-end decoder loss: decode E_u_pred → u values
                # Build a Quadtree from the saved keys to run the decoder
                qt = _qt_from_keys(keys, MAX_LEVEL, device, k1, k2)
                _, val_pred = ae_u.decoder(qt, E_u_pred)

                L_dec = torch.tensor(0.0, device=device); n_dec = 0
                for d in range(MAX_LEVEL + 1):
                    mask = qt.leaf_mask[d]
                    if mask is None or mask.numel() == 0 or not mask.any(): continue
                    L_dec = L_dec + mse(val_pred[d][mask], qt.values_gt[d][mask])
                    n_dec += 1
                if n_dec > 0: L_dec = L_dec / n_dec
                val_dec_losses.append(L_dec.item())

        mean_train_emb = np.mean(train_emb_losses)
        mean_val_emb   = np.mean(val_emb_losses)
        mean_val_dec   = np.mean(val_dec_losses)

        history['train_emb'].append(mean_train_emb)
        history['val_emb'].append(mean_val_emb)
        history['val_dec'].append(mean_val_dec)

        print(f"Epoch {epoch+1:4d}/{args.epochs}  "
              f"train_emb={mean_train_emb:.6f}  "
              f"val_emb={mean_val_emb:.6f}  "
              f"val_dec={mean_val_dec:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if mean_val_emb < best_val_loss:
            best_val_loss = mean_val_emb
            torch.save({
                'epoch': epoch,
                'operator_state_dict': operator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_emb_loss': mean_val_emb,
                'val_dec_loss': mean_val_dec,
                'args': vars(args),
            }, best_path)
            print(f"  -> Saved best operator (val_emb={mean_val_emb:.6f})")

        if (epoch + 1) % 10 == 0:
            _plot_losses(history, args.out_dir, epoch + 1)
            _run_e2e_demo(ae_f, operator, ae_u, args.out_dir, epoch + 1, device)

    print(f"\nTraining complete. Best val_emb={best_val_loss:.6f}")
    print(f"Best operator: {best_path}")
    _plot_losses(history, args.out_dir, args.epochs)


# ==========================================
# Topology helpers: rebuild structural indices from keys
# ==========================================

def _build_neighs_from_keys(keys_list, max_depth, device):
    """Rebuild same-depth neighbor index lists from Morton keys."""
    neighs = []
    for depth, keys in enumerate(keys_list):
        keys = keys.to(device); N = len(keys)
        if N == 0:
            neighs.append(torch.empty((0, 9), dtype=torch.long, device=device)); continue
        if depth == 0:
            ni = torch.full((N, 9), -1, dtype=torch.long, device=device); ni[:, 4] = 0
            neighs.append(ni); continue
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor(
            [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],
            device=device, dtype=torch.long)
        nc = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        res = 1 << depth; nx, ny = nc[...,0], nc[...,1]
        valid = (nx >= 0) & (nx < res) & (ny >= 0) & (ny < res)
        n_keys = torch.full((N, 9), -1, dtype=torch.long, device=device)
        if valid.any():
            n_keys[valid] = Morton2D.xy2key(nx[valid], ny[valid], depth=depth)
        idx = torch.searchsorted(keys, n_keys.clamp(min=0)).clamp(0, N-1)
        found = valid & (keys[idx] == n_keys); idx[~found] = -1
        neighs.append(idx)
    return neighs


def _build_parents_from_keys(keys_list, max_depth, device):
    """Rebuild parent index lists."""
    parents = [None]
    for depth in range(1, max_depth + 1):
        kd = keys_list[depth].to(device); kp = keys_list[depth-1].to(device)
        if kd.numel() == 0:
            parents.append(torch.empty((0,), dtype=torch.long, device=device)); continue
        pidx = torch.searchsorted(kp, kd >> 2).clamp(0, len(kp)-1)
        parents.append(pidx)
    return parents


def _build_children_from_keys(keys_list, max_depth, device):
    """Rebuild children index lists."""
    children = []
    for depth in range(max_depth):
        kd = keys_list[depth].to(device); kn = keys_list[depth+1].to(device)
        if kd.numel() == 0:
            children.append(torch.empty((0,4), dtype=torch.long, device=device)); continue
        if kn.numel() == 0:
            children.append(torch.full((len(kd),4), -1, dtype=torch.long, device=device)); continue
        ck = (kd.unsqueeze(1) << 2) + torch.arange(4, device=device).view(1,4)
        idx = torch.searchsorted(kn, ck).clamp(0, len(kn)-1); found = kn[idx] == ck
        children.append(torch.where(found, idx, torch.full_like(idx, -1)))
    children.append(None)  # no children at max_depth
    return children


def _qt_from_keys(keys_list, max_depth, device, k1, k2):
    """
    Reconstruct a minimal Quadtree from saved keys for decoder use.
    Populates values_gt from the analytic solution so we can compute decoder loss.
    """
    # Build leaf_keys and analytic values from the saved keys
    # Leaves = nodes at max_depth OR nodes with no children
    leaf_keys_by_depth  = []
    leaf_vals_by_depth  = []

    for d in range(max_depth + 1):
        kd = keys_list[d]
        # Determine leaves: nodes that have no children in d+1
        if d < max_depth:
            kn = keys_list[d+1]
            # A node is a leaf if none of its 4 child keys exist in kn
            ck = (kd.unsqueeze(1) << 2) + torch.arange(4, device=kd.device).view(1,4)
            idx = torch.searchsorted(kn.cpu(), ck.cpu()).clamp(0, len(kn)-1)
            found = kn.cpu()[idx] == ck.cpu()
            is_leaf = ~found.any(dim=1)
        else:
            is_leaf = torch.ones(len(kd), dtype=torch.bool)

        leaf_k = kd[is_leaf]
        if leaf_k.numel() == 0:
            leaf_keys_by_depth.append(torch.empty((0,), dtype=torch.long))
            leaf_vals_by_depth.append(torch.empty((0,1), dtype=torch.float32))
        else:
            ix, iy = Morton2D.key2xy(leaf_k.cpu(), depth=d)
            res = float(1 << d)
            x = (ix.float() + 0.5) / res
            y = (iy.float() + 0.5) / res
            u_vals = torch.tensor(
                [[u_function(float(x[i]), float(y[i]), k1, k2)] for i in range(len(x))],
                dtype=torch.float32)
            leaf_keys_by_depth.append(leaf_k.cpu())
            leaf_vals_by_depth.append(u_vals)

    qt = Quadtree(max_depth=max_depth, device=device)
    qt.build_from_leaves(leaf_keys_by_depth, leaf_vals_by_depth, leaf_vals_by_depth)
    return qt


# ==========================================
# End-to-end demo: encoder_f -> operator -> decoder_u
# ==========================================

def _run_e2e_demo(ae_f, operator, ae_u, out_dir, epoch, device, n_demos=3):
    """Generate a few plots showing the full pipeline."""
    operator.eval()
    cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(n_demos, 4, figsize=(24, 6 * n_demos))
    if n_demos == 1: axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row in range(n_demos):
            lkbd, fcbd, fnbd, ucbd, unbd, leaves, k1, k2 = generate_poisson_pair()

            # Build qt_f for encoder
            qt_f = Quadtree(max_depth=MAX_LEVEL, device=device)
            qt_f.build_from_leaves(lkbd, fnbd, fcbd)

            # Full pipeline:
            # 1. Encode f
            E_f = ae_f.encoder(qt_f)

            # 2. Rebuild topology helpers from qt_f keys
            keys    = qt_f.keys
            neighs  = _build_neighs_from_keys(keys, MAX_LEVEL, device)
            parents = _build_parents_from_keys(keys, MAX_LEVEL, device)
            children = _build_children_from_keys(keys, MAX_LEVEL, device)

            # 3. Operator: predict E_u
            E_u_pred = operator(E_f, keys, neighs, parents, children, device)

            # 4. Decode u using qt_f topology + predicted E_u
            _, val_pred_u = ae_u.decoder(qt_f, E_u_pred)

            # Gather predictions and GT for leaves
            pred_dict = {}; loss_dict = {}
            for d in range(MAX_LEVEL + 1):
                if qt_f.keys[d].numel() == 0: continue
                mask = qt_f.leaf_mask[d]
                if mask is None or mask.numel() == 0 or not mask.any(): continue
                kd = qt_f.keys[d][mask].cpu()
                ix, iy = Morton2D.key2xy(kd, depth=d)
                pr = val_pred_u[d][mask].cpu().numpy().flatten()
                # GT u
                res = float(1 << d)
                for i in range(len(kd)):
                    x_c = (ix[i].item() + 0.5) / res
                    y_c = (iy[i].item() + 0.5) / res
                    gt  = u_function(x_c, y_c, k1, k2)
                    key = (d, int(ix[i].item()), int(iy[i].item()))
                    pred_dict[key] = pr[i]
                    loss_dict[key] = (pr[i] - gt) ** 2

            max_loss = max(max(loss_dict.values()) if loss_dict else 1e-6, 1e-6)

            def node_key(nd):
                d=nd.level; res=1<<d
                ix=max(0,min(res-1,int(nd.x*res))); iy=max(0,min(res-1,int(nd.y*res)))
                return (d, ix, iy)

            # Col 0: GT f
            ax = axes[row, 0]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
            ax.set_title(f"GT f  (k1={k1}, k2={k2})", fontsize=9)
            for nd in leaves:
                c = cmap(np.clip((nd.f_val + 1)/2, 0, 1))
                ax.add_patch(patches.Rectangle((nd.x,nd.y),nd.size,nd.size,lw=0.3,ec='k',fc=c))

            # Col 1: GT u
            ax = axes[row, 1]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
            u_scale = 4*np.pi**2*(k1**2+k2**2)
            ax.set_title(f"GT u  (scale ~1/{u_scale:.0f})", fontsize=9)
            for nd in leaves:
                c = cmap(np.clip((nd.u_val * u_scale + 1)/2, 0, 1))
                ax.add_patch(patches.Rectangle((nd.x,nd.y),nd.size,nd.size,lw=0.3,ec='k',fc=c))

            # Col 2: Predicted u (end-to-end)
            ax = axes[row, 2]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
            ax.set_title(f"Pred u  (encoder_f → op → decoder_u)", fontsize=9)
            for nd in leaves:
                v = pred_dict.get(node_key(nd), 0.0)
                c = cmap(np.clip((v * u_scale + 1)/2, 0, 1))
                ax.add_patch(patches.Rectangle((nd.x,nd.y),nd.size,nd.size,lw=0.3,ec='k',fc=c))

            # Col 3: Loss map
            loss_cmap = plt.get_cmap('hot')
            ax = axes[row, 3]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
            avg_loss = np.mean(list(loss_dict.values())) if loss_dict else 0
            ax.set_title(f"MSE map  (mean={avg_loss:.5f})", fontsize=9)
            for nd in leaves:
                ll = loss_dict.get(node_key(nd), 0.0)
                c = loss_cmap(np.clip(ll/max_loss, 0, 1))
                ax.add_patch(patches.Rectangle((nd.x,nd.y),nd.size,nd.size,lw=0.3,ec='k',fc=c))

    plt.suptitle(f"End-to-end Poisson solve — epoch {epoch}  "
                 f"[encoder_f → operator → decoder_u]", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"e2e_epoch{epoch:04d}.png")
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


def _plot_losses(history, out_dir, epoch):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title, color in [
        (axes[0], 'train_emb', 'Train Emb Loss (E_f→E_u MSE)', 'steelblue'),
        (axes[1], 'val_emb',   'Val Emb Loss',                   'coral'),
        (axes[2], 'val_dec',   'Val Dec Loss (u values MSE)',     'darkred'),
    ]:
        ax.plot(history[key], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=10); ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Operator Training Losses — epoch {epoch}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'losses_epoch{epoch:04d}.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# Entry point
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir',    required=True,  help='Dir with saved .pt embedding pairs')
    parser.add_argument('--model_f',    required=True,  help='Path to best_model_f.pt')
    parser.add_argument('--model_u',    required=True,  help='Path to best_model_u.pt')
    parser.add_argument('--out_dir',    default='operator/poisson')
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--n_heads',    type=int,   default=4)
    parser.add_argument('--n_layers',   type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.1)
    parser.add_argument('--lambda_emb', type=float, default=1.0,
                        help='Weight on embedding MSE loss')
    parser.add_argument('--lambda_dec', type=float, default=0.0,
                        help='Weight on decoder value loss (0=off, expensive when on)')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Emb dir: {args.emb_dir}")
    train(args)