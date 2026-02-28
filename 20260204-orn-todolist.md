1. Need to check if the key2xy or the xy2key are correctly implemented 
2. Need to check if the rows anre actually y and the cols are actually x 
3. Not very sture of the global context. I don't think it is correct
    ~ global_context: (Batch, 1, Dim) -> Frequency/Phase context 
    ~ But Global_context should have some more information than just a single number right?


-----

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. FIXED COORDINATE ENGINE
# ==========================================
def xy2key(x, y, depth=8):
    key = torch.zeros_like(x, dtype=torch.long)
    for i in range(depth):
        key |= ((x >> i) & 1) << (2 * i + 1) # Odd bits: X (cols)
        key |= ((y >> i) & 1) << (2 * i)     # Even bits: Y (rows)
    return key

def key2xy(key, depth=8):
    y = torch.zeros_like(key, dtype=torch.long)
    x = torch.zeros_like(key, dtype=torch.long)
    for i in range(depth):
        y |= ((key >> (2 * i)) & 1) << i
        x |= ((key >> (2 * i + 1)) & 1) << i
    return x, y

# ==========================================
# 2. UPDATED MODEL WITH ATTENTION & ALIGNMENT
# ==========================================
class RefinerNetV2(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        # 1. Hierarchical Encoder (Produces Depth 4 and Depth 8 features)
        self.enc_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, feat_dim, 3, stride=2, padding=1) # 128 -> 64 (Depth 6 context)
        )
        self.enc_layer2 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1), # Depth 8 features
            nn.ReLU()
        )
        
        # 2. Multi-Scale Alignment Head
        # Pulls context from a coarser level to guide phase reconstruction
        self.context_projection = nn.Linear(feat_dim, feat_dim)

        # 3. Sparse Attention (ROI + Halo + Global)
        self.attention = SparseRefinementAttention(feat_dim)

        # 4. High-Capacity Upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 32, kernel_size=4, stride=2, padding=1), # 10 -> 20
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 20 -> 40
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, row, col, patch_size=10):
        # --- PHASE 1: Hierarchical Encoding ---
        # We need context at multiple depths
        d6_feats = self.enc_layer1(x) # [B, 64, 64, 64]
        d8_feats = F.interpolate(d6_feats, size=(256, 256), mode='bilinear') # [B, 64, 256, 256]
        
        # --- PHASE 2: Alignment & Halo ---
        # Extract the ROI Features (Current level)
        roi_feats = d8_feats[:, :, row:row+patch_size, col:col+patch_size]
        
        # Extract the Halo (Boundary context) - Simple 2D padding/slicing for demo
        # A 12x12 area centered on the 10x10 patch
        halo_row_start = max(0, row-1)
        halo_col_start = max(0, col-1)
        halo_feats = d8_feats[:, :, halo_row_start:halo_row_start+patch_size+2, 
                                    halo_col_start:halo_col_start+patch_size+2]

        # Extract Aligned Ancestry (Global/Parent phase)
        # We look at the Depth 6 features at the corresponding location
        p_row, p_col = row // 4, col // 4 # Coordinate mapping to coarser grid
        parent_context = d6_feats[:, :, p_row:p_row+3, p_col:p_col+3] # 3x3 local neighborhood
        
        # --- PHASE 3: Attention Fusion ---
        # Flatten for sequence processing
        B, C, H, W = roi_feats.shape
        roi_seq = roi_feats.flatten(2).permute(0, 2, 1) # [B, 100, 64]
        halo_seq = halo_feats.flatten(2).permute(0, 2, 1) # [B, 144, 64]
        parent_seq = parent_context.flatten(2).permute(0, 2, 1) # [B, 9, 64]
        
        # Patch attends to its Boundary (Halo) and its Ancestry (Parent)
        refined_seq = self.attention(roi_seq, halo_seq, parent_seq)
        
        # --- PHASE 4: Upsampling ---
        refined_grid = refined_seq.permute(0, 2, 1).view(B, C, H, W)
        output = self.upsampler(refined_grid)
        
        return output

class SparseRefinementAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, roi, halo, parent):
        # Context = Boundary information + Parent (Global Phase) information
        context = torch.cat([halo, parent], dim=1) 
        
        # Standard Cross-Attention
        attn_out, _ = self.mha(query=roi, key=context, value=context)
        return self.norm(roi + attn_out)