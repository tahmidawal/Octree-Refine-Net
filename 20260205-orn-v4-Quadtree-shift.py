import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 1. MORTON & COORDINATE UTILS (Corrected)
# ==========================================
def xy2key(x, y, depth=8):
    """Encodes X (cols) and Y (rows) into Morton Key."""
    key = torch.zeros_like(x, dtype=torch.long)
    for i in range(depth):
        key |= ((x >> i) & 1) << (2 * i + 1)
        key |= ((y >> i) & 1) << (2 * i)
    return key

def key2xy(key, depth=8):
    """Decodes Morton Key into X (cols) and Y (rows)."""
    y = torch.zeros_like(key, dtype=torch.long)
    x = torch.zeros_like(key, dtype=torch.long)
    for i in range(depth):
        y |= ((key >> (2 * i)) & 1) << i
        x |= ((key >> (2 * i + 1)) & 1) << i
    return x, y

# ==========================================
# 2. FOURIER FEATURES (The "Spectral" Fix)
# ==========================================
class FourierEmbedding(nn.Module):
    def __init__(self, in_dims=2, embed_dim=64, scale=30.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dims, embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x: [Batch, 2, H, W]
        x_proj = (2 * np.pi * x.permute(0, 2, 3, 1)) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).permute(0, 3, 1, 2)

# ==========================================
# 3. THE MODEL: FUSED-DEPTH QUADNET
# ==========================================
class QuadTreeUNet(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        
        # --- ENCODERS ---
        
        # 1. Global Coarse Encoder (Depth 8 Input -> Depth 8 Features)
        self.global_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim), nn.GELU()
        )

        # 2. Sparse Detail Encoder (Depth 10 Patch A -> Depth 8 Features)
        # Compresses 40x40 high-res patch into 10x10 parent features
        self.detail_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 40 -> 20
            nn.GELU(),
            nn.Conv2d(32, feat_dim, 3, stride=2, padding=1), # 20 -> 10 (Matches D8 resolution)
            nn.BatchNorm2d(feat_dim), nn.GELU()
        )

        # --- DECODERS ---

        # 3. Global Reconstruction Head (Rebuilds the coarse quadtree)
        self.global_recon = nn.Sequential(
            nn.Conv2d(feat_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

        # 4. Shared Sparse Refiner (The "Generator")
        # Takes D8 Parent Features + Fourier Pos -> Generates D10 Children
        self.pos_enc = FourierEmbedding(in_dims=2, embed_dim=feat_dim, scale=30.0)
        
        self.refiner = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            
            # Upsample 10 -> 20
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            
            # Upsample 20 -> 40
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, coarse_grid, patch_A, loc_A, loc_B):
        """
        coarse_grid: [B, 1, 256, 256]
        patch_A: [B, 1, 40, 40] (High Res Input)
        loc_A: (row, col) tuple for Patch A
        loc_B: (row, col) tuple for Patch B
        """
        # 1. Encode Global Context (Depth 8)
        global_feats = self.global_enc(coarse_grid) # [B, 64, 256, 256]

        # 2. Encode Patch A Details (Depth 10 -> Depth 8)
        # This represents the "Information Injection" from child to parent
        detail_feats_A = self.detail_enc(patch_A) # [B, 64, 10, 10]

        # 3. Fusion (Simulating the Octree Merge)
        # We perform a residual fusion at Patch A's location in the global map
        # This makes the global map "aware" that A has high-res info available
        fused_feats = global_feats.clone()
        ra, ca = loc_A
        fused_feats[:, :, ra:ra+10, ca:ca+10] = fused_feats[:, :, ra:ra+10, ca:ca+10] + detail_feats_A

        # 4. Task 1: Reconstruct Coarse Quadtree
        recon_coarse = self.global_recon(fused_feats)

        # 5. Task 2: Refine/Generate Sparse Patches
        # We use the SHARED refiner to process both A (Reconstruction) and B (Generation)
        
        # Helper to run refiner on a specific patch location
        def run_refiner(row, col):
            # Extract Halo Context (12x12) from FUSED map
            pad = 1
            r_start, r_end = max(0, row-pad), min(256, row+10+pad)
            c_start, c_end = max(0, col-pad), min(256, col+10+pad)
            
            roi_feats = fused_feats[:, :, r_start:r_end, c_start:c_end]
            
            # Manual padding for edge cases
            th, tw = 10 + 2*pad, 10 + 2*pad
            if roi_feats.shape[2] != th or roi_feats.shape[3] != tw:
                roi_feats = F.pad(roi_feats, (0, tw-roi_feats.shape[3], 0, th-roi_feats.shape[2]))

            # Fourier Positional Encoding (The Phase Guide)
            B, _, H, W = roi_feats.shape
            y_grid = torch.linspace(-1, 1, H, device=coarse_grid.device).view(1, 1, H, 1).expand(B, -1, -1, W)
            x_grid = torch.linspace(-1, 1, W, device=coarse_grid.device).view(1, 1, 1, W).expand(B, -1, H, -1)
            grid = torch.cat([x_grid, y_grid], dim=1)
            
            pos = self.pos_enc(grid)
            combined = torch.cat([roi_feats, pos], dim=1)
            
            # Generate High Res (48x48 -> Crop to 40x40)
            out = self.refiner(combined)
            out = out[:, :, 4:-4, 4:-4] 
            return out

        out_A = run_refiner(ra, ca) # Should match input Patch A
        rb, cb = loc_B
        out_B = run_refiner(rb, cb) # Should match Ground Truth B (Pure Generation)

        return recon_coarse, out_A, out_B

# ==========================================
# 4. HIGH-FREQ DATA GENERATION
# ==========================================
def generate_data(batch_size=8):
    grid_size = 256
    patch_size = 10
    
    # Coarse coords
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    
    inputs_coarse = []
    inputs_A_hr = []
    targets_B_hr = []
    locs = []

    for _ in range(batch_size):
        # High Frequency Mix (Stress Test)
        k = np.random.uniform(15, 45, size=4)
        signal = torch.sin(2*np.pi*k[0]*gx) * torch.sin(2*np.pi*k[1]*gy)
        signal += torch.sin(2*np.pi*k[2]*gx) * torch.sin(2*np.pi*k[3]*gy)
        signal *= 0.5

        # Pick Locations
        # A: The "Known" High Res Patch
        ra = np.random.randint(patch_size, grid_size - patch_size)
        ca = np.random.randint(patch_size, grid_size - patch_size)
        
        # B: The "Unknown" Target Patch (ensure non-overlapping)
        while True:
            rb = np.random.randint(patch_size, grid_size - patch_size)
            cb = np.random.randint(patch_size, grid_size - patch_size)
            if abs(ra-rb) > 12 or abs(ca-cb) > 12: break

        # Generate High Res Ground Truths
        def get_hr_patch(r, c):
            hr_x = torch.linspace(x[c], x[c+patch_size-1], patch_size*4)
            hr_y = torch.linspace(y[r], y[r+patch_size-1], patch_size*4)
            hgy, hgx = torch.meshgrid(hr_y, hr_x, indexing='ij')
            res = torch.sin(2*np.pi*k[0]*hgx) * torch.sin(2*np.pi*k[1]*hgy)
            res += torch.sin(2*np.pi*k[2]*hgx) * torch.sin(2*np.pi*k[3]*hgy)
            return res * 0.5

        patch_A = get_hr_patch(ra, ca)
        patch_B = get_hr_patch(rb, cb)

        inputs_coarse.append(signal.unsqueeze(0))
        inputs_A_hr.append(patch_A.unsqueeze(0))
        targets_B_hr.append(patch_B.unsqueeze(0))
        locs.append({'A': (ra, ca), 'B': (rb, cb)})

    return torch.stack(inputs_coarse), torch.stack(inputs_A_hr), torch.stack(targets_B_hr), locs

# ==========================================
# 5. TRAINING & TESTING
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QuadTreeUNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

print(f"Starting Training on {device}...")
print("Task: Reconstruct Coarse + Autoencode Patch A + Generate Patch B")

for epoch in range(1001):
    coarse, patch_A, target_B, locs = generate_data(batch_size=8)
    coarse, patch_A, target_B = coarse.to(device), patch_A.to(device), target_B.to(device)
    
    # We process batch one-by-one for the sparse locations (simplification for script)
    # In a real O-CNN, this would be a batched index lookup
    loss_total = 0
    loss_recon_A = 0
    loss_gen_B = 0
    
    optimizer.zero_grad()
    
    for i in range(8):
        loc_A = locs[i]['A']
        loc_B = locs[i]['B']
        
        # Forward Pass
        out_coarse, out_A, out_B = model(coarse[i:i+1], patch_A[i:i+1], loc_A, loc_B)
        
        # Losses
        l_glob = criterion(out_coarse, coarse[i:i+1])
        l_A = criterion(out_A, patch_A[i:i+1]) # Reconstruction Loss
        l_B = criterion(out_B, target_B[i:i+1]) # Generation Loss
        
        loss = l_glob + l_A + l_B
        loss.backward()
        
        loss_total += loss.item()
        loss_recon_A += l_A.item()
        loss_gen_B += l_B.item()

    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Total: {loss_total/8:.5f} | Recon A: {loss_recon_A/8:.5f} | Gen B: {loss_gen_B/8:.5f}")

# --- TESTING ---
print("\n--- FINAL EVALUATION (Comparison vs Bicubic) ---")
model.eval()
with torch.no_grad():
    coarse, patch_A, target_B, locs = generate_data(batch_size=4)
    coarse, patch_A, target_B = coarse.to(device), patch_A.to(device), target_B.to(device)
    
    for i in range(4):
        loc_A, loc_B = locs[i]['A'], locs[i]['B']
        out_coarse, out_A, out_B = model(coarse[i:i+1], patch_A[i:i+1], loc_A, loc_B)
        
        # Bicubic Baseline for B (The "Void")
        r, c = loc_B
        coarse_patch_B = coarse[i:i+1, :, r:r+10, c:c+10]
        bicubic_B = F.interpolate(coarse_patch_B, size=(40,40), mode='bicubic', align_corners=False)
        
        mse_neural = F.mse_loss(out_B, target_B[i:i+1]).item()
        mse_bicubic = F.mse_loss(bicubic_B, target_B[i:i+1]).item()
        
        print(f"Sample {i}:")
        print(f"  Patch A (Recon) MSE: {F.mse_loss(out_A, patch_A[i:i+1]).item():.6f}")
        print(f"  Patch B (Gen)   MSE: {mse_neural:.6f} vs Bicubic: {mse_bicubic:.6f}")
        if mse_neural < mse_bicubic:
            print(f"  >>> Neural is {mse_bicubic / mse_neural:.1f}x better than Bicubic")
        else:
            print(f"  >>> Neural failed to beat Bicubic")

# Save a plot for the last sample
r, c = locs[-1]['B']
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(coarse[-1, 0].cpu(), cmap='twilight')
axes[0].set_title("Input Coarse Grid")
axes[0].add_patch(plt.Rectangle((c, r), 10, 10, color='red', fill=False, lw=2))

axes[1].imshow(target_B[-1, 0].cpu(), cmap='twilight')
axes[1].set_title("Ground Truth Patch B")

axes[2].imshow(bicubic_B[0, 0].cpu(), cmap='twilight')
axes[2].set_title(f"Bicubic (MSE {mse_bicubic:.5f})")

axes[3].imshow(out_B[0, 0].cpu(), cmap='twilight')
axes[3].set_title(f"Neural Gen (MSE {mse_neural:.5f})")

plt.savefig('quadtree_generation_result.png')
print("\nSaved visualization to quadtree_generation_result.png")