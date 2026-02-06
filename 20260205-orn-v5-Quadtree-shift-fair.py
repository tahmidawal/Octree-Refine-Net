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
def generate_data(batch_size=8, include_full_fine=False):
    grid_size = 256
    fine_size = 1024  # 4x the coarse resolution
    patch_size = 10
    
    # Coarse coords
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    
    # Fine coords (for full 1024x1024 GT)
    x_fine = torch.linspace(0, 1, fine_size)
    y_fine = torch.linspace(0, 1, fine_size)
    gy_fine, gx_fine = torch.meshgrid(y_fine, x_fine, indexing='ij')
    
    inputs_coarse = []
    inputs_A_hr = []
    targets_B_hr = []
    full_fine_gts = []
    locs = []
    freq_params = []

    for _ in range(batch_size):
        # High Frequency Mix (Stress Test)
        k = np.random.uniform(15, 45, size=4)
        signal = torch.sin(2*np.pi*k[0]*gx) * torch.sin(2*np.pi*k[1]*gy)
        signal += torch.sin(2*np.pi*k[2]*gx) * torch.sin(2*np.pi*k[3]*gy)
        signal *= 0.5
        
        # Generate full fine resolution GT (1024x1024)
        if include_full_fine:
            fine_signal = torch.sin(2*np.pi*k[0]*gx_fine) * torch.sin(2*np.pi*k[1]*gy_fine)
            fine_signal += torch.sin(2*np.pi*k[2]*gx_fine) * torch.sin(2*np.pi*k[3]*gy_fine)
            fine_signal *= 0.5
            full_fine_gts.append(fine_signal.unsqueeze(0))

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
        freq_params.append(k)

    if include_full_fine:
        return (torch.stack(inputs_coarse), torch.stack(inputs_A_hr), 
                torch.stack(targets_B_hr), torch.stack(full_fine_gts), locs, freq_params)
    return torch.stack(inputs_coarse), torch.stack(inputs_A_hr), torch.stack(targets_B_hr), locs

# ==========================================
# 5. TRAINING & TESTING
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QuadTreeUNet().to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join('training_logs', run_timestamp)
plots_dir = os.path.join(run_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
log_path = os.path.join(run_dir, 'logs.txt')
log_f = open(log_path, 'w', encoding='utf-8')

def log_print(message):
    print(message)
    log_f.write(str(message) + '\n')
    log_f.flush()

log_print(f'Run directory: {run_dir}')
log_print(f'Model parameters | Total: {total_params:,} | Trainable: {trainable_params:,}')
log_print(f'Device: {device}')

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

log_print("Task: Reconstruct Coarse + Autoencode Patch A + Generate Patch B")
log_print("Starting training...")

loss_history = []
best_loss = float('inf')
best_model_path = os.path.join(run_dir, 'best_model.pt')

for epoch in range(1001):
    coarse, patch_A, target_B, locs = generate_data(batch_size=8)
    coarse, patch_A, target_B = coarse.to(device), patch_A.to(device), target_B.to(device)
    
    # We process batch one-by-one for the sparse locations (simplification for script)
    # In a real O-CNN, this would be a batched index lookup
    loss_total = 0
    loss_recon_A = 0
    loss_gen_B = 0
    loss_coarse = 0
    
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
        loss_coarse += l_glob.item()
        loss_recon_A += l_A.item()
        loss_gen_B += l_B.item()

    optimizer.step()
    
    avg_loss = loss_total / 8
    loss_history.append(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
    
    if epoch % 100 == 0:
        log_print(f"Epoch {epoch} | Total: {avg_loss:.5f} | Coarse: {loss_coarse/8:.5f} | Recon A: {loss_recon_A/8:.5f} | Gen B: {loss_gen_B/8:.5f}")

# --- SAVE LOSS CURVE ---
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
plt.close()
log_print(f"Saved loss curve to {plots_dir}/loss_curve.png")

# --- TESTING ---
log_print("\n--- FINAL EVALUATION (Comparison vs Bicubic) ---")
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
        
        log_print(f"Sample {i}:")
        log_print(f"  Patch A (Recon) MSE: {F.mse_loss(out_A, patch_A[i:i+1]).item():.6f}")
        log_print(f"  Patch B (Gen)   MSE: {mse_neural:.6f} vs Bicubic: {mse_bicubic:.6f}")
        if mse_neural < mse_bicubic:
            log_print(f"  >>> Neural is {mse_bicubic / mse_neural:.1f}x better than Bicubic")
        else:
            log_print(f"  >>> Neural failed to beat Bicubic")

# --- FAIR "PROJECTED ERROR TEST" (Same Scale Comparison) ---
log_print("\n" + "="*80)
log_print("FAIR COMPARISON: PROJECTED ERROR TEST (All errors measured at Fine Scale 40x40)")
log_print("="*80)
log_print("Comparing against TRUE HIGH-RES GROUND TRUTH to verify refinement adds real accuracy")
log_print("-"*80)
log_print(f"{'Sample':<8} {'Coarse->Fine MSE':<18} {'Neural Fine MSE':<18} {'Improvement':<15}")
log_print("-"*80)

with torch.no_grad():
    for i in range(4):
        loc_A, loc_B = locs[i]['A'], locs[i]['B']
        out_coarse, out_A, out_B = model(coarse[i:i+1], patch_A[i:i+1], loc_A, loc_B)
        
        # 1. Ground Truth High-Res Signal
        gt_fine = target_B[i:i+1]
        
        # 2. Neural Refinement (Fine Scale)
        pred_fine = out_B
        
        # 3. Coarse Representation Projected to Fine Scale
        # Take the MODEL'S coarse output and project it up to 40x40
        # This represents "What the coarse grid thinks the world looks like"
        r, c = loc_B
        coarse_patch = out_coarse[:, :, r:r+10, c:c+10]
        pred_coarse_projected = F.interpolate(coarse_patch, size=(40,40), mode='bicubic', align_corners=False)
        
        # 4. Calculate Errors on the SAME SCALE (Fine Scale)
        mse_neural_fine = F.mse_loss(pred_fine, gt_fine).item()
        mse_coarse_fine = F.mse_loss(pred_coarse_projected, gt_fine).item()
        
        improvement = mse_coarse_fine / mse_neural_fine if mse_neural_fine > 0 else float('inf')
        
        log_print(f"{i:<8} {mse_coarse_fine:<18.6f} {mse_neural_fine:<18.6f} {improvement:<15.1f}x")

log_print("-"*80)
log_print("Interpretation:")
log_print("  - Coarse->Fine MSE: Error when using upsampled coarse reconstruction")
log_print("  - Neural Fine MSE:  Error of the neural refinement")
log_print("  - Improvement > 1:  Neural refinement adds REAL accuracy over coarse")
log_print("="*80)

# --- REVERSE PROJECTION: Full 1024x1024 Fine GT Downsampled to Coarse Scale ---
log_print("\n" + "="*80)
log_print("REVERSE PROJECTION: Coarse Reconstruction vs Downsampled FULL 1024x1024 Fine GT")
log_print("="*80)
log_print("Generate TRUE 1024x1024 fine GT, downsample to 256x256, compare with coarse reconstruction")
log_print("-"*80)

# Generate test data WITH full fine resolution GT
coarse_eval, patch_A_eval, target_B_eval, full_fine_gt, locs_eval, freq_params = generate_data(
    batch_size=4, include_full_fine=True
)
coarse_eval = coarse_eval.to(device)
patch_A_eval = patch_A_eval.to(device)
target_B_eval = target_B_eval.to(device)
full_fine_gt = full_fine_gt.to(device)

log_print(f"{'Sample':<8} {'Full Coarse MSE':<18} {'Full Coarse RelL2':<20}")
log_print("-"*80)

with torch.no_grad():
    for i in range(4):
        loc_A, loc_B = locs_eval[i]['A'], locs_eval[i]['B']
        out_coarse, out_A, out_B = model(coarse_eval[i:i+1], patch_A_eval[i:i+1], loc_A, loc_B)
        
        # Downsample full 1024x1024 fine GT to 256x256 coarse resolution
        gt_fine_full = full_fine_gt[i:i+1]  # [1, 1, 1024, 1024]
        gt_coarse_from_fine = F.interpolate(gt_fine_full, size=(256, 256), mode='area')  # [1, 1, 256, 256]
        
        # Compare coarse reconstruction vs downsampled fine GT
        mse_full_coarse = F.mse_loss(out_coarse, gt_coarse_from_fine).item()
        rel_l2_full_coarse = torch.norm(out_coarse - gt_coarse_from_fine).item() / torch.norm(gt_coarse_from_fine).item()
        
        log_print(f"{i:<8} {mse_full_coarse:<18.6f} {rel_l2_full_coarse:<20.6f}")

log_print("-"*80)

# Now compare patches at both scales
log_print("\nPatch-level comparison (Fine GT downsampled to coarse 10x10):")
log_print(f"{'Sample':<8} {'Patch':<10} {'Coarse Recon MSE':<18} {'Coarse Recon RelL2':<20}")
log_print("-"*80)

with torch.no_grad():
    for i in range(4):
        loc_A, loc_B = locs_eval[i]['A'], locs_eval[i]['B']
        out_coarse, out_A, out_B = model(coarse_eval[i:i+1], patch_A_eval[i:i+1], loc_A, loc_B)
        
        # --- Patch A ---
        ra, ca = loc_A
        # Extract 40x40 region from 1024x1024 fine GT (coordinates scaled by 4)
        gt_A_from_full = full_fine_gt[i:i+1, :, ra*4:(ra+10)*4, ca*4:(ca+10)*4]  # [1, 1, 40, 40]
        gt_A_coarse = F.interpolate(gt_A_from_full, size=(10, 10), mode='area')  # [1, 1, 10, 10]
        pred_A_coarse = out_coarse[:, :, ra:ra+10, ca:ca+10]
        
        mse_A_coarse = F.mse_loss(pred_A_coarse, gt_A_coarse).item()
        rel_l2_A_coarse = torch.norm(pred_A_coarse - gt_A_coarse).item() / torch.norm(gt_A_coarse).item()
        
        log_print(f"{i:<8} {'A':<10} {mse_A_coarse:<18.6f} {rel_l2_A_coarse:<20.6f}")
        
        # --- Patch B ---
        rb, cb = loc_B
        gt_B_from_full = full_fine_gt[i:i+1, :, rb*4:(rb+10)*4, cb*4:(cb+10)*4]  # [1, 1, 40, 40]
        gt_B_coarse = F.interpolate(gt_B_from_full, size=(10, 10), mode='area')  # [1, 1, 10, 10]
        pred_B_coarse = out_coarse[:, :, rb:rb+10, cb:cb+10]
        
        mse_B_coarse = F.mse_loss(pred_B_coarse, gt_B_coarse).item()
        rel_l2_B_coarse = torch.norm(pred_B_coarse - gt_B_coarse).item() / torch.norm(gt_B_coarse).item()
        
        log_print(f"{i:<8} {'B':<10} {mse_B_coarse:<18.6f} {rel_l2_B_coarse:<20.6f}")

log_print("-"*80)
log_print("Interpretation:")
log_print("  - Full Coarse MSE: Error of 256x256 coarse reconstruction vs downsampled 1024x1024 fine GT")
log_print("  - Patch Coarse MSE: Error at specific patch locations")
log_print("  - This tests if coarse reconstruction matches what the TRUE fine signal looks like at coarse scale")
log_print("="*80)

# --- FAIR COMPARISON VISUALIZATION PLOT ---
log_print("\n--- GENERATING FAIR COMPARISON VISUALIZATION ---")

# Collect data for bar chart
fair_comparison_data = {
    'sample': [],
    'coarse_vs_fine_gt_mse': [],  # Coarse recon vs downsampled 1024 GT
    'neural_fine_mse': [],         # Neural refinement vs fine GT (Patch B)
    'bicubic_fine_mse': [],        # Bicubic vs fine GT (Patch B)
    'coarse_upsampled_mse': [],    # Coarse upsampled to fine vs fine GT (Patch B)
}

with torch.no_grad():
    for i in range(4):
        loc_A, loc_B = locs_eval[i]['A'], locs_eval[i]['B']
        rb, cb = loc_B
        
        out_coarse, out_A, out_B = model(coarse_eval[i:i+1], patch_A_eval[i:i+1], loc_A, loc_B)
        
        # 1. Coarse recon vs downsampled 1024x1024 GT
        gt_fine_full = full_fine_gt[i:i+1]
        gt_coarse_from_fine = F.interpolate(gt_fine_full, size=(256, 256), mode='area')
        mse_coarse_vs_fine_gt = F.mse_loss(out_coarse, gt_coarse_from_fine).item()
        
        # 2. Neural refinement vs fine GT (Patch B)
        mse_neural = F.mse_loss(out_B, target_B_eval[i:i+1]).item()
        
        # 3. Bicubic baseline vs fine GT (Patch B)
        coarse_patch_B = coarse_eval[i:i+1, :, rb:rb+10, cb:cb+10]
        bicubic_B = F.interpolate(coarse_patch_B, size=(40, 40), mode='bicubic', align_corners=False)
        mse_bicubic = F.mse_loss(bicubic_B, target_B_eval[i:i+1]).item()
        
        # 4. Coarse upsampled to fine vs fine GT (Patch B)
        coarse_patch_upsampled = F.interpolate(coarse_patch_B, size=(40, 40), mode='bicubic', align_corners=False)
        mse_coarse_upsampled = F.mse_loss(coarse_patch_upsampled, target_B_eval[i:i+1]).item()
        
        fair_comparison_data['sample'].append(i)
        fair_comparison_data['coarse_vs_fine_gt_mse'].append(mse_coarse_vs_fine_gt)
        fair_comparison_data['neural_fine_mse'].append(mse_neural)
        fair_comparison_data['bicubic_fine_mse'].append(mse_bicubic)
        fair_comparison_data['coarse_upsampled_mse'].append(mse_coarse_upsampled)

# Create comprehensive fair comparison figure
fig = plt.figure(figsize=(20, 16))

# --- Row 1: Bar chart comparison ---
ax1 = fig.add_subplot(3, 2, 1)
x = np.arange(4)
width = 0.2
bars1 = ax1.bar(x - 1.5*width, fair_comparison_data['coarse_vs_fine_gt_mse'], width, label='Coarse vs Downsampled Fine GT', color='#2ecc71')
bars2 = ax1.bar(x - 0.5*width, fair_comparison_data['coarse_upsampled_mse'], width, label='Coarse→Fine (Bicubic Upsample)', color='#e74c3c')
bars3 = ax1.bar(x + 0.5*width, fair_comparison_data['neural_fine_mse'], width, label='Neural Refinement', color='#3498db')
bars4 = ax1.bar(x + 1.5*width, fair_comparison_data['bicubic_fine_mse'], width, label='Bicubic Baseline', color='#9b59b6')
ax1.set_xlabel('Sample')
ax1.set_ylabel('MSE')
ax1.set_title('Fair Comparison: MSE at Different Scales')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Sample {i}' for i in range(4)])
ax1.legend(loc='upper right')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# --- Row 1: Improvement factors ---
ax2 = fig.add_subplot(3, 2, 2)
improvements = [fair_comparison_data['coarse_upsampled_mse'][i] / fair_comparison_data['neural_fine_mse'][i] 
                for i in range(4)]
colors = ['#27ae60' if imp > 1 else '#c0392b' for imp in improvements]
bars = ax2.bar(x, improvements, color=colors, edgecolor='black', linewidth=1.5)
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Improvement Factor (Coarse→Fine MSE / Neural MSE)')
ax2.set_title('Neural Refinement Improvement Over Upsampled Coarse')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Sample {i}' for i in range(4)])
for bar, imp in zip(bars, improvements):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{imp:.1f}x', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# --- Row 2-3: Visual comparison for one sample ---
sample_idx = 0
loc_A, loc_B = locs_eval[sample_idx]['A'], locs_eval[sample_idx]['B']
rb, cb = loc_B

with torch.no_grad():
    out_coarse, out_A, out_B = model(coarse_eval[sample_idx:sample_idx+1], patch_A_eval[sample_idx:sample_idx+1], loc_A, loc_B)
    
    # Get all the relevant data
    gt_fine_full = full_fine_gt[sample_idx, 0].cpu().numpy()
    gt_coarse_from_fine = F.interpolate(full_fine_gt[sample_idx:sample_idx+1], size=(256, 256), mode='area')[0, 0].cpu().numpy()
    pred_coarse = out_coarse[0, 0].cpu().numpy()
    
    # Patch B comparisons
    gt_B = target_B_eval[sample_idx, 0].cpu().numpy()
    pred_B = out_B[0, 0].cpu().numpy()
    coarse_patch_B = coarse_eval[sample_idx:sample_idx+1, :, rb:rb+10, cb:cb+10]
    bicubic_B = F.interpolate(coarse_patch_B, size=(40, 40), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()

# Row 2: Coarse scale comparison
ax3 = fig.add_subplot(3, 4, 5)
ax3.imshow(gt_coarse_from_fine, cmap='twilight', vmin=-1, vmax=1)
ax3.set_title('True Fine GT\n(1024→256 downsampled)')
ax3.axis('off')

ax4 = fig.add_subplot(3, 4, 6)
ax4.imshow(pred_coarse, cmap='twilight', vmin=-1, vmax=1)
ax4.set_title('Coarse Reconstruction\n(256x256)')
ax4.axis('off')

ax5 = fig.add_subplot(3, 4, 7)
error_coarse = np.abs(pred_coarse - gt_coarse_from_fine)
im5 = ax5.imshow(error_coarse, cmap='hot', vmin=0, vmax=0.2)
ax5.set_title(f'Coarse Error Map\nMSE: {np.mean(error_coarse**2):.6f}')
ax5.axis('off')
plt.colorbar(im5, ax=ax5, fraction=0.046)

ax6 = fig.add_subplot(3, 4, 8)
ax6.imshow(gt_fine_full, cmap='twilight', vmin=-1, vmax=1)
ax6.set_title('True Fine GT\n(1024x1024)')
ax6.axis('off')

# Row 3: Fine scale (Patch B) comparison
ax7 = fig.add_subplot(3, 4, 9)
ax7.imshow(gt_B, cmap='twilight', vmin=-1, vmax=1)
ax7.set_title('Patch B Ground Truth\n(40x40 Fine)')
ax7.axis('off')

ax8 = fig.add_subplot(3, 4, 10)
ax8.imshow(bicubic_B, cmap='twilight', vmin=-1, vmax=1)
ax8.set_title(f'Bicubic Upsample\nMSE: {np.mean((bicubic_B - gt_B)**2):.6f}')
ax8.axis('off')

ax9 = fig.add_subplot(3, 4, 11)
ax9.imshow(pred_B, cmap='twilight', vmin=-1, vmax=1)
ax9.set_title(f'Neural Refinement\nMSE: {np.mean((pred_B - gt_B)**2):.6f}')
ax9.axis('off')

ax10 = fig.add_subplot(3, 4, 12)
error_neural = np.abs(pred_B - gt_B)
error_bicubic = np.abs(bicubic_B - gt_B)
# Show side-by-side error comparison
combined_error = np.concatenate([error_bicubic, error_neural], axis=1)
im10 = ax10.imshow(combined_error, cmap='hot', vmin=0, vmax=0.2)
ax10.axvline(x=40, color='white', linewidth=2)
ax10.set_title('Error: Bicubic (L) vs Neural (R)')
ax10.axis('off')
plt.colorbar(im10, ax=ax10, fraction=0.046)

plt.suptitle('Fair Comparison: Coarse vs Fine Scale Evaluation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fair_comparison_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
log_print(f"Saved fair comparison visualization to {plots_dir}/fair_comparison_visualization.png")

# --- FULL QUADTREE ERROR PLOTS FOR MULTIPLE EXAMPLES ---
log_print("\n--- GENERATING FULL QUADTREE ERROR PLOTS ---")
log_print("\n--- RELATIVE L2 ERROR SUMMARY ---")
log_print(f"{'Sample':<8} {'Coarse RelL2':<15} {'Patch A RelL2':<15} {'Patch B RelL2':<15} {'Patch B Bicubic RelL2':<20}")
log_print("-" * 75)

with torch.no_grad():
    for idx in range(4):
        loc_A, loc_B = locs[idx]['A'], locs[idx]['B']
        ra, ca = loc_A
        rb, cb = loc_B
        
        out_coarse, out_A, out_B = model(coarse[idx:idx+1], patch_A[idx:idx+1], loc_A, loc_B)
        
        # Bicubic baseline for this sample
        coarse_patch_B = coarse[idx:idx+1, :, rb:rb+10, cb:cb+10]
        bicubic_B = F.interpolate(coarse_patch_B, size=(40,40), mode='bicubic', align_corners=False)
        
        # The coarse grid IS the ground truth at depth 8
        gt_coarse = coarse[idx, 0].cpu().numpy()
        pred_coarse = out_coarse[0, 0].cpu().numpy()
        
        # Compute error map
        error_coarse = np.abs(pred_coarse - gt_coarse)
        
        # Patch A
        gt_A = patch_A[idx, 0].cpu().numpy()
        pred_A = out_A[0, 0].cpu().numpy()
        error_A = np.abs(pred_A - gt_A)
        
        # Patch B
        gt_B = target_B[idx, 0].cpu().numpy()
        pred_B = out_B[0, 0].cpu().numpy()
        error_B = np.abs(pred_B - gt_B)
        
        # Bicubic baseline error for B
        bicubic_B_np = bicubic_B[0, 0].cpu().numpy()
        error_bicubic = np.abs(bicubic_B_np - gt_B)
        
        # Compute Relative L2 errors: ||pred - gt||_2 / ||gt||_2
        rel_l2_coarse = np.linalg.norm(pred_coarse - gt_coarse) / np.linalg.norm(gt_coarse)
        rel_l2_A = np.linalg.norm(pred_A - gt_A) / np.linalg.norm(gt_A)
        rel_l2_B = np.linalg.norm(pred_B - gt_B) / np.linalg.norm(gt_B)
        rel_l2_bicubic = np.linalg.norm(bicubic_B_np - gt_B) / np.linalg.norm(gt_B)
        
        log_print(f"{idx:<8} {rel_l2_coarse:<15.6f} {rel_l2_A:<15.6f} {rel_l2_B:<15.6f} {rel_l2_bicubic:<20.6f}")
        
        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Coarse level comparison
        im0 = axes[0, 0].imshow(gt_coarse, cmap='twilight')
        axes[0, 0].set_title("Ground Truth (Coarse D8)")
        axes[0, 0].add_patch(plt.Rectangle((ca, ra), 10, 10, color='lime', fill=False, lw=2, label='Patch A'))
        axes[0, 0].add_patch(plt.Rectangle((cb, rb), 10, 10, color='red', fill=False, lw=2, label='Patch B'))
        axes[0, 0].legend(loc='upper right')
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].imshow(pred_coarse, cmap='twilight')
        axes[0, 1].set_title("Reconstructed Coarse")
        axes[0, 1].add_patch(plt.Rectangle((ca, ra), 10, 10, color='lime', fill=False, lw=2))
        axes[0, 1].add_patch(plt.Rectangle((cb, rb), 10, 10, color='red', fill=False, lw=2))
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[0, 2].imshow(error_coarse, cmap='hot')
        axes[0, 2].set_title(f"Coarse Error (RelL2: {rel_l2_coarse:.6f})")
        axes[0, 2].add_patch(plt.Rectangle((ca, ra), 10, 10, color='lime', fill=False, lw=2))
        axes[0, 2].add_patch(plt.Rectangle((cb, rb), 10, 10, color='cyan', fill=False, lw=2))
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Row 2: High-res patch comparisons
        im3 = axes[1, 0].imshow(error_A, cmap='hot')
        axes[1, 0].set_title(f"Patch A Error (RelL2: {rel_l2_A:.6f})")
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(error_B, cmap='hot')
        axes[1, 1].set_title(f"Patch B Neural Error (RelL2: {rel_l2_B:.6f})")
        plt.colorbar(im4, ax=axes[1, 1])
        
        im5 = axes[1, 2].imshow(error_bicubic, cmap='hot')
        axes[1, 2].set_title(f"Patch B Bicubic Error (RelL2: {rel_l2_bicubic:.6f})")
        plt.colorbar(im5, ax=axes[1, 2])
        
        plt.suptitle(f'Sample {idx}: Full Quadtree Error Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'full_quadtree_error_sample{idx}.png'), dpi=150)
        plt.close()
    
    log_print("-" * 75)
    log_print(f"Saved full quadtree error plots for 4 samples to {plots_dir}/")

log_f.close()
log_print = print  # Reset to avoid errors after close
print(f"\nTraining complete. All outputs saved to {run_dir}")