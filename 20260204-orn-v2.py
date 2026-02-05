import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 1. FIXED COORDINATE ENGINE
# ==========================================
def xy2key(x, y, depth=8):
    """Encodes X (cols) and Y (rows) into Morton Key."""
    key = torch.zeros_like(x, dtype=torch.long)
    for i in range(depth):
        key |= ((x >> i) & 1) << (2 * i + 1) # Odd bits: X (cols)
        key |= ((y >> i) & 1) << (2 * i)     # Even bits: Y (rows)
    return key

def key2xy(key, depth=8):
    """Decodes Morton Key into X (cols) and Y (rows)."""
    y = torch.zeros_like(key, dtype=torch.long)
    x = torch.zeros_like(key, dtype=torch.long)
    for i in range(depth):
        y |= ((key >> (2 * i)) & 1) << i
        x |= ((key >> (2 * i + 1)) & 1) << i
    return x, y  # Return X, Y
    


    

'''
2. Octree Alignment (The "Ancestry" Lookup)This is the most powerful part of the O-CNN logic. 
It allows a high-res node at Depth 10 to "reach back" and grab features from its parent at Depth 8.
How it works: In a Morton-coded quadtree, the key of a parent is always just the key of the child shifted
 right by 2 bits (in 2D).Child at $(400, 400)$ Depth 10.Parent is at $(100, 100)$ Depth 8.Parent_Key = 
 Child_Key >> (2 * (10 - 8))Python
 
 def octree_align(child_keys, parent_keys_list, parent_features, child_depth, parent_depth):
    """
    Aligns high-res child nodes with their coarse-res parents.
    """
    # 1. Calculate what the parent key address should be
    diff = child_depth - parent_depth
    target_parent_keys = child_keys >> (2 * diff)
    
    # 2. Find where those parent keys sit in the parent_keys_list
    # This is O(log N) efficiency
    indices = torch.searchsorted(parent_keys_list, target_parent_keys)
    
    # 3. Pull parent features for every child
    # Now every high-res pixel has its global context 'pre-attached'
    aligned_features = parent_features[indices]
    return aligned_features
'''

# ==========================================
# 3. OCTREE ALIGNMENT (THE "ANCESTRY" LOOKUP)
# ==========================================


# As we have the keys we can align the features from the child to the parent
# This is one of the key features of the octree alignment. Since the child can just geet the parents keys by shifting the keys ;by 2 to the right. This 
# Can be used in order to align the features from the child to the parent.

def octree_align(child_keys, parent_keys_list, parent_features, child_depth, parent_depth):
    diff = child_depth - parent_depth
    target_parent_keys = child_keys >> (2 * diff)
    indices = torch.searchsorted(parent_keys_list, target_parent_keys)
    aligned_features = parent_features[indices]
    return aligned_features


'''

3. Integrated Attention (Local + Surround + Global)With the keys and alignment ready, we can build a 
Sparse Attention Block. This ensures the $40 \times 40$ reconstruction isn't just "guessing" but is actively 
looking at its surroundings.The Strategy:Query ($Q$): The $10 \times 10$ patch features.Key/Value ($K, V$): 
The "Halo" nodes (Surround) + the "Ancestry" features (Global context from lower depths).Positional Encoding: 
Use the Morton Key itself. It is a unique 1D signature of 2D space.Python

class SparseRefinementAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)

    def forward(self, roi_feats, halo_feats, global_context):
        # roi_feats: (Batch, 100, Dim) -> The patch to refine
        # halo_feats: (Batch, N_halo, Dim) -> Spatial continuity
        # global_context: (Batch, 1, Dim) -> Frequency/Phase context
        
        # Combine Halo and Global context as the 'Knowledge Base'
        context = torch.cat([halo_feats, global_context], dim=1)
        
        # Cross-attention: Patch looks at neighbors and global field
        # to decide how to fill in the high-res sine wave
        refined_feats, _ = self.attn(query=roi_feats, key=context, value=context)
        
        return refined_feats + roi_feats # Residual connection

'''


class SparseRefinementAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)

    def forward(self, roi_feats, halo_feats, global_context):
        # roi_feats: (Batch, 100, Dim) -> patch to refine
        # halo_feats: (Batch, N_halo, Dim) -> Spatial continuity
        # global_context: (Batch, 1, Dim) -> Frequency/Phase context
        
        context = torch.cat([halo_feats, global_context], dim=1)
        
        # Cross-attention: Patch looks at neighbors and global field
        # to decide how to fill in the high-res sine wave
        refined_feats, _ = self.attn(query=roi_feats, key=context, value=context)
        
        return refined_feats + roi_feats # Residual connection
        



# ===================

# X_mod: Halo Key Generation
 
# The goal is that instead of searching around using indices we will use the morton encoding to find the neighbors. 
# The assumption is that we will be able to do so very effectively using the morton encoding. 
# ===================

def get_halo_neighbors(patch_key, depth):

    # For example a purpose can be that given a specific patch lets  find the keys of the ring around it. Like give it in 10x10 and then we get out 12x12 

    rows, cols = key2xy(patch_key, depth)

    # Lets include the search space to be 3x3 aroudn every pixel in the patch which should get the job done 
    offsets = torch.tensor([-1, 0, 1])
    off_y, off_x = torch.meshgrid(offsets, offsets, indexing='ij')

    # Generate all the potential neighbor coordinates which would give us the coordinates 
    neigh_rows = rows.view(-1, 1) + off_y.flatten()
    neigh_cols = cols.view(-1, 1) + off_x.flatten()

    # Re-encode to keys
    neighbor_keys = xy2key(neigh_rows, neigh_cols, depth)
    return torch.unique(neighbor_keys)

# ==========================================
# 2. DATA GENERATION
# ==========================================
def generate_sinusoidal_batch(batch_size=8, grid_size=256, patch_size=10):
    '''Generates coarse grids and high-res ground truth patches.'''
    # Coarse coordinates [0, 255]
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    inputs = []
    targets = []
    coords_info = []

    for _ in range(batch_size):
        # Varying frequencies
        k1, k2 = np.random.uniform(1, 10, size=2)
        term1 = torch.sin(2 * np.pi * k1 * grid_x) * torch.sin(2 * np.pi * k2 * grid_y)
        is_mixture = np.random.rand() < 0.7
        if is_mixture:
            k3, k4 = np.random.uniform(1, 10, size=2)
            term2 = torch.sin(2 * np.pi * k3 * grid_x) * torch.sin(2 * np.pi * k4 * grid_y)
            full_signal = term1 + term2
        else:
            k3, k4 = None, None
            full_signal = term1
        
        # Select random ROI center
        i = np.random.randint(patch_size, grid_size - patch_size)
        j = np.random.randint(patch_size, grid_size - patch_size)
        
        # Patch at depth 8 (10x10)
        patch_coarse = full_signal[i:i+patch_size, j:j+patch_size]
        
        # Ground Truth at depth 10 (4x resolution -> 40x40)
        # We generate the true high-res signal for that spatial ROI
        hr_x = torch.linspace(x[j], x[j+patch_size-1], patch_size * 4)
        hr_y = torch.linspace(y[i], y[i+patch_size-1], patch_size * 4)
        hr_grid_y, hr_grid_x = torch.meshgrid(hr_y, hr_x, indexing='ij')
        term1_hr = torch.sin(2 * np.pi * k1 * hr_grid_x) * torch.sin(2 * np.pi * k2 * hr_grid_y)
        if is_mixture:
            term2_hr = torch.sin(2 * np.pi * k3 * hr_grid_x) * torch.sin(2 * np.pi * k4 * hr_grid_y)
            patch_fine_gt = term1_hr + term2_hr
        else:
            patch_fine_gt = term1_hr
        
        inputs.append(full_signal.unsqueeze(0)) # Global context
        targets.append(patch_fine_gt.unsqueeze(0))
        coords_info.append((i, j, k1, k2, k3, k4, is_mixture))

    return torch.stack(inputs), torch.stack(targets), coords_info

# ==========================================
# 3. MODEL DEFINITION (RefinerNetV2)
# ==========================================
class RefinerNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Global Context Encoder
        # Processes the full 256x256 grid to understand the global wave phase
        self.global_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), # Depth 8 features
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 2. Refinement Head (High Capacity)
        self.refiner = nn.Sequential(
            # Input: 32 channels + 2 Coord channels (Positional Encoding)
            nn.ConvTranspose2d(32 + 2, 32, kernel_size=4, stride=2, padding=1), # 12 -> 24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),     # 24 -> 48
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, row, col, patch_size=10):
        # 1. Global Feature Extraction
        global_feats = self.global_enc(x)
        
        # 2. "Halo" Extraction (The Structural Fix)
        # Instead of slicing [0:10], we slice [-1:11] (12x12)
        # This provides real context for the boundaries.
        pad = 1
        r_start = max(0, row - pad)
        r_end = min(x.shape[2], row + patch_size + pad)
        c_start = max(0, col - pad)
        c_end = min(x.shape[3], col + patch_size + pad)
        
        # Safe slicing (handles image edges)
        roi_feats = global_feats[:, :, r_start:r_end, c_start:c_end]
        
        # Handle edge cases (if ROI is at 0,0, pad manually to ensure 12x12)
        if roi_feats.shape[2] != patch_size + 2*pad or roi_feats.shape[3] != patch_size + 2*pad:
            diff_h = (patch_size + 2*pad) - roi_feats.shape[2]
            diff_w = (patch_size + 2*pad) - roi_feats.shape[3]
            roi_feats = F.pad(roi_feats, (0, diff_w, 0, diff_h))

        # 3. Explicit Positional Encoding (The Phase Fix)
        # We append normalized coordinates to the features so the network 
        # knows *exactly* where this patch is in the sine wave.
        B, C, H, W = roi_feats.shape
        y_coords = torch.linspace(r_start, r_end, H, device=x.device) / 256.0
        x_coords = torch.linspace(c_start, c_end, W, device=x.device) / 256.0
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack coordinates onto features: [B, 32+2, 12, 12]
        pos_encoding = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        roi_input = torch.cat([roi_feats, pos_encoding], dim=1)

        # 4. Refine (12x12 -> 48x48)
        refined_expanded = self.refiner(roi_input)
        
        # 5. Crop to Center (Remove the halo/padding artifacts)
        # We generated 48x48, but valid data is the center 40x40
        out_pad = pad * 4 # Resolution increased by 4x
        valid_out = refined_expanded[:, :, out_pad:-out_pad, out_pad:-out_pad]
        
        return valid_out

# ==========================================
# 4. TRAINING LOOP
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RefinerNetV2().to(device)
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
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

log_print('Starting training...')
loss_history = []
best_loss = float('inf')
best_model_path = os.path.join(run_dir, 'best_model.pt')
for epoch in range(501):
    inputs, targets, info = generate_sinusoidal_batch(batch_size=16)
    i, j, _, _, _, _, _ = info[0] # Using first sample coords for simplicity in slicing

    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optimizer.zero_grad()
    # To keep batch training simple, we'll use a fixed-ish ROI per batch 
    # or loop through batch. For this demo, we assume random ROIs.
    total_loss = 0
    for b in range(inputs.shape[0]):
        row, col = info[b][0], info[b][1]
        pred = model(inputs[b:b+1], row, col)
        total_loss += criterion(pred, targets[b:b+1])
    
    total_loss /= inputs.shape[0]
    total_loss.backward()
    optimizer.step()
    loss_value = total_loss.item()
    loss_history.append(loss_value)
    if loss_value < best_loss:
        best_loss = loss_value
        torch.save(model.state_dict(), best_model_path)
    
    if epoch % 100 == 0:
        log_print(f'Epoch {epoch} | Loss: {loss_value:.6f}')

log_print(f'Best loss: {best_loss:.6f}')
log_print(f'Best model saved to: {best_model_path}')

model.load_state_dict(torch.load(best_model_path, map_location=device))
log_print('Loaded best model for evaluation.')

fig = plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_plot_path = os.path.join(plots_dir, 'loss_vs_epoch.png')
plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
log_print(f'Loss curve saved to {loss_plot_path}')

# ==========================================
# 5. EVALUATION & PLOTTING
# ==========================================
model.eval()
examples = 4
test_in, test_gt, test_info = generate_sinusoidal_batch(batch_size=examples)
test_in = test_in.to(device)
test_gt = test_gt.to(device)

with torch.no_grad():
    for ex in range(examples):
        row, col, k1, k2, k3, k4, is_mixture = test_info[ex]
        pred = model(test_in[ex:ex+1], row, col).squeeze().detach().cpu().numpy()
        gt = test_gt[ex].squeeze().detach().cpu().numpy()

        coarse_patch = test_in[ex, 0, row:row+10, col:col+10].unsqueeze(0).unsqueeze(0)
        bicubic_upsampled = nn.functional.interpolate(coarse_patch, size=(40, 40), mode='bicubic', align_corners=False).squeeze().detach().cpu().numpy()

        error_map = np.abs(gt - pred)
        mse_pred = float(np.mean((gt - pred) ** 2))
        mse_bicubic = float(np.mean((gt - bicubic_upsampled) ** 2))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].imshow(test_in[ex, 0].detach().cpu().numpy(), cmap='twilight')
        if is_mixture:
            axes[0, 0].set_title(f'Global Input (k={k1:.1f},{k2:.1f} + {k3:.1f},{k4:.1f})')
        else:
            axes[0, 0].set_title(f'Global Input (k={k1:.1f}, {k2:.1f})')
        axes[0, 0].add_patch(plt.Rectangle((col, row), 10, 10, color='red', fill=False, linewidth=2))
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')

        coarse_zoom = test_in[ex, 0, row:row+10, col:col+10].detach().cpu().numpy()
        axes[0, 1].imshow(coarse_zoom, cmap='twilight')
        axes[0, 1].set_title('Coarse 10x10 Patch')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')

        axes[0, 2].imshow(gt, cmap='twilight')
        axes[0, 2].set_title('Ground Truth 40x40')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('y')

        axes[1, 0].imshow(pred, cmap='twilight')
        axes[1, 0].set_title(f'Neural Prediction\nMSE: {mse_pred:.6f}')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')

        axes[1, 1].imshow(bicubic_upsampled, cmap='twilight')
        axes[1, 1].set_title(f'Bicubic\nMSE: {mse_bicubic:.6f}')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')

        axes[1, 2].imshow(error_map, cmap='hot')
        axes[1, 2].set_title('Absolute Error (GT - Neural)')
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('y')

        for ax in axes.flat:
            plt.colorbar(ax.images[0], ax=ax, shrink=0.8)

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f'example_{ex:02d}_comparison.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log_print(f'Example {ex} plot saved to {out_path} | Neural MSE: {mse_pred:.6f} | Bicubic MSE: {mse_bicubic:.6f}')

log_print('Finished evaluation.')
log_f.close()