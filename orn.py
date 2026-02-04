import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 1. MORTON ENCODING (SHUFFLED KEYS)
# ==========================================
def xy2key(x, y, depth):
    '''Interleave bits of x and y to create a Morton Key.'''
    key = torch.zeros_like(x, dtype=torch.long)
    for i in range(depth):
        key |= ((x >> i) & 1) << (2 * i + 1)
        key |= ((y >> i) & 1) << (2 * i)
    return key

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
# 3. MODEL DEFINITION
# ==========================================
class RefinerNet(nn.Module):
    def __init__(self):
        super(RefinerNet, self).__init__()
        # Global context encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Refinement upsampler (Sparse logic: only processes the patch)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 10->20
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 20->40
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, i, j, patch_size=10):
        # 1. Extract Global Features
        features = self.encoder(x)
        
        # 2. Extract ROI Local features (Like octree_split logic)
        # In a real Quadtree, we'd use Morton keys to find these indices.
        # Here we use spatial slicing for clarity.
        roi_features = features[:, :, i:i+patch_size, j:j+patch_size]
        
        # 3. Predict finer children
        refined_patch = self.upsampler(roi_features)
        return refined_patch

# ==========================================
# 4. TRAINING LOOP
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RefinerNet().to(device)
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