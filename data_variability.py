import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the data generation function from orn.py
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
        hr_x = torch.linspace(x[j], x[j+patch_size-1], patch_size * 4)
        hr_y = torch.linspace(y[i], y[i+patch_size-1], patch_size * 4)
        hr_grid_y, hr_grid_x = torch.meshgrid(hr_y, hr_x, indexing='ij')
        term1_hr = torch.sin(2 * np.pi * k1 * hr_grid_x) * torch.sin(2 * np.pi * k2 * hr_grid_y)
        if is_mixture:
            term2_hr = torch.sin(2 * np.pi * k3 * hr_grid_x) * torch.sin(2 * np.pi * k4 * hr_grid_y)
            patch_fine_gt = term1_hr + term2_hr
        else:
            patch_fine_gt = term1_hr
        
        inputs.append(full_signal.unsqueeze(0))
        targets.append(patch_fine_gt.unsqueeze(0))
        coords_info.append((i, j, k1, k2, k3, k4, is_mixture))

    return torch.stack(inputs), torch.stack(targets), coords_info

# Generate multiple batches to show variability
print("Generating data to show variability...")
os.makedirs('plots', exist_ok=True)
inputs, targets, info = generate_sinusoidal_batch(batch_size=12)

# Create visualization
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Data Variability: Different Sinusoidal Patterns and ROI Locations', fontsize=16)

for idx in range(12):
    row = idx // 4
    col = idx % 4
    i, j, k1, k2, k3, k4, is_mixture = info[idx]
    
    # Show global signal with ROI marked
    im = axes[row, col].imshow(inputs[idx, 0].numpy(), cmap='twilight', vmin=-1, vmax=1)
    axes[row, col].add_patch(plt.Rectangle((j, i), 10, 10, color='red', fill=False, linewidth=2))
    if is_mixture:
        axes[row, col].set_title(f'k={k1:.1f},{k2:.1f} + {k3:.1f},{k4:.1f}\nROI: ({i},{j})')
    else:
        axes[row, col].set_title(f'k1={k1:.1f}, k2={k2:.1f}\nROI: ({i},{j})')
    axes[row, col].set_xlabel('x')
    axes[row, col].set_ylabel('y')
    plt.colorbar(im, ax=axes[row, col], shrink=0.6)

plt.tight_layout()
plt.savefig('plots/data_variability.png', dpi=150, bbox_inches='tight')
print("Data variability plot saved to plots/data_variability.png")

# Print statistics about variability
print(f"\nData Variability Statistics:")
print(f"Mixture ratio: {np.mean([i[6] for i in info]):.2f}")
print(f"Frequency k1 range: [{min([i[2] for i in info]):.2f}, {max([i[2] for i in info]):.2f}]")
print(f"Frequency k2 range: [{min([i[3] for i in info]):.2f}, {max([i[3] for i in info]):.2f}]")
print(f"Frequency k3 range (mixture only): [{min([i[4] for i in info if i[6]]):.2f}, {max([i[4] for i in info if i[6]]):.2f}]")
print(f"Frequency k4 range (mixture only): [{min([i[5] for i in info if i[6]]):.2f}, {max([i[5] for i in info if i[6]]):.2f}]")
print(f"ROI row range: [{min([i[0] for i in info]):.0f}, {max([i[0] for i in info]):.0f}]")
print(f"ROI col range: [{min([i[1] for i in info]):.0f}, {max([i[1] for i in info]):.0f}]")

# Show the actual patches being used for training
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle('Training Patches: Coarse (10x10) vs Ground Truth (40x40)', fontsize=16)

for idx in range(6):
    i, j, k1, k2, k3, k4, is_mixture = info[idx]
    
    # Coarse patch
    coarse_patch = inputs[idx, 0, i:i+10, j:j+10]
    axes[0, idx].imshow(coarse_patch.numpy(), cmap='twilight', vmin=-1, vmax=1)
    if is_mixture:
        axes[0, idx].set_title(f'Coarse 10x10\nk={k1:.1f},{k2:.1f} + {k3:.1f},{k4:.1f}')
    else:
        axes[0, idx].set_title(f'Coarse 10x10\nk1={k1:.1f}, k2={k2:.1f}')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].set_ylabel('y')
    
    # Ground truth patch
    gt_patch = targets[idx, 0]
    axes[1, idx].imshow(gt_patch.numpy(), cmap='twilight', vmin=-1, vmax=1)
    axes[1, idx].set_title(f'Ground Truth 40x40')
    axes[1, idx].set_xlabel('x')
    axes[1, idx].set_ylabel('y')

plt.tight_layout()
plt.savefig('plots/training_patches.png', dpi=150, bbox_inches='tight')
print("Training patches plot saved to plots/training_patches.png")

plt.show()
