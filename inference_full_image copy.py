"""
Full Image Inference Script for ResidualRefiner

This script loads a trained ResidualRefiner model and runs inference on every 10x10 patch
in a 256x256 image (in parallel batches), stitching the 40x40 outputs to create a full
1024x1024 (4x upsampled) image. Compares against ground truth and bicubic upsampling.

Usage:
    python inference_full_image.py --model_path training_logs/YYYYMMDD_HHMMSS/best_model.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

# ==========================================
# 1. FOURIER FEATURE ENCODING
# ==========================================
class FourierEmbedding(nn.Module):
    def __init__(self, in_dims=2, embed_dim=64, scale=30.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dims, embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = (2 * np.pi * x.permute(0, 2, 3, 1)) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).permute(0, 3, 1, 2)

# ==========================================
# 2. RESIDUAL REFINER NETWORK
# ==========================================
class ResidualRefiner(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        
        self.pos_enc = FourierEmbedding(in_dims=2, embed_dim=feat_dim)

        self.global_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, feat_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1)
        )

        self.refiner_head = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x, row, col, patch_size=10):
        d6_feats = self.global_enc(x)
        d8_feats = F.interpolate(d6_feats, size=(256, 256), mode='bilinear', align_corners=False)
        
        pad = 1
        r_start = max(0, row - pad)
        r_end = min(x.shape[2], row + patch_size + pad)
        c_start = max(0, col - pad)
        c_end = min(x.shape[3], col + patch_size + pad)
        
        roi_feats = d8_feats[:, :, r_start:r_end, c_start:c_end]
        
        target_h, target_w = patch_size + 2*pad, patch_size + 2*pad
        if roi_feats.shape[2] != target_h or roi_feats.shape[3] != target_w:
            diff_h = target_h - roi_feats.shape[2]
            diff_w = target_w - roi_feats.shape[3]
            roi_feats = F.pad(roi_feats, (0, diff_w, 0, diff_h))

        B, _, H, W = roi_feats.shape
        y_grid = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, -1, -1, W)
        x_grid = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, -1, H, -1)
        grid = torch.cat([x_grid, y_grid], dim=1)
        
        pos_feats = self.pos_enc(grid)
        
        combined = torch.cat([roi_feats, pos_feats], dim=1)
        residual = self.refiner_head(combined)
        
        out_pad = pad * 4 
        valid_residual = residual[:, :, out_pad:-out_pad, out_pad:-out_pad]
        
        return valid_residual

# ==========================================
# 3. BATCHED INFERENCE FOR FULL IMAGE
# ==========================================
def run_full_image_inference(model, input_image, device, patch_size=10, batch_size=64):
    """
    Run inference on all non-overlapping 10x10 patches and stitch to 4x resolution.
    
    Input: 256x256 image
    Output: 1024x1024 image (each 10x10 -> 40x40)
    
    Grid: 256/10 = 25.6 -> We'll use 25 patches (covering 250x250), 
          then handle the edge separately or pad.
    For simplicity, we'll process a 250x250 region (25x25 patches).
    """
    model.eval()
    
    H, W = input_image.shape[2], input_image.shape[3]
    n_patches_h = H // patch_size  # 25
    n_patches_w = W // patch_size  # 25
    
    output_h = n_patches_h * patch_size * 4  # 1000
    output_w = n_patches_w * patch_size * 4  # 1000
    
    # Prepare output tensor
    output_image = torch.zeros(1, 1, output_h, output_w, device=device)
    
    # Collect all patch coordinates
    patch_coords = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            row = i * patch_size
            col = j * patch_size
            patch_coords.append((row, col, i, j))
    
    # Process in batches
    with torch.no_grad():
        for batch_start in range(0, len(patch_coords), batch_size):
            batch_coords = patch_coords[batch_start:batch_start + batch_size]
            
            for row, col, i, j in batch_coords:
                # Get residual prediction
                residual = model(input_image, row, col, patch_size)
                
                # Get bicubic baseline
                coarse_patch = input_image[:, :, row:row+patch_size, col:col+patch_size]
                bicubic_base = F.interpolate(coarse_patch, size=(40, 40), mode='bicubic', align_corners=False)
                
                # Final = bicubic + residual
                final_patch = bicubic_base + residual
                
                # Place in output
                out_row = i * patch_size * 4
                out_col = j * patch_size * 4
                output_image[:, :, out_row:out_row+40, out_col:out_col+40] = final_patch
    
    return output_image

# ==========================================
# 4. GENERATE TEST SIGNAL WITH GROUND TRUTH
# ==========================================
def generate_test_signal(grid_size=256, k1=15, k2=20, k3=None, k4=None):
    """Generate a sinusoidal test signal at both low-res and high-res."""
    # Low-res (256x256)
    x_lr = torch.linspace(0, 1, grid_size)
    y_lr = torch.linspace(0, 1, grid_size)
    grid_y_lr, grid_x_lr = torch.meshgrid(y_lr, x_lr, indexing='ij')
    
    signal_lr = torch.sin(2 * np.pi * k1 * grid_x_lr) * torch.sin(2 * np.pi * k2 * grid_y_lr)
    if k3 is not None and k4 is not None:
        signal_lr += torch.sin(2 * np.pi * k3 * grid_x_lr) * torch.sin(2 * np.pi * k4 * grid_y_lr)
        signal_lr /= 2.0
    
    # High-res ground truth (1000x1000 to match 25x25 patches * 40)
    hr_size = (grid_size // 10) * 40  # 1000
    x_hr = torch.linspace(0, (grid_size // 10) * 10 / grid_size, hr_size)  # 0 to 250/256
    y_hr = torch.linspace(0, (grid_size // 10) * 10 / grid_size, hr_size)
    grid_y_hr, grid_x_hr = torch.meshgrid(y_hr, x_hr, indexing='ij')
    
    signal_hr = torch.sin(2 * np.pi * k1 * grid_x_hr) * torch.sin(2 * np.pi * k2 * grid_y_hr)
    if k3 is not None and k4 is not None:
        signal_hr += torch.sin(2 * np.pi * k3 * grid_x_hr) * torch.sin(2 * np.pi * k4 * grid_y_hr)
        signal_hr /= 2.0
    
    return signal_lr, signal_hr, (k1, k2, k3, k4)

# ==========================================
# 5. MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Full image inference with ResidualRefiner')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--k1', type=float, default=15.0, help='Frequency k1')
    parser.add_argument('--k2', type=float, default=20.0, help='Frequency k2')
    parser.add_argument('--k3', type=float, default=None, help='Frequency k3 (optional mixture)')
    parser.add_argument('--k4', type=float, default=None, help='Frequency k4 (optional mixture)')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Load model
    print(f'Loading model from: {args.model_path}')
    model = ResidualRefiner().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print('Model loaded successfully.')
    
    # Generate test signal
    print(f'Generating test signal with k1={args.k1}, k2={args.k2}, k3={args.k3}, k4={args.k4}')
    signal_lr, signal_hr_gt, freqs = generate_test_signal(
        k1=args.k1, k2=args.k2, k3=args.k3, k4=args.k4
    )
    
    input_image = signal_lr.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 256, 256]
    
    # Run full image inference
    print('Running full image inference (25x25 = 625 patches)...')
    neural_output = run_full_image_inference(model, input_image, device)
    neural_output_np = neural_output.squeeze().cpu().numpy()
    
    # Bicubic baseline for comparison (full image)
    # Crop input to 250x250 to match
    input_cropped = input_image[:, :, :250, :250]
    bicubic_full = F.interpolate(input_cropped, size=(1000, 1000), mode='bicubic', align_corners=False)
    bicubic_full_np = bicubic_full.squeeze().cpu().numpy()
    
    # Ground truth
    gt_np = signal_hr_gt.numpy()
    
    # Compute metrics
    mse_neural = float(np.mean((gt_np - neural_output_np) ** 2))
    mse_bicubic = float(np.mean((gt_np - bicubic_full_np) ** 2))
    improvement = ((mse_bicubic - mse_neural) / mse_bicubic) * 100 if mse_bicubic > 0 else 0
    
    print(f'\n=== Results ===')
    print(f'Neural MSE:  {mse_neural:.8f}')
    print(f'Bicubic MSE: {mse_bicubic:.8f}')
    print(f'Improvement: {improvement:+.2f}%')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Row 1: Full images
    axes[0, 0].imshow(signal_lr.numpy(), cmap='twilight')
    axes[0, 0].set_title(f'Input 256x256\n(k1={args.k1}, k2={args.k2})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_np, cmap='twilight')
    axes[0, 1].set_title('Ground Truth 1000x1000')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(neural_output_np, cmap='twilight')
    axes[0, 2].set_title(f'Neural Output 1000x1000\nMSE: {mse_neural:.6f}')
    axes[0, 2].axis('off')
    
    # Row 2: Bicubic and error maps
    axes[1, 0].imshow(bicubic_full_np, cmap='twilight')
    axes[1, 0].set_title(f'Bicubic 1000x1000\nMSE: {mse_bicubic:.6f}')
    axes[1, 0].axis('off')
    
    error_neural = np.abs(gt_np - neural_output_np)
    axes[1, 1].imshow(error_neural, cmap='hot', vmin=0, vmax=error_neural.max())
    axes[1, 1].set_title(f'Neural Error Map\nMax: {error_neural.max():.4f}')
    axes[1, 1].axis('off')
    
    error_bicubic = np.abs(gt_np - bicubic_full_np)
    axes[1, 2].imshow(error_bicubic, cmap='hot', vmin=0, vmax=error_bicubic.max())
    axes[1, 2].set_title(f'Bicubic Error Map\nMax: {error_bicubic.max():.4f}')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Full Image 4x Upsampling Comparison | Improvement: {improvement:+.2f}%', fontsize=14)
    plt.tight_layout()
    
    out_path = os.path.join(args.output_dir, f'full_image_comparison_{timestamp}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nComparison plot saved to: {out_path}')
    
    # Save a zoomed comparison (center 200x200 region)
    zoom_start = 400
    zoom_end = 600
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    axes2[0].imshow(gt_np[zoom_start:zoom_end, zoom_start:zoom_end], cmap='twilight')
    axes2[0].set_title('Ground Truth (Zoomed)')
    axes2[0].axis('off')
    
    axes2[1].imshow(neural_output_np[zoom_start:zoom_end, zoom_start:zoom_end], cmap='twilight')
    axes2[1].set_title('Neural (Zoomed)')
    axes2[1].axis('off')
    
    axes2[2].imshow(bicubic_full_np[zoom_start:zoom_end, zoom_start:zoom_end], cmap='twilight')
    axes2[2].set_title('Bicubic (Zoomed)')
    axes2[2].axis('off')
    
    plt.suptitle('Center Region Zoom (200x200)', fontsize=14)
    plt.tight_layout()
    
    zoom_path = os.path.join(args.output_dir, f'zoomed_comparison_{timestamp}.png')
    plt.savefig(zoom_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Zoomed comparison saved to: {zoom_path}')

if __name__ == '__main__':
    main()
