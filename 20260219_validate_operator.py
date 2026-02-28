#!/usr/bin/env python3
"""
Validation script for LatentPDEOperator.
Tests the operator on 10 random Poisson problems, generating u from f autoregressively.
Visualizes: Input f | GT u | Pred u | Error map | Loss by depth | Loss histogram
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from datetime import datetime

# Import from training scripts
from train_poisson_ae import (
    TreeAE, Quadtree, generate_poisson_pair, fourier_encode, 
    node_centers_from_keys, Morton2D, MAX_LEVEL, MIN_LEVEL
)

# ==========================================
# 1. LatentPDEOperator (same as training)
# ==========================================

class LatentPDEOperator(nn.Module):
    def __init__(self, emb_dim=128, pos_freqs=6, hidden_dim=128, num_heads=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.pos_freqs = pos_freqs
        pos_dim = 3 + 2 * pos_freqs * 3
        
        self.kv_proj = nn.Linear(emb_dim + pos_dim, hidden_dim * 2)
        self.q_proj = nn.Linear(pos_dim, hidden_dim)
        
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.emb_head = nn.Linear(hidden_dim, emb_dim)
        self.split_head = nn.Linear(hidden_dim, 1)
        
        nn.init.zeros_(self.emb_head.weight)
        nn.init.zeros_(self.emb_head.bias)

    def extract_context(self, qt_f, E_f):
        context_feats = []
        for d in range(MAX_LEVEL + 1):
            if E_f[d] is not None and E_f[d].shape[0] > 0:
                kd = qt_f.keys[d]
                pos = node_centers_from_keys(kd, d, MAX_LEVEL, qt_f.device)
                pos_enc = fourier_encode(pos, self.pos_freqs)
                feat = torch.cat([E_f[d], pos_enc], dim=-1)
                context_feats.append(feat)
                
        if len(context_feats) == 0:
            return None
        return torch.cat(context_feats, dim=0).unsqueeze(0) 

    def forward(self, context_f, target_pos):
        kv = self.kv_proj(context_f)
        k, v = kv.chunk(2, dim=-1)
        
        q = self.q_proj(target_pos).unsqueeze(0)
        
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        
        attn_out, _ = self.attn(query=q, key=k, value=v)
        
        x = self.norm_attn(q + attn_out)
        x = self.mlp(x) + x
        
        emb_pred = self.emb_head(x).squeeze(0)
        split_logits = self.split_head(x).squeeze(0).squeeze(-1)
        
        return emb_pred, split_logits


# ==========================================
# 2. Tree Reconstruction
# ==========================================

def build_skeleton_quadtree(keys_by_depth, device):
    qt = Quadtree(max_depth=MAX_LEVEL, device=device)
    for d in range(MAX_LEVEL + 1):
        qt.keys[d] = keys_by_depth[d]
        
    qt.parent_idx[0] = None
    for d in range(MAX_LEVEL):
        kd = qt.keys[d]; kn = qt.keys[d+1]
        if kd.numel() == 0:
            qt.children_idx[d] = torch.empty((0, 4), dtype=torch.long, device=device)
            continue
        if kn.numel() == 0:
            qt.children_idx[d] = torch.full((len(kd), 4), -1, dtype=torch.long, device=device)
            continue
        child_keys = (kd.unsqueeze(1) << 2) + torch.arange(4, device=device).view(1, 4)
        idx = torch.searchsorted(kn, child_keys).clamp(0, len(kn)-1)
        found = (kn[idx] == child_keys)
        qt.children_idx[d] = torch.where(found, idx, torch.full_like(idx, -1))
        
    qt.children_idx[MAX_LEVEL] = None
    
    for d in range(MAX_LEVEL + 1):
        n = len(qt.keys[d])
        if n == 0:
            qt.leaf_mask[d] = torch.zeros((0,), dtype=torch.bool, device=device)
        elif d == MAX_LEVEL:
            qt.leaf_mask[d] = torch.ones((n,), dtype=torch.bool, device=device)
        else:
            has_child = (qt.children_idx[d] != -1).any(dim=1)
            qt.leaf_mask[d] = ~has_child
    
    for d in range(MAX_LEVEL + 1):
        qt._construct_neigh(d)
        
    return qt


@torch.no_grad()
def generate_u_autoregressive(model_op, model_u, qt_f, E_f, device):
    model_op.eval()
    model_u.eval()
    
    context_f = model_op.extract_context(qt_f, E_f)
    
    predicted_keys = [torch.empty((0,), dtype=torch.long, device=device) for _ in range(MAX_LEVEL + 1)]
    predicted_embs = [None] * (MAX_LEVEL + 1)
    
    current_keys = torch.tensor([0], dtype=torch.long, device=device)
    
    for d in range(MAX_LEVEL + 1):
        if current_keys.numel() == 0:
            break
            
        predicted_keys[d] = current_keys
        
        pos = node_centers_from_keys(current_keys, d, MAX_LEVEL, device)
        pos_enc = fourier_encode(pos, model_op.pos_freqs)
        
        emb_pred, split_logits = model_op(context_f, pos_enc)
        predicted_embs[d] = emb_pred
        
        if d < MAX_LEVEL:
            split_mask = split_logits > 0.0
            
            if d < MIN_LEVEL:
                split_mask = torch.ones_like(split_mask, dtype=torch.bool)
                
            keys_to_split = current_keys[split_mask]
            
            if keys_to_split.numel() > 0:
                child_keys = (keys_to_split.unsqueeze(1) << 2) + torch.arange(4, device=device).view(1, 4)
                current_keys = torch.unique(child_keys.view(-1), sorted=True)
            else:
                current_keys = torch.empty((0,), dtype=torch.long, device=device)

    qt_u_pred = build_skeleton_quadtree(predicted_keys, device)
    _, val_pred = model_u.decode(qt_u_pred, predicted_embs)
    return qt_u_pred, val_pred


# ==========================================
# 3. Visualization Utilities
# ==========================================

def _draw_quadtree_field(ax, qt, val_dict, title, cmap, vmin=None, vmax=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    
    all_vals = []
    for d in range(MAX_LEVEL + 1):
        if qt.leaf_mask[d] is not None and qt.leaf_mask[d].any():
            if val_dict[d] is not None:
                v = val_dict[d][qt.leaf_mask[d]]
                if v.numel() > 0:
                    all_vals.append(v.detach().cpu().flatten())
    
    if len(all_vals) == 0:
        ax.set_title(f"{title}\n[no data]", fontsize=9)
        return
        
    all_vals = torch.cat(all_vals).numpy()
    if vmin is None:
        vmin = all_vals.min()
    if vmax is None:
        vmax = all_vals.max()
    val_range = max(vmax - vmin, 1e-10)
    
    ax.set_title(f"{title}\n[{vmin:.4f}, {vmax:.4f}]", fontsize=9)
    
    for d in range(MAX_LEVEL + 1):
        if qt.leaf_mask[d] is None or not qt.leaf_mask[d].any():
            continue
        if val_dict[d] is None:
            continue
            
        mask = qt.leaf_mask[d]
        kd = qt.keys[d][mask]
        vals = val_dict[d][mask].detach().cpu().flatten().numpy()
        
        ix, iy = Morton2D.key2xy(kd.cpu(), depth=d)
        res = 1 << d
        size = 1.0 / res
        
        for i in range(len(kd)):
            x = ix[i].item() * size
            y = iy[i].item() * size
            normalized = (vals[i] - vmin) / val_range
            color = cmap(np.clip(normalized, 0, 1))
            rect = patches.Rectangle((x, y), size, size,
                                      linewidth=0.3, edgecolor='black', facecolor=color)
            ax.add_patch(rect)


def compute_error_field(qt_gt, qt_pred, val_pred):
    """
    Compute error field: |u_gt - u_pred| at matching leaf nodes.
    Returns error_dict by depth and all_errors list.
    """
    error_by_depth = {d: [] for d in range(MAX_LEVEL + 1)}
    error_dict = {d: None for d in range(MAX_LEVEL + 1)}
    all_errors = []
    
    for d in range(MAX_LEVEL + 1):
        if qt_gt.leaf_mask[d] is None or not qt_gt.leaf_mask[d].any():
            continue
        if qt_gt.values_gt[d] is None:
            continue
            
        gt_mask = qt_gt.leaf_mask[d]
        gt_keys = qt_gt.keys[d][gt_mask]
        gt_vals = qt_gt.values_gt[d][gt_mask].detach().cpu().flatten()
        
        if qt_pred.leaf_mask[d] is None or val_pred[d] is None:
            continue
            
        pred_mask = qt_pred.leaf_mask[d]
        pred_keys = qt_pred.keys[d][pred_mask]
        pred_vals = val_pred[d][pred_mask].detach().cpu().flatten()
        
        # Create error tensor for this depth
        errors = torch.zeros(len(gt_keys))
        
        for i, gk in enumerate(gt_keys.tolist()):
            match_idx = (pred_keys == gk).nonzero(as_tuple=True)[0]
            if len(match_idx) > 0:
                pv = pred_vals[match_idx[0]].item()
                gv = gt_vals[i].item()
                err = abs(pv - gv)
                errors[i] = err
                error_by_depth[d].append(err)
                all_errors.append(err)
            else:
                errors[i] = 0.0  # No match
        
        # Store full error tensor (for all nodes at this depth, not just leaves)
        full_errors = torch.zeros(len(qt_gt.keys[d]))
        full_errors[gt_mask] = errors
        error_dict[d] = full_errors.unsqueeze(-1)
    
    return error_dict, error_by_depth, all_errors


def compute_mse_by_depth(qt_gt, qt_pred, val_pred):
    """Compute MSE losses by depth level."""
    loss_by_depth = {d: [] for d in range(MAX_LEVEL + 1)}
    all_losses = []
    
    for d in range(MAX_LEVEL + 1):
        if qt_gt.leaf_mask[d] is None or not qt_gt.leaf_mask[d].any():
            continue
        if qt_gt.values_gt[d] is None:
            continue
            
        gt_mask = qt_gt.leaf_mask[d]
        gt_keys = qt_gt.keys[d][gt_mask]
        gt_vals = qt_gt.values_gt[d][gt_mask].detach().cpu().flatten()
        
        if qt_pred.leaf_mask[d] is None or val_pred[d] is None:
            continue
            
        pred_mask = qt_pred.leaf_mask[d]
        pred_keys = qt_pred.keys[d][pred_mask]
        pred_vals = val_pred[d][pred_mask].detach().cpu().flatten()
        
        for i, gk in enumerate(gt_keys.tolist()):
            match_idx = (pred_keys == gk).nonzero(as_tuple=True)[0]
            if len(match_idx) > 0:
                pv = pred_vals[match_idx[0]].item()
                gv = gt_vals[i].item()
                loss = (pv - gv) ** 2
                loss_by_depth[d].append(loss)
                all_losses.append(loss)
    
    return loss_by_depth, all_losses


# ==========================================
# 4. Validation Function
# ==========================================

def validate_operator(operator_path, model_f_path, model_u_path, output_dir, num_samples=10, device='cpu'):
    """
    Validate the LatentPDEOperator by generating u from f autoregressively.
    """
    print(f"\n{'='*60}")
    print(f"Validating LatentPDEOperator")
    print(f"{'='*60}")
    
    # Load operator checkpoint
    op_ckpt = torch.load(operator_path, map_location=device)
    print(f"Operator checkpoint loaded:")
    print(f"  - Step: {op_ckpt['step']}")
    print(f"  - Loss: {op_ckpt['loss']:.6f}")
    print(f"  - Emb Loss: {op_ckpt['emb_loss']:.6f}")
    print(f"  - Split Loss: {op_ckpt['split_loss']:.6f}")
    
    # Load models
    model_op = LatentPDEOperator().to(device)
    model_op.load_state_dict(op_ckpt['model_state_dict'])
    model_op.eval()
    
    model_f = TreeAE(max_depth=MAX_LEVEL).to(device)
    model_u = TreeAE(max_depth=MAX_LEVEL).to(device)
    
    model_f.load_state_dict(torch.load(model_f_path, map_location=device)['model_state_dict'])
    model_u.load_state_dict(torch.load(model_u_path, map_location=device)['model_state_dict'])
    model_f.eval()
    model_u.eval()
    
    for p in model_f.parameters(): p.requires_grad = False
    for p in model_u.parameters(): p.requires_grad = False
    
    print(f"All models loaded successfully")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmap = plt.get_cmap('viridis')
    error_cmap = plt.get_cmap('hot')
    
    # Collect stats across all samples
    all_samples_loss_by_depth = {d: [] for d in range(MAX_LEVEL + 1)}
    all_sample_losses = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate data
            (f_leaf_keys, f_clean, f_noisy, f_leaves,
             u_leaf_keys, u_clean, u_noisy, u_leaves, k1, k2, denom) = generate_poisson_pair()
            
            qt_f = Quadtree(max_depth=MAX_LEVEL, device=device)
            qt_f.build_from_leaves(f_leaf_keys, f_noisy, f_clean)
            qt_u = Quadtree(max_depth=MAX_LEVEL, device=device)
            qt_u.build_from_leaves(u_leaf_keys, u_noisy, u_clean)
            
            # Encode f
            E_f = model_f.encode(qt_f)
            
            # Generate u autoregressively
            qt_u_pred, val_pred = generate_u_autoregressive(model_op, model_u, qt_f, E_f, device)
            
            # Compute losses
            loss_by_depth, all_losses = compute_mse_by_depth(qt_u, qt_u_pred, val_pred)
            error_dict, error_by_depth, all_errors = compute_error_field(qt_u, qt_u_pred, val_pred)
            
            n_gt = sum([len(qt_u.keys[d]) for d in range(MAX_LEVEL+1)])
            n_pred = sum([len(qt_u_pred.keys[d]) for d in range(MAX_LEVEL+1)])
            mean_loss = np.mean(all_losses) if all_losses else 0.0
            
            print(f"  Sample {i+1}/{num_samples}: k1={k1}, k2={k2}, GT nodes={n_gt}, Pred nodes={n_pred}, MSE={mean_loss:.6f}")
            
            # ==========================================
            # Create 2x3 plot
            # ==========================================
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Row 0, Col 0: Input f
            _draw_quadtree_field(axes[0, 0], qt_f, qt_f.values_gt, 
                                 f"Input f (k1={k1}, k2={k2})", cmap)
            
            # Row 0, Col 1: Ground truth u
            _draw_quadtree_field(axes[0, 1], qt_u, qt_u.values_gt,
                                 f"GT u (nodes={n_gt})", cmap)
            
            # Row 0, Col 2: Predicted u
            _draw_quadtree_field(axes[0, 2], qt_u_pred, val_pred,
                                 f"Pred u (nodes={n_pred})", cmap)
            
            # Row 1, Col 0: Error map |u - u_pred|
            ax_err = axes[1, 0]
            ax_err.set_xlim(0, 1); ax_err.set_ylim(0, 1)
            ax_err.set_aspect('equal'); ax_err.axis('off')
            
            max_err = max(all_errors) if all_errors else 1e-6
            ax_err.set_title(f"|u - u_pred| Error Map\n[0, {max_err:.4f}]", fontsize=9)
            
            for d in range(MAX_LEVEL + 1):
                if qt_u.leaf_mask[d] is None or not qt_u.leaf_mask[d].any():
                    continue
                if error_dict[d] is None:
                    continue
                    
                mask = qt_u.leaf_mask[d].cpu()
                kd = qt_u.keys[d].cpu()[mask]
                errs = error_dict[d][mask].detach().cpu().flatten().numpy()
                
                ix, iy = Morton2D.key2xy(kd.cpu(), depth=d)
                res = 1 << d
                size = 1.0 / res
                
                for j in range(len(kd)):
                    x = ix[j].item() * size
                    y = iy[j].item() * size
                    normalized = errs[j] / max(max_err, 1e-10)
                    color = error_cmap(np.clip(normalized, 0, 1))
                    rect = patches.Rectangle((x, y), size, size,
                                              linewidth=0.3, edgecolor='black', facecolor=color)
                    ax_err.add_patch(rect)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=error_cmap, norm=plt.Normalize(0, max_err))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_err, fraction=0.046, pad=0.04)
            cbar.set_label('|Error|', fontsize=8)
            
            # Row 1, Col 1: Avg Loss by Depth (bar chart)
            ax_depth = axes[1, 1]
            depths = [d for d in range(MAX_LEVEL + 1) if loss_by_depth[d]]
            avg_losses = [np.mean(loss_by_depth[d]) for d in depths]
            
            if depths:
                bars = ax_depth.bar(depths, avg_losses, color='steelblue', edgecolor='black')
                ax_depth.set_xlabel('Depth Level', fontsize=10)
                ax_depth.set_ylabel('Avg MSE Loss', fontsize=10)
                ax_depth.set_title('Avg Loss by Depth', fontsize=11)
                ax_depth.set_xticks(range(MAX_LEVEL + 1))
                ax_depth.grid(True, axis='y', alpha=0.3)
                for bar, val in zip(bars, avg_losses):
                    ax_depth.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  f'{val:.4f}', ha='center', va='bottom', fontsize=7)
            else:
                ax_depth.text(0.5, 0.5, 'No loss data', ha='center', va='center')
                ax_depth.axis('off')
            
            # Row 1, Col 2: Loss Distribution Histogram
            ax_hist = axes[1, 2]
            if all_losses:
                ax_hist.hist(all_losses, bins=50, color='coral', edgecolor='black', alpha=0.7)
                ax_hist.axvline(np.mean(all_losses), color='red', linestyle='--', 
                                label=f'Mean: {np.mean(all_losses):.5f}')
                ax_hist.set_xlabel('MSE Loss', fontsize=10)
                ax_hist.set_ylabel('Count', fontsize=10)
                ax_hist.set_title('Loss Distribution (All Leaves)', fontsize=11)
                ax_hist.legend(fontsize=8)
                ax_hist.grid(True, alpha=0.3)
            else:
                ax_hist.text(0.5, 0.5, 'No loss data', ha='center', va='center')
                ax_hist.axis('off')
            
            plt.suptitle(f"Operator Validation - Sample {i+1}: k1={k1}, k2={k2} (Step {op_ckpt['step']})", fontsize=14)
            plt.tight_layout()
            
            output_path = f'{output_dir}/sample_{i+1:02d}_k{k1}_{k2}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    -> Saved {output_path}")
            
            # Accumulate for summary
            for d in range(MAX_LEVEL + 1):
                if loss_by_depth[d]:
                    all_samples_loss_by_depth[d].append(np.mean(loss_by_depth[d]))
            all_sample_losses.extend(all_losses)
    
    # ==========================================
    # Summary plot
    # ==========================================
    fig_summary, axes_s = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Avg loss by depth across all samples
    ax_s1 = axes_s[0]
    depths = [d for d in range(MAX_LEVEL + 1) if all_samples_loss_by_depth[d]]
    avg_of_avg = [np.mean(all_samples_loss_by_depth[d]) for d in depths]
    std_of_avg = [np.std(all_samples_loss_by_depth[d]) for d in depths]
    
    if depths:
        bars = ax_s1.bar(depths, avg_of_avg, yerr=std_of_avg, 
                         color='steelblue', edgecolor='black', capsize=5, alpha=0.8)
        ax_s1.set_xlabel('Depth Level', fontsize=12)
        ax_s1.set_ylabel('Average MSE Loss', fontsize=12)
        ax_s1.set_title(f'Average Loss by Depth (Across {num_samples} Samples)', fontsize=14)
        ax_s1.set_xticks(range(MAX_LEVEL + 1))
        ax_s1.grid(True, axis='y', alpha=0.3)
        
        for bar, val, std in zip(bars, avg_of_avg, std_of_avg):
            ax_s1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.0001,
                       f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    
    # Right: Overall loss histogram
    ax_s2 = axes_s[1]
    if all_sample_losses:
        ax_s2.hist(all_sample_losses, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax_s2.axvline(np.mean(all_sample_losses), color='red', linestyle='--',
                      label=f'Mean: {np.mean(all_sample_losses):.5f}')
        ax_s2.set_xlabel('MSE Loss', fontsize=12)
        ax_s2.set_ylabel('Count', fontsize=12)
        ax_s2.set_title(f'Overall Loss Distribution ({len(all_sample_losses)} leaves)', fontsize=14)
        ax_s2.legend(fontsize=10)
        ax_s2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Operator Validation Summary (Step {op_ckpt['step']}, Loss={op_ckpt['loss']:.6f})", fontsize=14)
    plt.tight_layout()
    
    summary_path = f'{output_dir}/summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig_summary)
    print(f"\n  -> Saved summary: {summary_path}")
    
    return op_ckpt


# ==========================================
# 5. Main
# ==========================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    # Paths
    operator_path = '/home/tahmid/Development/OctreeRefineNet/plots/operator_20260219_223308/best_operator.pt'
    model_f_path = '/home/tahmid/Development/OctreeRefineNet/plots/poisson_20260219_200646/best_model_f.pt'
    model_u_path = '/home/tahmid/Development/OctreeRefineNet/plots/poisson_20260219_200646/best_model_u.pt'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'/home/tahmid/Development/OctreeRefineNet/plots/val_operator_{timestamp}'
    
    print(f"Output directory: {output_dir}")
    
    ckpt = validate_operator(
        operator_path=operator_path,
        model_f_path=model_f_path,
        model_u_path=model_u_path,
        output_dir=output_dir,
        num_samples=10,
        device=device
    )
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Operator Step: {ckpt['step']}")
    print(f"Operator Loss: {ckpt['loss']:.6f}")
    print(f"All plots saved to: {output_dir}")
