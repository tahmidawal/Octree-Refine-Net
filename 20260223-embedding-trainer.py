import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from datetime import datetime

# Import your existing definitions from the AE script
from train_poisson_ae import (
    TreeAE, Quadtree, generate_poisson_pair, fourier_encode, 
    node_centers_from_keys, Morton2D, MAX_LEVEL, MIN_LEVEL
)

# ==========================================
# 1. Lean Cross-Attention Operator
# ==========================================

class LatentPDEOperator(nn.Module):
    def __init__(self, emb_dim=128, pos_freqs=6, hidden_dim=128, num_heads=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.pos_freqs = pos_freqs
        pos_dim = 3 + 2 * pos_freqs * 3
        
        # Project context (f) into Keys and Values
        self.kv_proj = nn.Linear(emb_dim + pos_dim, hidden_dim * 2)
        # Project target spatial coordinates (u) into Queries
        self.q_proj = nn.Linear(pos_dim, hidden_dim)
        
        # CRITICAL: LayerNorms for Attention stability
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Deeper MLP for better operator capacity
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.emb_head = nn.Linear(hidden_dim, emb_dim)
        self.split_head = nn.Linear(hidden_dim, 1)
        
        # Trick: Initialize emb_head to zero so initial predictions match the small latent variance
        nn.init.zeros_(self.emb_head.weight)
        nn.init.zeros_(self.emb_head.bias)

    def extract_context(self, qt_f, E_f):
        """Flattens the entire f quadtree into an unordered set of context tokens."""
        context_feats = []
        for d in range(MAX_LEVEL + 1):
            if E_f[d] is not None and E_f[d].shape[0] > 0:
                kd = qt_f.keys[d]
                pos = node_centers_from_keys(kd, d, MAX_LEVEL, qt_f.device)
                pos_enc = fourier_encode(pos, self.pos_freqs)
                # Token = [Embedding, Positional Encoding]
                feat = torch.cat([E_f[d], pos_enc], dim=-1)
                context_feats.append(feat)
                
        if len(context_feats) == 0:
            return None
        # Shape: (1, Total_N_f, emb_dim + pos_dim)
        return torch.cat(context_feats, dim=0).unsqueeze(0) 

    def forward(self, context_f, target_pos):
        """
        context_f: (1, N_f, features)
        target_pos: (N_u, pos_dim) spatial queries for u
        """
        # 1. Prepare K, V
        kv = self.kv_proj(context_f)
        k, v = kv.chunk(2, dim=-1)
        
        # 2. Prepare Q
        q = self.q_proj(target_pos).unsqueeze(0)
        
        # 3. Normalize
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        
        # 4. Attention
        attn_out, _ = self.attn(query=q, key=k, value=v)
        
        # 5. Residual + Norm + MLP
        x = self.norm_attn(q + attn_out)
        x = self.mlp(x) + x  # Second residual
        
        # Squeeze out the batch dimension here to fix the PyTorch broadcasting bug!
        emb_pred = self.emb_head(x).squeeze(0)
        split_logits = self.split_head(x).squeeze(0).squeeze(-1)
        
        return emb_pred, split_logits

# ==========================================
# 2. Visualization Utilities
# ==========================================

def _draw_quadtree_field(ax, qt, val_dict, title, cmap, vmin=None, vmax=None):
    """
    Draw a quadtree field from keys and values.
    val_dict: dict mapping depth -> tensor of values (N_d, 1) or (N_d,)
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    
    # Collect all leaf values for normalization
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


def compute_leaf_losses(qt_gt, qt_pred, val_pred):
    """
    Compute per-leaf MSE losses between ground truth and predicted values.
    Returns loss_by_depth dict and all_losses list.
    """
    loss_by_depth = {d: [] for d in range(MAX_LEVEL + 1)}
    all_losses = []
    
    for d in range(MAX_LEVEL + 1):
        # Get GT leaves
        if qt_gt.leaf_mask[d] is None or not qt_gt.leaf_mask[d].any():
            continue
        if qt_gt.values_gt[d] is None:
            continue
            
        gt_mask = qt_gt.leaf_mask[d]
        gt_keys = qt_gt.keys[d][gt_mask]
        gt_vals = qt_gt.values_gt[d][gt_mask].detach().cpu().flatten()
        
        # Get predicted values at same keys (if they exist)
        if qt_pred.leaf_mask[d] is None or val_pred[d] is None:
            continue
            
        pred_mask = qt_pred.leaf_mask[d]
        pred_keys = qt_pred.keys[d][pred_mask]
        pred_vals = val_pred[d][pred_mask].detach().cpu().flatten()
        
        # Match keys between GT and pred
        for i, gk in enumerate(gt_keys.tolist()):
            # Find this key in pred
            match_idx = (pred_keys == gk).nonzero(as_tuple=True)[0]
            if len(match_idx) > 0:
                pv = pred_vals[match_idx[0]].item()
                gv = gt_vals[i].item()
                loss = (pv - gv) ** 2
                loss_by_depth[d].append(loss)
                all_losses.append(loss)
    
    return loss_by_depth, all_losses


def plot_operator_comparison(qt_f, qt_u_gt, qt_u_pred, val_pred, out_path, step, k1, k2, loss_history=None):
    """
    Plot 2x3 panel comparison:
    Row 0: Input f | GT u | Predicted u
    Row 1: Loss curve | Avg Loss by Depth | Loss Distribution
    """
    cmap = plt.get_cmap('viridis')
    loss_cmap = plt.get_cmap('hot')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 0, Col 0: Input f
    _draw_quadtree_field(axes[0, 0], qt_f, qt_f.values_gt, 
                         f"Input f (k1={k1}, k2={k2})", cmap)
    
    # Row 0, Col 1: Ground truth u
    _draw_quadtree_field(axes[0, 1], qt_u_gt, qt_u_gt.values_gt,
                         f"GT u (nodes={sum(len(qt_u_gt.keys[d]) for d in range(MAX_LEVEL+1))})", cmap)
    
    # Row 0, Col 2: Predicted u
    _draw_quadtree_field(axes[0, 2], qt_u_pred, val_pred,
                         f"Pred u (nodes={sum(len(qt_u_pred.keys[d]) for d in range(MAX_LEVEL+1))})", cmap)
    
    # Compute losses
    loss_by_depth, all_losses = compute_leaf_losses(qt_u_gt, qt_u_pred, val_pred)
    
    # Row 1, Col 0: Training loss curve
    ax_loss = axes[1, 0]
    if loss_history is not None and len(loss_history['total']) > 0:
        ax_loss.plot(loss_history['total'], label='Total', color='navy', linewidth=1.5)
        ax_loss.plot(loss_history['emb'], label='Embedding', color='steelblue', linewidth=1, alpha=0.7)
        ax_loss.plot(loss_history['split'], label='Split', color='coral', linewidth=1, alpha=0.7)
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Training Loss Curve')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')
    else:
        ax_loss.text(0.5, 0.5, 'No loss history', ha='center', va='center')
        ax_loss.axis('off')
    
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
    
    plt.suptitle(f"Operator Training - Step {step}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# 3. Dynamic Tree Reconstruction
# ==========================================

def build_skeleton_quadtree(keys_by_depth, device):
    """Creates a skeleton Quadtree object purely from predicted keys for the Decoder."""
    qt = Quadtree(max_depth=MAX_LEVEL, device=device)
    for d in range(MAX_LEVEL + 1):
        qt.keys[d] = keys_by_depth[d]
        
    # Build parent/child relationships needed by model_u.decode
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
    
    # Build leaf_mask: a node is a leaf if it has no children
    for d in range(MAX_LEVEL + 1):
        n = len(qt.keys[d])
        if n == 0:
            qt.leaf_mask[d] = torch.zeros((0,), dtype=torch.bool, device=device)
        elif d == MAX_LEVEL:
            # All nodes at max depth are leaves
            qt.leaf_mask[d] = torch.ones((n,), dtype=torch.bool, device=device)
        else:
            # A node is a leaf if all its children are -1
            has_child = (qt.children_idx[d] != -1).any(dim=1)
            qt.leaf_mask[d] = ~has_child
    
    # Build neighbors for QuadConvs in Decoder
    for d in range(MAX_LEVEL + 1):
        qt._construct_neigh(d)
        
    return qt

@torch.no_grad()
def generate_u_autoregressive(model_op, model_u, qt_f, E_f, device):
    """Inference loop: Predicts the u mesh structure and values dynamically."""
    model_op.eval()
    model_u.eval()
    
    context_f = model_op.extract_context(qt_f, E_f)
    
    predicted_keys = [torch.empty((0,), dtype=torch.long, device=device) for _ in range(MAX_LEVEL + 1)]
    predicted_embs = [None] * (MAX_LEVEL + 1)
    
    # Start at root
    current_keys = torch.tensor([0], dtype=torch.long, device=device)
    
    for d in range(MAX_LEVEL + 1):
        if current_keys.numel() == 0:
            break
            
        predicted_keys[d] = current_keys
        
        # Get coordinates for these keys to use as queries
        pos = node_centers_from_keys(current_keys, d, MAX_LEVEL, device)
        pos_enc = fourier_encode(pos, model_op.pos_freqs)
        
        # Query the operator
        emb_pred, split_logits = model_op(context_f, pos_enc)
        predicted_embs[d] = emb_pred
        
        if d < MAX_LEVEL:
            # Threshold the predicted splits (Sigmoid > 0.5 means logit > 0)
            split_mask = split_logits > 0.0
            
            # Force split up to MIN_LEVEL just like the GT generator
            if d < MIN_LEVEL:
                split_mask = torch.ones_like(split_mask, dtype=torch.bool)
                
            keys_to_split = current_keys[split_mask]
            
            if keys_to_split.numel() > 0:
                # Generate child keys mathematically
                child_keys = (keys_to_split.unsqueeze(1) << 2) + torch.arange(4, device=device).view(1, 4)
                current_keys = torch.unique(child_keys.view(-1), sorted=True)
            else:
                current_keys = torch.empty((0,), dtype=torch.long, device=device)

    # Reconstruct standard tree structure so model_u.decode can process it
    qt_u_pred = build_skeleton_quadtree(predicted_keys, device)
    
    # Finally, decode values from our predicted embeddings!
    _, val_pred = model_u.decode(qt_u_pred, predicted_embs)
    return qt_u_pred, val_pred

# ==========================================
# 3. Training Loop
# ==========================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Operator Training on {device}")

    # --- 1. Load Pre-trained AEs ---
    model_f_path = '/home/tahmid/Development/OctreeRefineNet/plots/poisson_20260219_200646/best_model_f.pt'
    model_u_path = '/home/tahmid/Development/OctreeRefineNet/plots/poisson_20260219_200646/best_model_u.pt'
    
    model_f = TreeAE(max_depth=MAX_LEVEL).to(device)
    model_u = TreeAE(max_depth=MAX_LEVEL).to(device)
    
    try:
        model_f.load_state_dict(torch.load(model_f_path, map_location=device)['model_state_dict'])
        model_u.load_state_dict(torch.load(model_u_path, map_location=device)['model_state_dict'])
        print("Successfully loaded pre-trained Autoencoders.")
    except Exception as e:
        print(f"Error loading models. Check paths! Error: {e}")
        return

    # Freeze AEs completely
    model_f.eval(); model_u.eval()
    for p in model_f.parameters(): p.requires_grad = False
    for p in model_u.parameters(): p.requires_grad = False

    # --- 2. Initialize Operator ---
    model_op = LatentPDEOperator().to(device)
    optimizer = torch.optim.Adam(model_op.parameters(), lr=1e-3)
    
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"plots/operator_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    num_steps = 5000
    best_loss = float('inf')
    loss_history = {'total': [], 'emb': [], 'split': []}
    
    for step in range(num_steps):
        # 1. Generate Data
        (f_leaf_keys, f_clean, f_noisy, f_leaves,
         u_leaf_keys, u_clean, u_noisy, u_leaves, k1, k2, denom) = generate_poisson_pair()

        qt_f = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_f.build_from_leaves(f_leaf_keys, f_noisy, f_clean)
        qt_u = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_u.build_from_leaves(u_leaf_keys, u_noisy, u_clean)

        # 2. Get f embeddings for context (No gradients needed for encoder)
        with torch.no_grad():
            E_f = model_f.encode(qt_f)

        context_f = model_op.extract_context(qt_f, E_f)
        if context_f is None:
            continue

        model_op.train()
        optimizer.zero_grad()
        
        predicted_embs = [None] * (MAX_LEVEL + 1)
        L_split_total = torch.tensor(0.0, device=device)
        nodes_counted = 0
        
        # 3. Teacher Forcing: Predict embeddings at every depth (on GT tree structure)
        for d in range(MAX_LEVEL + 1):
            kd = qt_u.keys[d]
            if kd.numel() == 0:
                predicted_embs[d] = torch.zeros((0, model_op.emb_dim), device=device)
                continue
                
            pos = node_centers_from_keys(kd, d, MAX_LEVEL, device)
            pos_enc = fourier_encode(pos, model_op.pos_freqs)
            
            emb_pred, split_logits = model_op(context_f, pos_enc)
            predicted_embs[d] = emb_pred
            
            # Split loss stays the same
            if d < MAX_LEVEL and qt_u.split_gt[d] is not None:
                L_split_total += bce(split_logits, qt_u.split_gt[d]) * len(kd)
                
            nodes_counted += len(kd)

        # 4. Decode predicted embeddings through frozen u decoder
        # IMPORTANT: call .decoder() directly, NOT .decode() which wraps in no_grad
        split_logits_dec, val_pred = model_u.decoder(qt_u, predicted_embs)

        # 5. Loss on decoded VALUES vs ground truth values
        L_val = torch.tensor(0.0, device=device)
        n_val = 0
        for d in range(MAX_LEVEL + 1):
            mask = qt_u.leaf_mask[d]
            if mask is None or mask.numel() == 0 or not mask.any():
                continue
            L_val += mse(val_pred[d][mask], qt_u.values_gt[d][mask])
            n_val += 1
        if n_val > 0:
            L_val = L_val / n_val

        L_total = L_val + 0.5 * (L_split_total / nodes_counted)
        L_total.backward()
        optimizer.step()

        # Track loss history
        loss_history['total'].append(L_total.item())
        loss_history['emb'].append(L_val.item())
        loss_history['split'].append(L_split_total.item()/nodes_counted)

        if step % 50 == 0:
            print(f"Step {step:4d} | Total Loss: {L_total.item():.5f} | Val: {L_val.item():.5f} | Split: {L_split_total.item()/nodes_counted:.5f}")

        # 4. Evaluation Loop (Autoregressive Generation)
        if step > 0 and step % 100 == 0:
            print(f" => Running Autoregressive Mesh Generation Eval...")
            qt_u_pred, val_pred = generate_u_autoregressive(model_op, model_u, qt_f, E_f, device)
            
            # Count generated nodes
            n_gen = sum([len(qt_u_pred.keys[d]) for d in range(MAX_LEVEL+1)])
            n_gt = sum([len(qt_u.keys[d]) for d in range(MAX_LEVEL+1)])
            print(f"    GT u nodes: {n_gt} | Generated u nodes: {n_gen}")
            
            # Plot comparison: f | GT u | Pred u with loss analysis
            plot_path = f"{out_dir}/comparison_step{step:04d}.png"
            plot_operator_comparison(qt_f, qt_u, qt_u_pred, val_pred, plot_path, step, k1, k2, loss_history)
            print(f"    -> Saved {plot_path}")
            
            # Save best checkpoint every 100 steps
            current_loss = L_total.item()
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save({
                    'step': step,
                    'model_state_dict': model_op.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                    'emb_loss': L_val.item(),
                    'split_loss': L_split_total.item()/nodes_counted,
                }, f"{out_dir}/best_operator.pt")
                print(f"    -> New best checkpoint saved (loss: {best_loss:.6f})")
    
    # Final checkpoint save
    torch.save({
        'step': num_steps - 1,
        'model_state_dict': model_op.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': L_total.item(),
        'emb_loss': L_val.item(),
        'split_loss': L_split_total.item()/nodes_counted,
    }, f"{out_dir}/final_operator.pt")
    print(f"\nTraining complete. Final checkpoint saved to {out_dir}/final_operator.pt")

if __name__ == '__main__':
    main()