"""
visualize_embeddings.py

Visualizes the quadtree embedding structure for f and u autoencoders.
For each sample, shows ALL nodes (not just leaves) at every depth,
colored by embedding L2 norm and PCA→RGB.

Usage:
    python visualize_embeddings.py --model_f PATH --model_u PATH
    
    If no paths given, trains fresh models for 500 steps first.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
import argparse, os, random

from train_poisson_ae import (
    TreeAE, Quadtree, generate_poisson_pair,
    Morton2D, MAX_LEVEL, fourier_encode, node_centers_from_keys
)

random.seed(42); np.random.seed(42); torch.manual_seed(42)


def draw_cells(ax, keys, depth, values, cmap, vmin, vmax, lw=0.3):
    """Draw quadtree cells colored by scalar values."""
    ix, iy = Morton2D.key2xy(keys.cpu(), depth=depth)
    res = 1 << depth
    size = 1.0 / res
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i in range(len(keys)):
        c = cmap(norm(values[i]))
        rect = patches.Rectangle(
            (ix[i].item() * size, iy[i].item() * size), size, size,
            linewidth=lw, edgecolor='grey', facecolor=c)
        ax.add_patch(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.axis('off')


def draw_cells_rgb(ax, keys, depth, rgb, lw=0.3):
    """Draw quadtree cells colored by RGB array (N,3)."""
    ix, iy = Morton2D.key2xy(keys.cpu(), depth=depth)
    res = 1 << depth
    size = 1.0 / res
    for i in range(len(keys)):
        rect = patches.Rectangle(
            (ix[i].item() * size, iy[i].item() * size), size, size,
            linewidth=lw, edgecolor='grey', facecolor=rgb[i])
        ax.add_patch(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.axis('off')


def pca_to_rgb(emb_np):
    """Project (N, D) embeddings to (N, 3) RGB via PCA."""
    if emb_np.shape[0] < 3:
        return np.ones((emb_np.shape[0], 3)) * 0.5, np.array([0.0, 0.0, 0.0])
    pca = PCA(n_components=3)
    proj = pca.fit_transform(emb_np)
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    rgb = (proj - lo) / (hi - lo + 1e-8)
    return rgb, pca.explained_variance_ratio_


def visualize(model_f, model_u, output_dir, device='cpu', n_samples=3):
    os.makedirs(output_dir, exist_ok=True)
    model_f.eval(); model_u.eval()

    for s in range(n_samples):
        (f_keys, f_clean, _, f_leaves,
         u_keys, u_clean, _, u_leaves,
         k1, k2, denom) = generate_poisson_pair(noise_std=0.0)

        qt_f = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_f.build_from_leaves(f_keys, f_clean, f_clean)
        qt_u = Quadtree(max_depth=MAX_LEVEL, device=device)
        qt_u.build_from_leaves(u_keys, u_clean, u_clean)

        with torch.no_grad():
            E_f = model_f.encode(qt_f)
            E_u = model_u.encode(qt_u)

        # Find which depths actually have nodes
        active = []
        for d in range(MAX_LEVEL + 1):
            has_f = qt_f.keys[d].numel() > 0 and E_f[d].numel() > 0
            has_u = qt_u.keys[d].numel() > 0 and E_u[d].numel() > 0
            if has_f or has_u:
                active.append(d)

        n_rows = len(active)
        # Columns: f GT | f ‖z‖ | f PCA | u GT | u ‖z‖ | u PCA
        fig, axes = plt.subplots(n_rows, 6, figsize=(36, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, 6)

        cmap_val = plt.get_cmap('RdBu_r')
        cmap_norm = plt.get_cmap('inferno')

        for ri, d in enumerate(active):
            # ── f side ──
            if qt_f.keys[d].numel() > 0 and E_f[d].numel() > 0:
                kf = qt_f.keys[d]
                ef = E_f[d].cpu()
                gf = qt_f.features_in[d].cpu().flatten().numpy()
                nf = ef.norm(dim=1).numpy()

                # GT values
                draw_cells(axes[ri, 0], kf, d, gf, cmap_val,
                           gf.min(), gf.max())
                axes[ri, 0].set_title(f'f GT  d={d}  n={len(kf)}', fontsize=10)

                # Embedding norm
                draw_cells(axes[ri, 1], kf, d, nf, cmap_norm,
                           nf.min(), nf.max())
                axes[ri, 1].set_title(f'f ‖z‖₂  [{nf.min():.2f}, {nf.max():.2f}]', fontsize=10)

                # PCA RGB
                rgb, var = pca_to_rgb(ef.numpy())
                draw_cells_rgb(axes[ri, 2], kf, d, rgb)
                axes[ri, 2].set_title(f'f PCA→RGB  var={var[0]:.2f},{var[1]:.2f},{var[2]:.2f}', fontsize=9)
            else:
                for c in range(3):
                    axes[ri, c].text(0.5, 0.5, f'f: empty d={d}', ha='center', va='center')
                    axes[ri, c].axis('off')

            # ── u side ──
            if qt_u.keys[d].numel() > 0 and E_u[d].numel() > 0:
                ku = qt_u.keys[d]
                eu = E_u[d].cpu()
                gu = qt_u.features_in[d].cpu().flatten().numpy()
                nu = eu.norm(dim=1).numpy()

                draw_cells(axes[ri, 3], ku, d, gu, cmap_val,
                           gu.min(), gu.max())
                axes[ri, 3].set_title(f'u GT  d={d}  n={len(ku)}', fontsize=10)

                draw_cells(axes[ri, 4], ku, d, nu, cmap_norm,
                           nu.min(), nu.max())
                axes[ri, 4].set_title(f'u ‖z‖₂  [{nu.min():.2f}, {nu.max():.2f}]', fontsize=10)

                rgb, var = pca_to_rgb(eu.numpy())
                draw_cells_rgb(axes[ri, 5], ku, d, rgb)
                axes[ri, 5].set_title(f'u PCA→RGB  var={var[0]:.2f},{var[1]:.2f},{var[2]:.2f}', fontsize=9)
            else:
                for c in range(3, 6):
                    axes[ri, c].text(0.5, 0.5, f'u: empty d={d}', ha='center', va='center')
                    axes[ri, c].axis('off')

        plt.suptitle(f'Embedding Structure — k1={k1}, k2={k2}  |  f_leaves={len(f_leaves)}, u_leaves={len(u_leaves)}',
                     fontsize=14)
        plt.tight_layout()
        path = f'{output_dir}/embeddings_sample{s}_k{k1}_{k2}.png'
        plt.savefig(path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_f', type=str, 
                        default='/home/tahmid/Development/OctreeRefineNet/plots/poisson_20260219_200646/best_model_f.pt')
    parser.add_argument('--model_u', type=str, 
                        default='/home/tahmid/Development/OctreeRefineNet/plots/poisson_20260219_200646/best_model_u.pt')
    parser.add_argument('--output', type=str, default='plots/embedding_viz')
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_f = TreeAE(max_depth=MAX_LEVEL).to(device)
    model_u = TreeAE(max_depth=MAX_LEVEL).to(device)

    if os.path.exists(args.model_f) and os.path.exists(args.model_u):
        model_f.load_state_dict(torch.load(args.model_f, map_location=device)['model_state_dict'])
        model_u.load_state_dict(torch.load(args.model_u, map_location=device)['model_state_dict'])
        print('Loaded pre-trained models.')
    else:
        print('Model paths not found — using untrained models (random embeddings).')
        print('Pass --model_f and --model_u for meaningful results.')

    visualize(model_f, model_u, args.output, device, args.samples)