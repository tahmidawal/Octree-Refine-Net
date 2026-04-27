"""For a few selected (k1, k2) cases, produce a detailed 2×4 figure each:

  Row 1: INPUT F | GROUND TRUTH u | u from MLP | u from NM-ROM (fine-cell)
  Row 2: mesh structure | |MLP error| | |NM-ROM error| | per-leaf error histogram
         (depths overlaid in solid/dashed)

Usage:
  python plot_per_case_detailed.py --exp-dir runs/expN [--k 2,2 3,3 5,3]
"""
import argparse
import sys
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import f_to_u_nmrom as M
sys.modules['__main__']._QuadNode = M._QuadNode


def draw_field(ax, leaf_list, vals, title, vmin=None, vmax=None,
               cmap='RdBu_r'):
    if vmin is None: vmin = float(np.nanmin(vals))
    if vmax is None: vmax = float(np.nanmax(vals))
    if cmap == 'RdBu_r':
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    cmap_obj = plt.get_cmap(cmap)
    rng = max(vmax - vmin, 1e-30)
    for li, (xc, yc, h, *_) in enumerate(leaf_list):
        t = (vals[li] - vmin) / rng
        ax.add_patch(patches.Rectangle(
            (xc - h/2, yc - h/2), h, h,
            facecolor=cmap_obj(np.clip(t, 0, 1)),
            edgecolor='none'))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def draw_mesh(ax, leaf_list, fine_mask):
    for li, (xc, yc, h, *_) in enumerate(leaf_list):
        c = '#fdfd9b' if fine_mask[li] > 0.5 else '#3b3b66'
        ax.add_patch(patches.Rectangle(
            (xc - h/2, yc - h/2), h, h, facecolor=c,
            edgecolor='black', linewidth=0.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('mesh: fine (yellow) / coarse (blue)', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def per_depth_hist(ax, depths, errs_dict):
    unique_d = sorted(set(int(d) for d in depths))
    colors = {'MLP only': '#2ca02c',
              'WARM (full)': '#1f77b4',
              'WARM (fine)': '#9467bd'}
    bins = np.logspace(-5, -1, 30)
    first_d = unique_d[0]
    for d in unique_d:
        mask = depths == d
        for lbl, errs in errs_dict.items():
            data = errs[mask]
            data = data[data > 0]
            if data.size == 0: continue
            ax.hist(data, bins=bins, alpha=0.45,
                    label=f'{lbl} d={d}' if d == first_d else None,
                    color=colors[lbl],
                    histtype='stepfilled' if d == first_d else 'step',
                    linestyle='solid' if d == first_d else 'dashed',
                    edgecolor='black', linewidth=0.4)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('|error| per leaf'); ax.set_ylabel('count (log)')
    ax.set_title(f'per-leaf |error|, by depth (solid=d={first_d}, dashed=deeper)', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')


def render_case(sample, ae_model, ae_params, mlp_model, mlp_params, encode_F,
                mlp_predict, out_png):
    qt = sample['qt']
    pd_f = M.topology_to_padded(qt, which='f')
    K_op, f_vec, u_ref, N, gidx, leaf_list = M.build_fvm_operator(qt)
    leaf_depths = np.array([lf[3] for lf in leaf_list], np.int32)
    max_d = int(leaf_depths.max())
    fine_mask = (leaf_depths == max_d).astype(np.float32)
    n_fine = int(fine_mask.sum()); n_coarse = N - n_fine

    leaf_xy = np.array([(lf[0], lf[1]) for lf in leaf_list], np.float32)
    F_field = np.array(M.poisson_f(leaf_xy[:,0], leaf_xy[:,1], qt['k1'], qt['k2']))

    # MLP-only
    z_F = encode_F(pd_f); z_u0 = mlp_predict(z_F)
    leaf_gather = M.build_leaf_gather(qt, gidx)
    @jax.jit
    def decode_z(z):
        E = M.z_to_E(z)
        _, vpred = ae_model.apply({'params': ae_params}, E, pd_f, method=ae_model.decode)
        return M.extract_leaf_values(vpred, leaf_gather) / M.U_SCALE
    u_mlp = np.array(decode_z(z_u0))

    # WARM full
    solve_warm, _ = M.make_nmrom_solver(ae_model, ae_params, pd_f, qt, K_op, f_vec,
                                         gidx, max_outer=8, max_inner=6, tol=1e-5)
    _ = solve_warm(z_init=z_u0)
    _, u_warm, _ = solve_warm(z_init=z_u0)
    u_warm = np.array(u_warm)

    # WARM fine
    solve_fine, _ = M.make_nmrom_solver(ae_model, ae_params, pd_f, qt, K_op, f_vec,
                                         gidx, max_outer=8, max_inner=6, tol=1e-5,
                                         residual_weights=fine_mask,
                                         lam_reg=1e-3, z_anchor=z_u0)
    _ = solve_fine(z_init=z_u0)
    _, u_fine, _ = solve_fine(z_init=z_u0)
    u_fine = np.array(u_fine)

    u_ref_np = np.array(u_ref)
    err_mlp  = np.abs(u_mlp  - u_ref_np)
    err_warm = np.abs(u_warm - u_ref_np)
    err_fine = np.abs(u_fine - u_ref_np)

    rel_mlp  = float(np.linalg.norm(u_mlp  - u_ref_np) / np.linalg.norm(u_ref_np))
    rel_warm = float(np.linalg.norm(u_warm - u_ref_np) / np.linalg.norm(u_ref_np))
    rel_fine = float(np.linalg.norm(u_fine - u_ref_np) / np.linalg.norm(u_ref_np))

    u_vmin = float(min(u_ref_np.min(), u_mlp.min(), u_fine.min()))
    u_vmax = float(max(u_ref_np.max(), u_mlp.max(), u_fine.max()))
    f_amp  = max(abs(F_field.min()), abs(F_field.max()), 1e-30)
    e_max  = float(max(err_mlp.max(), err_fine.max(), 1e-30))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    draw_field(axes[0,0], leaf_list, F_field,
               f"INPUT F   k=({qt['k1']:.0f},{qt['k2']:.0f})  N={N}",
               vmin=-f_amp, vmax=f_amp)
    draw_field(axes[0,1], leaf_list, u_ref_np, "GROUND TRUTH u  (analytical)",
               vmin=u_vmin, vmax=u_vmax)
    draw_field(axes[0,2], leaf_list, u_mlp,
               f"u from MLP only  rel-L2={rel_mlp:.3e}",
               vmin=u_vmin, vmax=u_vmax)
    draw_field(axes[0,3], leaf_list, u_fine,
               f"u from NM-ROM (fine-cell)  rel-L2={rel_fine:.3e}",
               vmin=u_vmin, vmax=u_vmax)

    draw_mesh(axes[1,0], leaf_list, fine_mask)
    draw_field(axes[1,1], leaf_list, err_mlp,  f"|MLP err|  max={err_mlp.max():.2e}",
               vmin=0, vmax=e_max, cmap='hot')
    draw_field(axes[1,2], leaf_list, err_fine, f"|NM-ROM(fine) err|  max={err_fine.max():.2e}",
               vmin=0, vmax=e_max, cmap='hot')
    per_depth_hist(axes[1,3], leaf_depths,
                   {'MLP only': err_mlp, 'WARM (full)': err_warm, 'WARM (fine)': err_fine})

    plt.suptitle(
        f"k=({qt['k1']:.0f}, {qt['k2']:.0f})   N={N}  fine={n_fine}  coarse={n_coarse}   "
        f"|   rel-L2: MLP={rel_mlp:.3e}  WARM-full={rel_warm:.3e}  WARM-fine={rel_fine:.3e}",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"Saved {out_png}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-dir', required=True, help='Experiment directory containing dataset/AE/MLP checkpoints')
    p.add_argument('--out-dir', default=None,
                   help='Output dir for plots (default: <exp-dir>/plots_detailed)')
    p.add_argument('--k', nargs='+', default=['2,2', '3,2', '3,3', '5,3'],
                   help='Space-separated list of k1,k2 pairs to plot')
    args = p.parse_args()

    exp = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp / 'plots' / 'detailed'
    out_dir.mkdir(parents=True, exist_ok=True)

    ds     = pickle.load(open(exp / 'paired_dataset.pkl', 'rb'))
    ae_ck  = pickle.load(open(exp / 'shared_ae.pkl',     'rb'))
    mlp_ck = pickle.load(open(exp / 'mlp_z_f_to_z_u.pkl','rb'))

    ae_model  = M.CompressedBottleneckAE(hidden=M.HIDDEN, emb_dim=M.EMB_DIM)
    ae_params = ae_ck['params']
    mlp_model = M.LatentMLP(hidden=mlp_ck.get('hidden', 512),
                             n_layers=mlp_ck.get('n_layers', 3))
    mlp_params = mlp_ck['params']

    @jax.jit
    def encode_F(pd_f):
        E = ae_model.apply({'params': ae_params}, pd_f, method=ae_model.encode)
        return M.E_to_z(E)
    @jax.jit
    def mlp_predict(z):
        return mlp_model.apply({'params': mlp_params}, z)

    # parse k pairs
    target_ks = [tuple(int(t) for t in s.split(',')) for s in args.k]

    # find a sample for each requested k pair (or its transpose if missing)
    val_set = ds['val']
    by_k = {}
    for s in val_set:
        by_k.setdefault((int(s['k1']), int(s['k2'])), s)

    chosen = []
    for kp in target_ks:
        if kp in by_k: chosen.append(by_k[kp])
        elif kp[::-1] in by_k: chosen.append(by_k[kp[::-1]])
        else: print(f"  (no val sample with k={kp}; skipping)")

    print(f"Selected {len(chosen)} cases:",
          [(int(s['k1']), int(s['k2'])) for s in chosen])

    for s in chosen:
        name = f"detailed_k{int(s['k1'])}_{int(s['k2'])}.png"
        render_case(s, ae_model, ae_params, mlp_model, mlp_params,
                     encode_F, mlp_predict, out_dir / name)

    print(f"\nDone. Plots in {out_dir}/")


if __name__ == '__main__':
    main()
