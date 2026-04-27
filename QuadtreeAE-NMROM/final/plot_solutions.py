"""For each test case in the val set, render a 4-panel grid:
   analytical u | NM-ROM warm (full) | NM-ROM warm (fine-cells) | MLP only.

Plus a 6×4 master grid combining several cases.

Usage:
  python plot_solutions.py --exp-dir runs/expN [--n-cases 6]
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


def draw_field(ax, leaf_list, vals, title, vmin, vmax, cmap='RdBu_r'):
    if cmap == 'RdBu_r':
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    cmap_obj = plt.get_cmap(cmap)
    rng = max(vmax - vmin, 1e-30)
    for li, (xc, yc, h, *_) in enumerate(leaf_list):
        t = (vals[li] - vmin) / rng
        ax.add_patch(patches.Rectangle(
            (xc - h/2, yc - h/2), h, h,
            facecolor=cmap_obj(np.clip(t, 0, 1)), edgecolor='none'))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-dir', required=True)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--n-cases', type=int, default=6)
    args = p.parse_args()

    exp = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp / 'plots'
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

    test_set = ds['val'][:args.n_cases]
    results = []
    for i, sample in enumerate(test_set):
        qt = sample['qt']
        pd_f = M.topology_to_padded(qt, which='f')
        K_op, f_vec, u_ref, N, gidx, leaf_list = M.build_fvm_operator(qt)
        leaf_depths = np.array([lf[3] for lf in leaf_list], np.int32)
        max_d = int(leaf_depths.max())
        fine_mask = (leaf_depths == max_d).astype(np.float32)

        z_F = encode_F(pd_f); z_u0 = mlp_predict(z_F)
        leaf_gather = M.build_leaf_gather(qt, gidx)
        @jax.jit
        def decode_z(z):
            E = M.z_to_E(z)
            _, vpred = ae_model.apply({'params': ae_params}, E, pd_f, method=ae_model.decode)
            return M.extract_leaf_values(vpred, leaf_gather) / M.U_SCALE
        u_mlp = np.array(decode_z(z_u0))

        solve_warm, _ = M.make_nmrom_solver(ae_model, ae_params, pd_f, qt, K_op, f_vec, gidx)
        _ = solve_warm(z_init=z_u0)
        _, u_warm, hist_w = solve_warm(z_init=z_u0)
        u_warm = np.array(u_warm)

        solve_fine, _ = M.make_nmrom_solver(ae_model, ae_params, pd_f, qt, K_op, f_vec, gidx,
                                             residual_weights=fine_mask, lam_reg=1e-3, z_anchor=z_u0)
        _ = solve_fine(z_init=z_u0)
        _, u_fine, hist_f = solve_fine(z_init=z_u0)
        u_fine = np.array(u_fine)

        u_ref_np = np.array(u_ref)
        rel = lambda u: float(np.linalg.norm(u - u_ref_np) / np.linalg.norm(u_ref_np))
        results.append({
            'k1': qt['k1'], 'k2': qt['k2'], 'leaf_list': leaf_list,
            'u_ref': u_ref_np, 'u_warm': u_warm, 'u_fine': u_fine, 'u_mlp': u_mlp,
            'iters_w': len(hist_w), 'iters_f': len(hist_f),
            'rel_w': rel(u_warm), 'rel_f': rel(u_fine), 'rel_m': rel(u_mlp),
        })
        print(f"Case {i+1}: k=({qt['k1']:.0f},{qt['k2']:.0f}) "
              f"rel-L2 warm={results[-1]['rel_w']:.3e} fine={results[-1]['rel_f']:.3e} "
              f"mlp={results[-1]['rel_m']:.3e}")

    # ── Big 6×4 grid ────────────────────────────────────────────────────
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    if n == 1: axes = axes.reshape(1, -1)
    for i, r in enumerate(results):
        vmin = float(r['u_ref'].min()); vmax = float(r['u_ref'].max())
        draw_field(axes[i, 0], r['leaf_list'], r['u_ref'],
                   f"u_analytical  k=({r['k1']:.0f},{r['k2']:.0f})", vmin, vmax)
        draw_field(axes[i, 1], r['leaf_list'], r['u_warm'],
                   f"NM-ROM WARM full  rel-L2={r['rel_w']:.2e} ({r['iters_w']} iters)", vmin, vmax)
        draw_field(axes[i, 2], r['leaf_list'], r['u_fine'],
                   f"NM-ROM WARM fine  rel-L2={r['rel_f']:.2e} ({r['iters_f']} iters)", vmin, vmax)
        draw_field(axes[i, 3], r['leaf_list'], r['u_mlp'],
                   f"MLP only  rel-L2={r['rel_m']:.2e}", vmin, vmax)

    plt.suptitle(f'F → u via NM-ROM: 4-method comparison ({n} cases)', fontsize=13, y=1.0)
    plt.tight_layout()
    out_grid = out_dir / 'solutions_grid.png'
    plt.savefig(out_grid, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out_grid}")


if __name__ == '__main__':
    main()
