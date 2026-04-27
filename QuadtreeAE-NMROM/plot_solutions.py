"""Run NM-ROM warm + cold on each of the 6 uniform-mesh test cases and produce
per-case plots: u_ref (analytical) | NM-ROM warm | NM-ROM cold | error(warm).

Then a 6x4 summary grid showing all of them at once."""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import f_to_u_nmrom as M
import sys
# Pickled trees reference _QuadNode by its original module path '__main__'
sys.modules['__main__']._QuadNode = M._QuadNode

print(f"JAX devices: {jax.devices()}")

EXPDIR = Path('f_to_u_runs/exp4_uniform')
ds = pickle.load(open(EXPDIR / 'paired_dataset.pkl', 'rb'))
ae_ck  = pickle.load(open(EXPDIR / 'shared_ae.pkl', 'rb'))
mlp_ck = pickle.load(open(EXPDIR / 'mlp_z_f_to_z_u.pkl', 'rb'))

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

# Use first 6 val samples
test_set = ds['val'][:6]

# ── Solve all 6 cases ────────────────────────────────────────────────────
results = []
for i, sample in enumerate(test_set):
    qt = sample['qt']
    pd_f = M.topology_to_padded(qt, which='f')
    K_op, f_vec, u_ref, N, gidx, leaf_list = M.build_fvm_operator(qt)
    print(f"\nCase {i+1}: k=({sample['k1']:.0f}, {sample['k2']:.0f})  N={N}")

    solver, _ = M.make_nmrom_solver(ae_model, ae_params, pd_f, qt, K_op, f_vec, gidx,
                                    max_outer=8, max_inner=6, tol=1e-5)
    # warmup
    _ = solver(z_init=jnp.zeros(M.LATENT_DIM))
    z_F = encode_F(pd_f)
    z_u0 = mlp_predict(z_F)

    _, u_warm, hist_w = solver(z_init=z_u0)
    _, u_cold, hist_c = solver(z_init=jnp.zeros(M.LATENT_DIM))

    # decode MLP guess directly
    leaf_gather = M.build_leaf_gather(qt, gidx)
    @jax.jit
    def decode_z(z):
        E = M.z_to_E(z)
        _, vpred = ae_model.apply({'params': ae_params}, E, pd_f, method=ae_model.decode)
        return M.extract_leaf_values(vpred, leaf_gather) / M.U_SCALE
    u_mlp = decode_z(z_u0)

    results.append({
        'k1': sample['k1'], 'k2': sample['k2'],
        'u_ref':  np.array(u_ref),
        'u_warm': np.array(u_warm),
        'u_cold': np.array(u_cold),
        'u_mlp':  np.array(u_mlp),
        'leaf_list': leaf_list,
        'iters_warm': len(hist_w), 'iters_cold': len(hist_c),
    })

# ── 1) High-quality 4-panel plots per-case (single PNG each) ─────────────
def draw_field(ax, leaf_list, vals, title, vmin, vmax, cmap='RdBu_r'):
    for li, (xc, yc, h, *_) in enumerate(leaf_list):
        norm_v = (vals[li] - vmin) / max(vmax - vmin, 1e-30)
        c = plt.cm.get_cmap(cmap)(norm_v)
        ax.add_patch(plt.Rectangle((xc - h/2, yc - h/2), h, h,
                                    facecolor=c, edgecolor='none'))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# ── 2) Big summary grid: 6 cases × 4 columns (ref, warm, cold, err_warm) ─
fig, axes = plt.subplots(6, 4, figsize=(14, 20))
for i, r in enumerate(results):
    vmin = float(r['u_ref'].min()); vmax = float(r['u_ref'].max())
    err = np.abs(r['u_warm'] - r['u_ref'])
    e_max = float(err.max())
    rel_w = float(np.linalg.norm(r['u_warm'] - r['u_ref']) /
                  np.linalg.norm(r['u_ref']))
    rel_c = float(np.linalg.norm(r['u_cold'] - r['u_ref']) /
                  np.linalg.norm(r['u_ref']))

    draw_field(axes[i, 0], r['leaf_list'], r['u_ref'],
               f"u_analytical  k=({r['k1']:.0f},{r['k2']:.0f})", vmin, vmax)
    draw_field(axes[i, 1], r['leaf_list'], r['u_warm'],
               f"NM-ROM WARM  rel-L2={rel_w:.2e}  iters={r['iters_warm']}", vmin, vmax)
    draw_field(axes[i, 2], r['leaf_list'], r['u_cold'],
               f"NM-ROM COLD  rel-L2={rel_c:.2e}  iters={r['iters_cold']}", vmin, vmax)
    draw_field(axes[i, 3], r['leaf_list'], err,
               f"|warm - analytical|  max={e_max:.2e}", 0, e_max, cmap='hot')

plt.suptitle('Uniform 64×64 mesh — 6 test cases  (warm = MLP warm-start, cold = z=0 init)',
             fontsize=13, y=0.998)
plt.tight_layout()
plt.savefig(EXPDIR / 'all_cases_grid.png', dpi=130, bbox_inches='tight')
print(f"\nSaved {EXPDIR / 'all_cases_grid.png'}")
plt.close()

# ── 3) Best-case 3-panel: just ref, warm, error ─────────────────────────
# Pick case 1 (k=(3,2)) as a clean example
r = results[0]
vmin = float(r['u_ref'].min()); vmax = float(r['u_ref'].max())
err  = np.abs(r['u_warm'] - r['u_ref']); e_max = float(err.max())
rel_w = float(np.linalg.norm(r['u_warm'] - r['u_ref']) / np.linalg.norm(r['u_ref']))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
draw_field(axes[0], r['leaf_list'], r['u_ref'],
           f"Analytical  u  for  k=({r['k1']:.0f},{r['k2']:.0f})", vmin, vmax)
draw_field(axes[1], r['leaf_list'], r['u_warm'],
           f"NM-ROM warm prediction  (rel-L2 = {rel_w:.2e})", vmin, vmax)
draw_field(axes[2], r['leaf_list'], err,
           f"Absolute error  (max = {e_max:.2e})", 0, e_max, cmap='hot')
plt.tight_layout()
plt.savefig(EXPDIR / 'best_case_3panel.png', dpi=150, bbox_inches='tight')
print(f"Saved {EXPDIR / 'best_case_3panel.png'}")
plt.close()
