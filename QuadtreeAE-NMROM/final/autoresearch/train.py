"""autoresearch/train.py — single-file experiment runner.

EDIT the CONFIG dict below, then run `python train.py` from this directory.

The script will:
  1. Generate (F, u) dataset if not already cached
  2. Train the autoencoder if not already cached
  3. Train the latent MLP if not already cached
  4. Run the NM-ROM benchmark with the configured solver
  5. Append a single row to results.tsv with the full config + metrics

Caching keys are content-derived hashes of the relevant sub-config slice, so
changing only solver knobs (lam_reg, residual_strategy, …) skips the
expensive AE training step.

Goal: make `goal_met == True` happen for some CONFIG. See program.md.
"""
from pathlib import Path
import json, hashlib, time, sys, pickle, os
import numpy as np

# ─── CONFIG (edit to try a new experiment) ────────────────────────────────────
CONFIG = {
    'tag':               'baseline',

    # Data
    'n_train':           200,
    'n_val':             40,
    'k_lo':              1,            # INTEGER — see program.md
    'k_hi':              5,
    'grad_thresh':       0.5,
    'min_level':         6,
    'max_level':         7,
    'target':            'analytical',  # 'analytical' or 'fom'

    # AE
    'ae_steps':          8000,
    'ae_lr':             1e-3,
    'hidden':            128,
    'emb_dim':           6,
    'u_scale':           50.0,

    # MLP
    'mlp_hidden':        512,
    'mlp_layers':        3,
    'mlp_steps':         5000,

    # NM-ROM
    'lam_reg':           1e-3,           # Tikhonov anchor; do NOT set 0 with masked residual
    'max_outer':         8,
    'max_inner':         6,
    'gn_tol':            1e-5,
    'residual_strategy': 'fine_only',    # 'full' | 'fine_only' | 'soft_fine'

    # Bench
    'n_test':            6,
    'seed':              0,

    'notes':             'baseline production config (n_train=200, ae_steps=8000)',
}

# ─── Goal predicates ──────────────────────────────────────────────────────────
def goal_predicate(rel_full, rel_fine, rel_coarse):
    """Return (goal_met: bool, why: str)."""
    if rel_full >= 1e-2:
        return False, f"rel_full={rel_full:.3e} ≥ 1e-2"
    if rel_fine > 0.5 * rel_coarse:
        return False, f"fine/coarse={rel_fine/max(rel_coarse,1e-30):.2f} > 0.5"
    return True, "rel_full<1e-2 and fine ≤ 0.5×coarse"


# ─── Bookkeeping ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
RUNS = ROOT / 'runs'
RUNS.mkdir(exist_ok=True)

# Make f_to_u_nmrom importable from the parent dir
sys.path.insert(0, str(ROOT.parent))


def cfg_hash(d):
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def data_cfg(c):
    return {k: c[k] for k in
            ('n_train','n_val','k_lo','k_hi','grad_thresh',
             'min_level','max_level','target','seed')}


def ae_cfg(c):
    return {**data_cfg(c),
            **{k: c[k] for k in
               ('ae_steps','ae_lr','hidden','emb_dim','u_scale')}}


def mlp_cfg(c):
    return {**ae_cfg(c),
            **{k: c[k] for k in ('mlp_hidden','mlp_layers','mlp_steps')}}


def setup_module_constants(c):
    """Patch module-level constants in f_to_u_nmrom that affect data shapes,
    BEFORE importing the rest of the pipeline. Must be called once per process."""
    import f_to_u_nmrom as M
    M.MAX_LEVEL  = c['max_level']
    M.MIN_LEVEL  = c['min_level']
    M.EMB_DIM    = c['emb_dim']
    M.HIDDEN     = c['hidden']
    M.U_SCALE    = c['u_scale']
    M.DEPTH_SIZES = [4**d for d in range(M.MAX_LEVEL + 1)]
    M.LATENT_DIM  = sum(M.DEPTH_SIZES[:M.D_BOTTLE + 1]) * M.EMB_DIM
    import jax.numpy as jnp
    _scales = []
    for _d in range(M.D_BOTTLE + 1):
        _scales.extend([float(2**(M.D_BOTTLE - _d))] * (M.DEPTH_SIZES[_d] * M.EMB_DIM))
    M.DEPTH_SCALE_VEC = jnp.array(_scales, dtype=jnp.float32)
    return M


# ─── Stages ───────────────────────────────────────────────────────────────────
def stage_data(M, c, data_dir):
    pkl = data_dir / 'paired_dataset.pkl'
    if pkl.exists():
        print(f"[data] reuse  {pkl.parent.name}/")
        return pkl
    print(f"[data] generating  {data_dir.name}/   ({c['n_train']} + {c['n_val']} trees, target={c['target']})")
    data_dir.mkdir(parents=True, exist_ok=True)
    M.generate_paired_dataset(
        n_train=c['n_train'], n_val=c['n_val'],
        k_lo=c['k_lo'], k_hi=c['k_hi'], seed=c['seed'],
        grad_thresh=c['grad_thresh'], save_path=pkl, target=c['target'])
    return pkl


def stage_ae(M, c, data_pkl, ae_dir):
    ae_pkl = ae_dir / 'shared_ae.pkl'
    if ae_pkl.exists():
        print(f"[ae]   reuse  {ae_dir.name}/  (val rel-L2 in checkpoint history)")
        return ae_pkl
    print(f"[ae]   training  {ae_dir.name}/   ({c['ae_steps']} steps, hidden={c['hidden']}, emb_dim={c['emb_dim']})")
    ae_dir.mkdir(parents=True, exist_ok=True)
    (ae_dir / 'plots').mkdir(exist_ok=True)
    with open(data_pkl, 'rb') as f: d = pickle.load(f)
    M.train_shared_ae(
        d['train'], d['val'],
        n_steps=c['ae_steps'], lr=c['ae_lr'], seed=c['seed'],
        out_dir=ae_dir, save_path=ae_pkl,
        hidden=c['hidden'], emb_dim=c['emb_dim'])
    return ae_pkl


def stage_mlp(M, c, data_pkl, ae_pkl, mlp_dir):
    mlp_pkl = mlp_dir / 'mlp_z_f_to_z_u.pkl'
    if mlp_pkl.exists():
        print(f"[mlp]  reuse  {mlp_dir.name}/")
        return mlp_pkl
    print(f"[mlp]  training  {mlp_dir.name}/  ({c['mlp_steps']} steps)")
    mlp_dir.mkdir(parents=True, exist_ok=True)
    with open(data_pkl, 'rb') as f: d = pickle.load(f)
    with open(ae_pkl, 'rb') as f: ae = pickle.load(f)
    ae_model = M.CompressedBottleneckAE(
        hidden=ae.get('hidden', c['hidden']),
        emb_dim=ae.get('emb_dim', c['emb_dim']))
    print('       encoding train / val pairs ...')
    Z_F_tr, Z_u_tr = M.encode_pairs(ae_model, ae['params'], d['train'])
    Z_F_va, Z_u_va = M.encode_pairs(ae_model, ae['params'], d['val'])
    M.train_mlp(Z_F_tr, Z_u_tr, Z_F_va, Z_u_va,
                hidden=c['mlp_hidden'], n_layers=c['mlp_layers'],
                n_steps=c['mlp_steps'], lr=1e-3, seed=c['seed'],
                save_path=mlp_pkl)
    return mlp_pkl


def stage_bench(M, c, data_pkl, ae_pkl, mlp_pkl, exp_dir):
    """Run the bench with the configured solver and return summary metrics."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)

    # Mirror the bench logic from f_to_u_nmrom.main(), but with the configured
    # lam_reg / residual_strategy passed through.
    import jax, jax.numpy as jnp
    from jax import jit
    import jax.scipy.sparse.linalg as jspla
    import numpy as np

    with open(data_pkl, 'rb') as f: d = pickle.load(f)
    with open(ae_pkl,   'rb') as f: ae = pickle.load(f)
    with open(mlp_pkl,  'rb') as f: ml = pickle.load(f)

    ae_model  = M.CompressedBottleneckAE(
        hidden=ae.get('hidden', c['hidden']),
        emb_dim=ae.get('emb_dim', c['emb_dim']))
    ae_params = ae['params']
    mlp_model = M.LatentMLP(hidden=ml['hidden'], n_layers=ml['n_layers'])
    mlp_params = ml['params']

    @jax.jit
    def encode_F(pd_f):
        E = ae_model.apply({'params': ae_params}, pd_f, method=ae_model.encode)
        return M.E_to_z(E)

    @jax.jit
    def mlp_predict(z_F):
        return mlp_model.apply({'params': mlp_params}, z_F)

    rows = []
    test_set = d['val'][:c['n_test']]
    for i, sample in enumerate(test_set):
        qt = sample['qt']
        pd_f = M.topology_to_padded(qt, which='f')
        K_op, f_vec, u_ref, N, gidx, leaf_list = M.build_fvm_operator(qt)
        leaf_depths = np.array([lf[3] for lf in leaf_list], np.int32)
        max_d = int(leaf_depths.max())

        if c['residual_strategy'] == 'full':
            w = np.ones(N, np.float32)
        elif c['residual_strategy'] == 'fine_only':
            w = (leaf_depths == max_d).astype(np.float32)
        elif c['residual_strategy'] == 'soft_fine':
            w = np.where(leaf_depths == max_d, 1.0, 0.1).astype(np.float32)
        else:
            raise ValueError(f"unknown residual_strategy={c['residual_strategy']!r}")
        fine_mask  = (leaf_depths == max_d).astype(np.float32)
        coarse_mask = 1 - fine_mask

        # MLP-only prediction
        z_F = encode_F(pd_f); z_u0 = mlp_predict(z_F)
        leaf_gather = M.build_leaf_gather(qt, gidx)
        @jax.jit
        def decode_z(z):
            E = M.z_to_E(z)
            _, vp = ae_model.apply({'params': ae_params}, E, pd_f, method=ae_model.decode)
            return M.extract_leaf_values(vp, leaf_gather) / M.U_SCALE
        u_mlp = np.array(decode_z(z_u0))

        # NM-ROM with the configured weights + Tikhonov
        solver, _ = M.make_nmrom_solver(
            ae_model, ae_params, pd_f, qt, K_op, f_vec, gidx,
            max_outer=c['max_outer'], max_inner=c['max_inner'], tol=c['gn_tol'],
            residual_weights=w,
            lam_reg=c['lam_reg'], z_anchor=z_u0)
        _ = solver(z_init=z_u0)            # JIT warm
        t0 = time.perf_counter()
        _, u_pred, hist = solver(z_init=z_u0)
        u_pred.block_until_ready()
        t_solve = time.perf_counter() - t0
        u_pred = np.array(u_pred)

        # FOM (CG) for reference
        @jit
        def cg_solve(b):
            sol, _ = jspla.cg(K_op, b, x0=jnp.zeros_like(b), tol=1e-8, maxiter=2000)
            return sol
        _ = cg_solve(f_vec)
        t0 = time.perf_counter()
        u_fom = cg_solve(f_vec); u_fom.block_until_ready()
        t_fom = time.perf_counter() - t0

        u_ref_np = np.array(u_ref)
        def rel(u_arr, mask=None):
            ua = np.array(u_arr)
            if mask is None:
                return float(np.linalg.norm(ua - u_ref_np) / max(np.linalg.norm(u_ref_np), 1e-30))
            num = np.sqrt(np.sum(mask * (ua - u_ref_np)**2))
            den = np.sqrt(np.sum(mask * u_ref_np**2))
            return float(num / max(den, 1e-30))

        rel_full   = rel(u_pred)
        rel_fine   = rel(u_pred, fine_mask)
        rel_coarse = rel(u_pred, coarse_mask) if coarse_mask.sum() > 0 else 0.0
        rel_mlp    = rel(u_mlp)
        rel_fom    = rel(u_fom)
        rows.append(dict(
            i=i+1, k1=qt['k1'], k2=qt['k2'], N=N,
            rel_full=rel_full, rel_fine=rel_fine, rel_coarse=rel_coarse,
            rel_mlp=rel_mlp, rel_fom=rel_fom,
            iters=len(hist), t_solve=t_solve, t_fom=t_fom))
        print(f"  case {i+1} k=({int(qt['k1'])},{int(qt['k2'])}) rel_full={rel_full:.3e} fine={rel_fine:.3e} coarse={rel_coarse:.3e} (iters={len(hist)} {t_solve:.2f}s)")

    # Aggregate
    avg = {f'avg_{k}': float(np.mean([r[k] for r in rows]))
           for k in ('rel_full','rel_fine','rel_coarse','rel_mlp','rel_fom',
                     'iters','t_solve','t_fom')}
    return rows, avg


# ─── results.tsv writer ───────────────────────────────────────────────────────
RESULTS_TSV = ROOT / 'results.tsv'
HEADER = ('tag\thash\tn_train\tae_steps\thidden\temb_dim\tu_scale\ttarget\tlam_reg'
          '\tresidual_strategy\trel_full\trel_fine\trel_coarse\tratio_fine_coarse'
          '\trel_mlp\trel_fom\tavg_iters\tavg_t_solve\tgoal_met\twhy\tnotes\n')


def append_results(c, h, avg, goal_met, why):
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(HEADER)
    ratio = avg['avg_rel_fine'] / max(avg['avg_rel_coarse'], 1e-30)
    line = (f"{c['tag']}\t{h}\t{c['n_train']}\t{c['ae_steps']}\t{c['hidden']}\t{c['emb_dim']}"
            f"\t{c['u_scale']}\t{c['target']}\t{c['lam_reg']:.1e}\t{c['residual_strategy']}"
            f"\t{avg['avg_rel_full']:.4e}\t{avg['avg_rel_fine']:.4e}\t{avg['avg_rel_coarse']:.4e}"
            f"\t{ratio:.3f}\t{avg['avg_rel_mlp']:.4e}\t{avg['avg_rel_fom']:.4e}"
            f"\t{avg['avg_iters']:.1f}\t{avg['avg_t_solve']:.3f}"
            f"\t{goal_met}\t{why}\t{c['notes']}\n")
    with open(RESULTS_TSV, 'a') as f: f.write(line)


def main():
    c = dict(CONFIG)   # copy
    M = setup_module_constants(c)
    h = cfg_hash(c)
    print(f"=== experiment {h}  tag={c['tag']!r} ===")

    data_dir = RUNS / f"data_{cfg_hash(data_cfg(c))}"
    ae_dir   = RUNS / f"ae_{cfg_hash(ae_cfg(c))}"
    mlp_dir  = RUNS / f"mlp_{cfg_hash(mlp_cfg(c))}"
    exp_dir  = RUNS / f"exp_{h}"

    data_pkl = stage_data(M, c, data_dir)
    ae_pkl   = stage_ae  (M, c, data_pkl, ae_dir)
    mlp_pkl  = stage_mlp (M, c, data_pkl, ae_pkl, mlp_dir)
    rows, avg = stage_bench(M, c, data_pkl, ae_pkl, mlp_pkl, exp_dir)

    print(f"\n=== summary ({len(rows)} cases) ===")
    print(f"  rel_full   = {avg['avg_rel_full']:.4e}")
    print(f"  rel_fine   = {avg['avg_rel_fine']:.4e}")
    print(f"  rel_coarse = {avg['avg_rel_coarse']:.4e}")
    print(f"  ratio fine/coarse = {avg['avg_rel_fine']/max(avg['avg_rel_coarse'],1e-30):.3f}")
    print(f"  rel_mlp    = {avg['avg_rel_mlp']:.4e}    rel_fom = {avg['avg_rel_fom']:.4e}")
    print(f"  iters      = {avg['avg_iters']:.1f}    t_solve = {avg['avg_t_solve']:.3f}s")

    goal_met, why = goal_predicate(avg['avg_rel_full'], avg['avg_rel_fine'], avg['avg_rel_coarse'])
    print(f"\n  GOAL_MET = {goal_met}    ({why})")

    append_results(c, h, avg, goal_met, why)
    print(f"\n  wrote row to {RESULTS_TSV.relative_to(ROOT)}")
    return goal_met


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
