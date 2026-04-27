# F → u NM-ROM on Adaptive Quadtree — Progress Log

## Goal

Given a forcing function `F` defined on an adaptive 2D quadtree, recover the
solution `u` of the Poisson equation

```
-∇²u = F   on [0,1]²,    u = 0 on ∂Ω
F(x,y; k1,k2) = sin(k1·π·x)·sin(k2·π·y)        (k1, k2 ∈ {1..5}²)
u_analytical = F / ((k1² + k2²)·π²)            (closed form for integer k)
```

We do *not* want to run the full FVM solve at inference. Instead, we build a
neural-network operator that maps `F → u` and refine its prediction with a
**Non-linear Manifold Reduced-Order Model (NM-ROM)** — a Gauss-Newton solve in
a 510-dim latent space.

## Architecture

```
F  ──► encoder ──► z_F (510 floats)
                      │
                      ▼
                    MLP (3-layer, hidden 512)
                      │
                      ▼
                    z_u₀ (510 floats)
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
         decoder         Gauss-Newton in latent space:
         u₀  ◄──         minimise ‖W(K·decode(z) − F·V)‖² + λ‖z − z_u₀‖²
       (MLP only)        where W is a per-leaf weight (1 on fine cells, 0 on coarse)
                         and λ‖·‖² is a Tikhonov anchor to z_u₀
                                                     │
                                                     ▼
                                                   z* ──► decoder ──► u*
```

### The autoencoder

Same architecture as `quadtree_nmrom_jax.py`'s `CompressedBottleneckAE`:

- **Encoder.** Per-depth input projection of `[leaf_value, fourier_pos_encoding]` →
  hidden=128. Bottom-up pooling: at each depth, mean-pool children into parents,
  add to existing parent features, mix with a 9-neighbour `QuadConv`. Bottleneck
  outputs `E[d] ∈ ℝ^{4ᵈ × 6}` for `d = 0, 1, 2, 3` — total **510 floats**.
- **Decoder.** Top-down: `[h_decoder, ancestor_skip(E[D_BOTTLE]), pos]` →
  `fuse → mix_conv → val_head`. Children spawned via `child_head` until depth 7.
- **D_BOTTLE = 3** (so 1+4+16+64 = 85 nodes at the bottleneck).
- **EMB_DIM = 6**, **HIDDEN = 128**.
- Latent dim = 85·6 = **510**.

### Training data

- 200 `(F-tree, u-tree)` pairs at training time, 40 at val.
- Random integer (k1, k2) ∈ {1..5}² (so **u = 0 on the boundary** is satisfied
  exactly by the analytical formula — see Bug 3 below).
- Tree refinement: depth-based, force-refine to MIN_LEVEL=6 then refine to
  MAX_LEVEL=7 wherever |∇F|/max|∇F| > 0.5. ~5000–10000 leaves per tree.
- Both F and u are stored on the same tree (one tree per sample, two views).

### Loss

MSE on per-depth leaf nodes, averaged across depths. With `U_SCALE = 50`
applied to u-values during training (so amplitudes are comparable to F):

```python
L = mean over depths of   mean_leaves((u_pred - u_gt)²)
```

### MLP

3-layer MLP, hidden 512, GeLU activations, MSE loss between `encode(F)` and
`encode(u·U_SCALE)` after the AE is trained. Tiny: ~1M params, trains in ~12 s.

### NM-ROM solver

Gauss-Newton in latent space with conjugate-gradient inner solver, depth-scaled
diagonal preconditioner, optional residual masking and Tikhonov regularisation:

```
Outer GN iter (max 8 outer × 6 inner = 48 total iters, JIT-compiled):
  u_pred       = decode(z) / U_SCALE
  r            = W * (K_op(u_pred) - f_vec)             # weighted residual
  g_red        = J_decode^T r + λ(z - z_anchor)         # Tikhonov gradient
  GN_op(dz)    = J_decode^T (W * K J_decode dz) + λ I dz
  dz           = CG(GN_op, -g_red, tol=1e-4, maxiter=100)
  z           += dz
  break if ‖r‖ < tol or ‖g_red‖ < tol
```

`W` is the per-leaf residual weight. **W=1 everywhere** is the standard
NM-ROM. **W=fine_mask** (1 on depth-MAX_LEVEL leaves, 0 on coarser ones)
restricts the PDE constraint to fine cells — the design that wins on
this problem.

Without Tikhonov (`lam_reg=0`), the masked solver is under-determined in z
and GN drifts wildly (we observed rel-L2 ~5 on the first attempt).
With Tikhonov toward the MLP guess (`lam_reg=1e-3, z_anchor=z_MLP`), the
latent stays close to the MLP prediction; GN converges in 3-7 iters.

### FVM operator

Symmetric assembly of the discrete Poisson operator on the leaves:

```
K_ii = +Σ_face cond_face                       (positive diagonal)
K_ij = -cond_face_ij                           (negative off-diagonal, symmetric)
rhs  = f · |V_c|                               (volume-scaled per leaf)
cond = face_length / face_distance              (handles same / coarser / finer faces)
```

Boundary face contribution: `cond_b = h/(h/2) = 2` added to diagonal (Dirichlet
u=0 enforced at the cell-face midpoint).

CG against this K gives the discrete-FVM solution `u_FOM` (the "FOM" reference).

## Results

### Final benchmark — adaptive integer-k mesh, 6 test cases, avg over cases

| Method                    | full-mesh rel-L2 | fine-cells rel-L2 | coarse-cells rel-L2 | iters | time/case |
|---------------------------|------------------|-------------------|---------------------|-------|-----------|
| COLD start                | 0.414            | 0.480             | 0.328               | 48    | 27.0 s    |
| WARM (full residual)      | 0.382            | 0.416             | 0.328               | 31    | 17.6 s    |
| **WARM (fine-cells-only)**| **0.113**        | **0.123**         | **0.097**           | **4.8** | **0.47 s** |
| MLP only                  | 0.101            | 0.114             | 0.079               | –     | –         |
| FOM (CG on FVM)           | 0.221            | 0.225             | –                   | –     | 0.04 s    |

The fine-cells-only NM-ROM is:
- **3.4× more accurate** than warm-full on the full mesh
- **2× better than the FOM** on the fine-cell metric (because anchoring coarse
  cells to MLP avoids the FVM's T-junction truncation error)
- Converges in **3–7 iters in 0.47 s** vs full residual's 31 iters in 17.6 s
  (37× wall-clock speedup)

It does *not* beat MLP-only on rel-L2-vs-analytical for this PDE family,
because the MLP is trained directly to reproduce u_analytical and the
discrete FVM's optimum (u_FOM) differs from u_analytical by ~0.22 due to
T-junction truncation. NM-ROM correctly minimises PDE residual but that
target is slightly off-truth on this mesh. The expected wins for NM-ROM:
out-of-distribution forcings (where MLP extrapolates poorly) and PDEs
where the discrete operator IS accurate (uniform meshes, or proper
multi-point flux at T-junctions).

## What we got working / fixed along the way

Five real bugs were found and fixed before the pipeline could be honestly
evaluated:

| # | Bug                                              | Symptom                                                                                     | Fix                                              |
|---|--------------------------------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------|
| 1 | F-vs-u amplitude imbalance (50× scale)           | u rel-L2 = 25 at step 200 — MSE dominated by F                                              | `U_SCALE = 50` rescaling on the u-view           |
| 2 | Non-symmetric K (off-diagonals scaled by V_c)    | FOM CG diverges, residual = 100+                                                            | Symmetric assembly + volume-scaled RHS           |
| 3 | Continuous-k boundary condition mismatch         | sin(k·π·x) only zeros at x=1 for **integer** k; non-integer k → formula violates BC → FOM disagrees with formula by 40% | Integer k ∈ {1..5}²                              |
| 4 | Eager Python GN loop                             | 365 s per case (vjp re-traces every iter)                                                   | `@jit gn_step` → 26 s per case (14× speedup)     |
| 5 | T-junction first-order FVM                       | FOM rel-L2 vs analytical ≈ 0.22 even with all bugs above fixed                              | (open) — switch to uniform mesh as clean baseline; or accept it as a discretisation limitation |
| 6 | Restricted-residual NM-ROM is under-determined   | First fine-cell-only attempt gave rel-L2 = 5 (random latent stationary point)               | Tikhonov regulariser λ‖z − z_MLP‖²              |

Bug 3 was the load-bearing one: the JOURNEY.md from the prior NM-ROM work
reported "rel-L2 ≈ 2" which was the signature of comparing FVM-correct
solutions against an analytical formula that didn't satisfy the BCs.
One-line fix.

## File layout

```
final/
├── PROGRESS.md                  ← this file
├── README.md                    ← quickstart commands
├── f_to_u_nmrom.py              ← main pipeline (data, train_ae, train_mlp, bench, all)
├── plot_finecell_summary.py     ← bar chart from benchmark_results.tsv
├── plot_per_case_detailed.py    ← per-case 8-panel: input/gt/predictions/error/histogram
├── plot_solutions.py            ← per-case 4-panel grid (analytical / WARM-full / WARM-fine / MLP)
└── runs/
    └── <run_name>/              ← one directory per experiment
        ├── paired_dataset.pkl       — (n_train + n_val) (F, u) pairs
        ├── shared_ae.pkl            — AE checkpoint
        ├── mlp_z_f_to_z_u.pkl       — MLP checkpoint
        ├── benchmark_results.tsv    — per-case + AVG metrics
        ├── logs/
        │   ├── data.log
        │   ├── ae_train.log
        │   ├── mlp_train.log
        │   └── bench.log
        └── plots/
            ├── ae_loss.png
            ├── finecell_summary.png
            ├── solutions_grid.png
            ├── bench/case_NN.png
            └── detailed/detailed_kK1_K2.png
```

## How to reproduce the result

End-to-end (one command, ~25 min on a single GPU with JAX):

```bash
python f_to_u_nmrom.py all --out runs/exp_full --n-train 200 --n-val 40 --ae-steps 8000 --n-test 6
python plot_finecell_summary.py --exp-dir runs/exp_full
python plot_per_case_detailed.py --exp-dir runs/exp_full
python plot_solutions.py --exp-dir runs/exp_full
```

Or stage by stage (see README.md).
