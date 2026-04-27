# autoresearch — F → u NM-ROM on Adaptive Quadtree

You are an autonomous research agent. Your job is to iterate on a fixed PDE
pipeline until it meets the success criteria below.

## The problem

Given a forcing function `F` on an adaptive 2D quadtree mesh, recover the
solution `u` of the Poisson equation

```
−∇²u = F   on [0, 1]²,    u = 0 on ∂Ω
F(x, y; k1, k2) = sin(k1·π·x)·sin(k2·π·y)
u_analytical    = F / ((k1² + k2²)·π²)
k1, k2 ∈ {1, 2, 3, 4, 5}   (integer — non-integer k breaks the boundary condition)
```

The pipeline:

```
F  ──► encoder ──► z_F  ──► MLP ──► z_u₀  ──► decoder ──► u₀  (warm start)
                                                  │
                                                  ▼
                                       NM-ROM Gauss-Newton
                                       minimizes ‖W(K·u(z) − f)‖² + λ‖z − z_u₀‖²
                                                  │
                                                  ▼
                                                 u*   on the same quadtree
```

`W` is per-leaf weight (1 on fine leaves at MAX_LEVEL, 0 on coarser leaves) so
NM-ROM only enforces the discrete PDE at fine cells. `λ‖z − z_u₀‖²` is a
Tikhonov anchor that prevents the under-determined masked problem from
drifting wildly in latent space.

See `../PROGRESS.md` for full architecture details, the bugs we fixed
to get here, and the current baseline numbers.

## Success criteria

The autoresearch loop should iterate **until both of these hold** on a
6-case held-out benchmark:

1. **Avg full-mesh rel-L2 vs analytical < 1e-2** (i.e., 1% accuracy)
2. **Fine-cells rel-L2 ≤ 0.5 × coarse-cells rel-L2** (fine cells at least
   2× more accurate than coarse cells)

Both conditions get logged in `results.tsv` as `goal_met` (true/false). When
`goal_met=true` shows up for any experiment, the loop stops.

## Current baseline

Best result so far on the production config (200 train, 8000 AE steps,
fine-cell NM-ROM with λ=1e-3):

| metric                      | value  |
|-----------------------------|--------|
| full-mesh rel-L2            | 0.118  |
| fine-cells rel-L2           | 0.130  |
| coarse-cells rel-L2         | 0.099  |
| ratio fine / coarse         | 1.31   |  ← fine is *worse* than coarse, want < 0.5 |
| WARM-fine iters             | 6.0    |
| WARM-fine time per case     | 0.49 s |

So we need to push full-mesh from 0.118 → 0.01 (12× improvement) AND
flip the fine/coarse ratio from 1.31 → ≤ 0.5 (fine becoming much more
accurate than coarse).

## Where the bottleneck is

Two constraints are active:

1. **AE manifold quality.** The AE compresses ~9000 leaf values to a
   510-float latent. Validation rel-L2 ≈ 0.10 for u even after 8000 steps.
   No matter how good NM-ROM is, the decoded answer can't be more accurate
   than the AE's representation of u.
2. **FVM truncation at T-junctions.** Two-point flux on adaptive meshes is
   first-order at coarse-fine interfaces. The FOM (CG on the discrete K)
   has rel-L2 ≈ 0.22 vs analytical — meaning the discrete equation NM-ROM
   is solving has 22% intrinsic error from the truth.

Both must be addressed to break 1e-2.

## How to iterate

The single file you edit is `train.py`. It has a `CONFIG` dict at the top.
After editing, run `python train.py` from this directory.

`train.py` will:
1. Run any pipeline stages whose inputs changed (data → AE → MLP → bench)
2. Reuse cached artifacts where possible (keyed by config hash of relevant
   sub-config)
3. Append one row to `results.tsv` with the full config and metrics
4. Print whether `goal_met` was achieved

### Knobs you can edit in CONFIG

| key                  | type   | default       | what it does |
|----------------------|--------|---------------|--------------|
| `tag`                | str    | 'baseline'    | short human-readable name |
| `n_train`, `n_val`   | int    | 200, 40       | dataset size |
| `k_lo`, `k_hi`       | int    | 1, 5          | integer k range (don't go non-integer — see PROGRESS.md bug 3) |
| `grad_thresh`        | float  | 0.5           | refine fine-mesh where \|∇F\| / max\|∇F\| > thresh. lower = more depth-7 cells |
| `min_level`          | int    | 6             | minimum tree depth (forces uniform refinement to here) |
| `max_level`          | int    | 7             | maximum tree depth (where MAX_LEVEL refinement caps out) |
| `target`             | str    | 'analytical'  | u target. 'analytical' = closed form. 'fom' = per-sample CG solve |
| `ae_steps`           | int    | 8000          | AE training steps |
| `ae_lr`              | float  | 1e-3          | AE peak LR (cosine schedule) |
| `hidden`             | int    | 128           | AE hidden width (bigger = more capacity) |
| `emb_dim`            | int    | 6             | per-node bottleneck embedding dim. latent = 85·emb_dim |
| `u_scale`            | float  | 50.0          | u amplitude scale during training so MSE balances F vs u |
| `mlp_hidden`         | int    | 512           | MLP width |
| `mlp_layers`         | int    | 3             | MLP depth |
| `mlp_steps`          | int    | 5000          | MLP training steps |
| `lam_reg`            | float  | 1e-3          | Tikhonov anchor strength. ZERO breaks NM-ROM. Try 1e-4, 1e-2 |
| `max_outer`          | int    | 8             | GN outer iters cap |
| `max_inner`          | int    | 6             | inner iter cap (max_outer × max_inner = total GN budget) |
| `gn_tol`             | float  | 1e-5          | GN convergence tolerance on residual & gradient norms |
| `residual_strategy`  | str    | 'fine_only'   | 'full' (all leaves), 'fine_only' (only depth=MAX leaves), 'soft_fine' (w_coarse=0.1) |
| `n_test`             | int    | 6             | bench test cases |
| `notes`              | str    | ''            | freeform notes — visible in results.tsv |

### What NOT to change

- `k_lo`, `k_hi` must stay **integer** (continuous k violates the BC and
  the analytical formula stops being a valid reference — see PROGRESS.md
  bug 3).
- Don't set `lam_reg = 0` with `residual_strategy != 'full'` — the masked
  problem is under-determined in latent space and GN will return garbage
  (rel-L2 = 5+). See PROGRESS.md bug 6.
- Don't disable `u_scale` (it balances MSE between F and u amplitudes
  — see PROGRESS.md bug 1).

## Hypotheses worth trying (ordered roughly by expected impact)

1. **Bigger AE.** `hidden = 256` or `hidden = 384`. The AE is the manifold
   ceiling — bigger encoder/decoder should reach lower rel-L2.
2. **More AE training.** `ae_steps = 16000` or `24000`. Current curves
   show u-rel-L2 still slowly descending at step 8000.
3. **Higher emb_dim.** `emb_dim = 12` (latent dim 1020) or `emb_dim = 24`.
   Trade-off vs latent compactness, but more bits in z = more accurate.
4. **FOM target.** `target = 'fom'`. Trains AE on FVM-discrete u instead
   of analytical. Aligns AE manifold with what NM-ROM is solving for.
   (Earlier attempt found u_FOM is harder to fit than u_analytical
   because of T-junction artifacts. Worth retrying with bigger AE.)
5. **Looser Tikhonov.** `lam_reg = 1e-4` or `1e-5`. Lets NM-ROM correct
   more aggressively. Watch for instability (rel-L2 explosion).
6. **More inner CG iterations.** `max_inner = 12`. Inner CG might not
   be converging in 6 iters on harder problems.
7. **`soft_fine` residual.** Tests whether keeping coarse-cell influence
   at 0.1 helps GN steer better.
8. **Finer mesh.** `min_level = 7` (uniform 128×128 minimum) — eliminates
   T-junctions but quadruples mesh size and AE compute.

## Workflow

1. Read `results.tsv` to see what's been tried and what worked.
2. Read `research_findings.md` (if it exists) for your own running notes
   from prior iterations of this loop — what hypotheses you've tried,
   which directions worked, dead ends, observations to come back to.
3. Decide a single hypothesis to test next based on prior results.
4. Edit `train.py`'s `CONFIG` dict.
5. Run `python train.py` from this directory.
6. Look at the new row in `results.tsv`. If `goal_met=true`, stop. Otherwise,
   integrate the new datapoint into your understanding and iterate.
7. **After every experiment, append a brief entry to `research_findings.md`:**
   what you tried, what number it produced, what surprised you, and what
   you'll try next. This is your durable lab notebook — it's how the next
   agent (or a continuation of you after compaction) gets up to speed
   without re-deriving every conclusion. Be terse: 3-6 lines per entry.
8. Aim to keep individual experiments under ~20 min wall-clock by reusing
   cached AE checkpoints (the runner does this automatically — same
   `(data, AE)` config hash → reuse — only re-bench if only solver knobs
   changed).
9. If 10–15 experiments don't move the needle, propose a structural change
   (bigger architectural shift, different residual weighting scheme, etc.)
   in `notes` rather than just sweeping the same knobs. Document the
   structural shift in `research_findings.md` with reasoning.

## research_findings.md format

A free-form markdown file you append to. Suggested structure for each
entry — keep it tight:

```
### exp_<hash>  tag: <tag>      (commit step / iter N)
- config delta from baseline:  hidden 128→256, ae_steps 8000→16000
- result:  rel_full 0.118 → 0.095   fine 0.130 → 0.103   ratio 1.31 → 1.15
- read:    bigger AE moved both fine and coarse down by ~12%, ratio barely improved
- next:    try emb_dim 6→12 to give the latent more capacity
```

When you reach the goal, write a final summary section at the top of the
file: what the winning configuration was and why it worked.

## How to read `results.tsv`

Tab-separated. Columns:
```
tag  hash  n_train  ae_steps  hidden  emb_dim  target  lam_reg  residual_strategy
rel_full  rel_fine  rel_coarse  ratio_fine_over_coarse  iters_fine  time_fine
goal_met  notes
```

Sort by `rel_full` ascending to see what's working. The goal column
flags the rows that hit both criteria.
