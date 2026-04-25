# Quadtree AE + NM-ROM — Project Journey

## What We Are Trying To Do

Solve 2D Poisson (-∇²u = f) on adaptive quadtree meshes using a
**Non-linear Manifold Reduced Order Model (NM-ROM)**.

The key idea: train a quadtree autoencoder whose decoder maps a small latent
vector z (510 floats) to a full PDE solution on an adaptive mesh. Then at
solve time, instead of running a full FEM/FVM solve (~10,000 DOFs), we run
Gauss-Newton iteration in the 510-dim latent space — orders of magnitude
cheaper.

**PDE:**
  -∇²u = sin(2π k₁ x) sin(2π k₂ y)  on [0,1]²,  u=0 on boundary
  Analytical solution: u = f / (4π²(k₁² + k₂²))

**Quadtree mesh:**
  MAX_LEVEL=7, MIN_LEVEL=6 → ~8,500–11,000 leaf nodes typical
  Refined by gradient magnitude of the forcing function f

**Latent space:**
  D_BOTTLE=3, EMB_DIM=6 → z ∈ ℝ⁵¹⁰ (concatenation of E[0..3])
  Bottleneck at coarse depth 3 (85 nodes × 6 = 510 floats)

**Online solve:**
  Given f, find z* = argmin_z ||K · decode(z) - f||²
  via Gauss-Newton + CG inner solve in ℝ⁵¹⁰

---

## The Scripts — What Each One Is

### `super.py`  ← REFERENCE (do not modify)
The original working NM-ROM solver for 3D Poisson on a *regular* grid,
using a ViT encoder + linear CP-tensor decoder. This is what we are
adapting to the quadtree setting. Key insight from this file: the decoder
must be near-linear in z for GN to converge reliably from cold start (z=0).
The skip connection `z + gelu(W z)` makes the Jacobian dominated by the
linear term.

### `champion.py`  ← REFERENCE (do not modify)
The best quadtree autoencoder in PyTorch (D_BOTTLE=3, EMB_DIM=6, hidden=128).
Val loss MSE = 0.000176 — 57× better than the original full AE, 3,490×
compression. Checkpoint saved at `../QuadtreeAE-Exp/champion_model.pt`.
This is the AE architecture we are porting to JAX and combining with NM-ROM.

### `quadtree_nmrom.py`  ← FIRST JAX ATTEMPT (has known bugs)
Full JAX port of the champion AE + GN solver. This was trained once and
produced rel-L2 ~2.1 (broken — see bugs below). Contains the correct
padded-array design for JIT compatibility (fixed shapes = 4^d per depth).
The GN solver design here is correct (eager loop, vjp_fn captured once)
but suffers from AE masking bug + leaf-order mismatch.

### `quadtree_nmrom_jax.py`  ← CURRENT WORKING SCRIPT
Cleaned-up version written in this session. Key improvements over
quadtree_nmrom.py:
  - JIT'd training loop (was untraceable, ~hours/epoch → ~0.5s/step)
  - Vectorized leaf gather (was 4096 scalar .at[].set() → per-depth batched)
  - Correct GN solver (eager, vjp_fn reused, Petrov-Galerkin J^T K J form)
  - Fixed optax schedule crash (warmup clamped to min(200, n_steps//5))
Smoke tested end-to-end on A100. Has NOT been trained for real yet.
May still carry over the AE masking bug and leaf-order mismatch from
quadtree_nmrom.py — these need to be verified before declaring it working.

---

## Training Results So Far

### AE alone (champion.py, PyTorch)
| Config | Latent | Val MSE | Status |
|--------|--------|---------|--------|
| D3 E6 H128 (champion) | 510 | **0.000176** | ✅ best |
| D3 E5 H128 | 425 | 0.000662 | ✅ good |
| D3 E4 H128 | 340 | 0.00139  | ✅ ok   |
| D3 E6 H128 (2k+2k) | 510 | 0.00197  | ok   |
| D3 E8 H128 | 680 | 0.00392  | worse — bigger emb hurts |
| D2 E24 H256 | 504 | 0.084   | ❌ D=2 too shallow |

### AE + NM-ROM (quadtree_nmrom.py, JAX, one completed run)
| Metric | Value |
|--------|-------|
| AE val MSE | 2.5e-5 (looks good on paper) |
| AE reconstruction rel-L2 | **~1.0–5.0** ❌ (catastrophic — bug) |
| NM-ROM avg rel-L2 | **2.10** ❌ (worse than zero prediction) |
| NM-ROM avg solve time | **326s** ❌ (should be <5s) |
| GN convergence | ❌ oscillates, never converges |

---

## Known Bugs

### Bug A — AE masking/reconstruction mismatch  [CRITICAL, must fix first]
**Symptom:** Training MSE = 2.5e-5 but reconstruction rel-L2 = 1.0–5.0.
The training loss and the evaluation metric disagree by orders of magnitude.
**Root cause (suspected):** The training loss sums over padded entries
(zeros) in the denominator while the eval computes relative L2 only over
real leaf nodes — or vice versa. The masking of valid vs. padded nodes is
inconsistent between the two code paths.
**Where to look:** `loss_fn` in quadtree_nmrom.py vs the AE reconstruction
check section. Compare how `leaf_mask` and `valid_mask` are applied in each.
**Fix:** Use a single canonical helper `masked_mse(pred, gt, leaf_mask, valid_mask)`
called from both training and eval. Assert `n_leaves > 0` before dividing.

### Bug B — Leaf order mismatch  [CRITICAL]
**Symptom:** Even if AE reconstruction were fixed, GN solves the wrong system
because the order of leaf nodes in the FVM operator K doesn't match the order
in the decoder output.
**Root cause:** `build_fvm_operator` walks leaves in one order (depth-first,
padded array), `extract_leaf_values` fills them in a different order.
**Fix:** Build one canonical `leaf_index_table: (depth, padded_idx) → global_idx`
and use it in BOTH the FVM assembly and `extract_leaf_values`. Assert that
the two orderings agree on a known case.

### Bug C — GN not converging  [likely downstream of A+B]
**Symptom:** `||r_red||` oscillates between 1e5–1e6, never decreasing monotonically.
**Root cause:** Probably inherits from bugs A+B (wrong AE + wrong ordering =
GN optimising an incoherent objective). But also possibly: the decoder is too
nonlinear in z for GN to converge from cold start z=0. super.py explicitly
notes this and uses a near-linear decoder with skip connection.
**Fix after A+B:** If GN still doesn't converge, (1) normalise the FVM system
by h_min² to improve conditioning, (2) warm-start z via the encoder applied
to a coarse FEM solution, (3) consider adding a linear skip as in super.py.

### Bug D — Slow solve (326s per case)  [FIXED in quadtree_nmrom_jax.py]
**Symptom:** Each GN iteration re-traces VJP/JVP from scratch. 12 iters × ~27s = 326s.
**Root cause:** `jax.vjp` called inside an eager Python loop without capturing
`vjp_fn` — each call recompiles.
**Fix (done):** Capture `vjp_fn` once before the CG loop and reuse it. The
decode forward is JIT-compiled once. Each GN iter is now ~1–5s.

---

## What Needs To Be Fixed (Priority Order)

1. **Fix Bug A** — write a standalone 20-line diagnostic:
   load checkpoint, run forward on one sample, print both the training-loss
   formula result AND the rel-L2 formula result side by side. Find the
   discrepancy. Fix the masking. Target: rel-L2 < 1e-2 on held-out k pairs.

2. **Fix Bug B** — build a single leaf index table and assert ordering
   consistency between FVM assembly and decoder extract. A mismatch will show
   up as rel-L2 not improving even when Bug A is fixed.

3. **Retrain** with fixed masking on the champion config (D3 E6 H128, 4k+4k
   steps). Should take ~35 min on A100. Target AE val MSE < 5e-4.

4. **Benchmark NM-ROM** on 6 test cases. Target: avg rel-L2 < 1e-2, solve
   time < 10s, GN converges in < 12 iters.

5. **If GN still oscillates after A+B+retrain:** add FVM system normalisation
   (scale K and f by h_min²) and/or warm-start from encoder.

---

## Files In This Folder

```
JOURNEY.md              ← this file
super.py                ← reference NM-ROM (3D regular grid, do not modify)
champion.py             ← reference AE (PyTorch, do not modify)
quadtree_nmrom.py       ← first JAX attempt (has bugs A+B+C+D)
quadtree_nmrom_jax.py   ← current working script (Bug D fixed, A+B+C TBD)
```

## Checkpoints (in QuadtreeAE-Solver/)
```
quadtree_ae_phase1.pkl  — best Phase 1 params from the one completed run
quadtree_ae_final.pkl   — best Phase 2 params (use for diagnosis)
```

## Session Notes
- Today (2026-04-25): fixed bugs C/D in quadtree_nmrom_jax.py, smoke tested
  end-to-end. Bugs A+B still need investigation before a real training run.
- Previous session: first full training run on quadtree_nmrom.py, discovered
  the rel-L2 ~2 failure, documented bugs A–D in JOURNEY.md (QuadtreeAE-Solver).
