# F → u NM-ROM — quickstart

Self-contained code for the F→u NM-ROM pipeline on adaptive quadtree meshes.
See **PROGRESS.md** for the architecture, design decisions, and full results.

## Requirements

- Python 3.10+
- JAX with GPU support (`pip install -U "jax[cuda12]"`)
- `flax`, `optax`, `numpy`, `matplotlib`

## Stage-by-stage

The main pipeline `f_to_u_nmrom.py` has five subcommands:

```bash
# 1. Generate paired (F, u) quadtree dataset (cached to runs/exp/paired_dataset.pkl)
python f_to_u_nmrom.py data --out runs/exp --n-train 200 --n-val 40

# 2. Train the shared autoencoder on F + u union  (~10 min, 8000 steps)
python f_to_u_nmrom.py train_ae --out runs/exp --ae-steps 8000

# 3. Encode train/val pairs and train the latent-space MLP  (~15 s)
python f_to_u_nmrom.py train_mlp --out runs/exp

# 4. Run NM-ROM benchmark: COLD / WARM-full / WARM-fine / MLP-only / FOM  (~5 min)
python f_to_u_nmrom.py bench --out runs/exp --n-test 6

# Or run all four in sequence
python f_to_u_nmrom.py all --out runs/exp --n-train 200 --n-val 40
```

## Plot scripts

Each takes `--exp-dir <path>` pointing at the run directory:

```bash
# Bar-chart summary from benchmark_results.tsv (per-case rel-L2, fine vs coarse, time, iters)
python plot_finecell_summary.py --exp-dir runs/exp

# Detailed 8-panel-per-case plot: input F, ground-truth u, MLP prediction, NM-ROM
# prediction, mesh structure, error maps, per-leaf error histogram by depth
python plot_per_case_detailed.py --exp-dir runs/exp --k 2,2 3,2 3,3 5,3

# 6-row × 4-column grid: analytical / WARM-full / WARM-fine / MLP for each case
python plot_solutions.py --exp-dir runs/exp --n-cases 6
```

## Output layout per experiment

```
runs/exp/
├── paired_dataset.pkl              # (n_train + n_val) (F, u) pairs on adaptive quadtrees
├── shared_ae.pkl                   # AE checkpoint (params + training history)
├── mlp_z_f_to_z_u.pkl              # MLP checkpoint
├── benchmark_results.tsv           # per-case + AVG rel-L2 / time / iters
├── ae_loss.png                     # AE training curves
├── plots_bench/case_NN.png         # per-case 6-panel from `bench` command
├── plots_detailed/                 # from plot_per_case_detailed.py
│   └── detailed_kK1_K2.png
├── plots_solutions/                # from plot_solutions.py
│   └── all_cases_grid.png
└── finecell_summary.png            # from plot_finecell_summary.py
```

## Knobs

```
f_to_u_nmrom.py:
  --n-train, --n-val            dataset size (default 200, 40)
  --k-lo, --k-hi                integer k bounds (default 1, 5)
  --grad-thresh                 |∇F| threshold for MAX_LEVEL refinement (default 0.5)
  --target  {analytical, fom}   u target: closed-form OR per-sample CG solve
  --ae-steps                    AE training steps (default 8000)
  --hidden, --emb-dim           AE capacity overrides
  --mlp-steps, --mlp-hidden, --mlp-layers
  --max-outer, --max-inner      GN iteration caps (default 8 × 6 = 48)
  --gn-tol                      GN convergence tolerance
  --n-test                      number of cases in bench
```

The integer-k default is important: with continuous k the analytical formula
violates u=0 on the boundary and FOM diverges from the formula by ~40%.

## Top-line numbers (full pipeline, integer-k adaptive mesh, 6 test cases)

| Method                    | full-mesh rel-L2 | fine-cells | iters | time |
|---------------------------|------------------|------------|-------|------|
| COLD start NM-ROM         | 0.414            | 0.480      | 48    | 27 s |
| WARM (full residual)      | 0.382            | 0.416      | 31    | 18 s |
| **WARM (fine-cells-only)**| **0.113**        | **0.123**  | **4.8** | **0.47 s** |
| MLP only                  | 0.101            | 0.114      | –     | –    |
| FOM (CG on FVM)           | 0.221            | 0.225      | –     | 0.04 s |

The fine-cells-only NM-ROM converges 6–8× faster than warm-full and is 2×
more accurate than FOM on the metric that matters (fine cells, where the
mesh has resolution).
