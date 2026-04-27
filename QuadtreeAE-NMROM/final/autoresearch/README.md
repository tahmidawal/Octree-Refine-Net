# autoresearch — F → u NM-ROM iteration loop

Karpathy-style minimal autoresearch setup. Two files matter:

- **`program.md`** — read-only problem description + agent instructions.
  Tells the agent the success criteria, the available knobs, and which
  hypotheses to try in what order.
- **`train.py`** — the **only** file the agent edits. Has a `CONFIG` dict
  at the top. Run `python train.py` from this directory and it will:
  1. Generate the dataset (cached by config hash)
  2. Train the AE (cached)
  3. Train the latent MLP (cached)
  4. Run the NM-ROM benchmark with the configured solver
  5. Append a single row to `results.tsv` with metrics + `goal_met` flag

Caching is by content hash of the relevant sub-config slice, so changing
only solver knobs re-uses the AE checkpoint and only re-runs the bench
(~3 min instead of ~15 min).

## How to spawn an agent on this

From `final/autoresearch/`:

```
agent: read program.md, then iterate by editing train.py and running it
       until results.tsv shows a row with goal_met=True. report progress
       every iteration.
```

## Goal (also in program.md)

`goal_met == True` requires both:
1. avg full-mesh rel-L2 vs analytical < 1e-2
2. fine-cells rel-L2 ≤ 0.5 × coarse-cells rel-L2

Current baseline:
- rel_full = 0.118
- rel_fine = 0.130
- rel_coarse = 0.099
- ratio fine/coarse = 1.31  (need ≤ 0.5 — fine is currently *worse* than coarse)
- iters = 6, time = 0.49 s/case

So we need both ~12× improvement in absolute rel-L2 and a 2.6× flip in
the fine/coarse ratio.

## Layout

```
autoresearch/
├── README.md
├── program.md         agent instructions
├── train.py           ← agent edits CONFIG, runs `python train.py`
├── results.tsv        append-only experiment log (tab-separated)
└── runs/              cached artifacts keyed by sub-config hash
    ├── data_<hash>/
    ├── ae_<hash>/
    ├── mlp_<hash>/
    └── exp_<hash>/    plots + per-experiment outputs
```

## Manual sanity check

```bash
cd final/autoresearch/
python train.py        # ~15 min (or ~3 min if AE/data are cached)
cat results.tsv | column -t -s $'\t'
```

The first run uses the baseline production config — should reproduce
rel_full ≈ 0.12, fine 0.13, coarse 0.10 from `final/runs/full/`.
