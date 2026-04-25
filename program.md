# autoresearch — Quadtree Autoencoder Compression

This is an autonomous research loop for finding the optimal compressed quadtree autoencoder configuration. The goal: **minimize val_loss while keeping the latent representation between 100–500 floats**.

## Context

We have a working quadtree autoencoder for PDE forcing functions (f(x,y) = sin(2*pi*k1*x)*sin(2*pi*k2*y)). The full model uses skip connections at every depth and achieves val_loss ~0.01. We've built a compressed variant that introduces a bottleneck at depth D_BOTTLE — the decoder only receives encoder embeddings at depths 0..D_BOTTLE and must reconstruct fine levels (up to MAX_LEVEL=7) using ancestor-indexed conditioning.

### What we know so far

| Config | Latent Floats | Best Val Loss | Notes |
|---|---|---|---|
| D=4, emb=128 | 43,648 | 0.021 | Great quality, too many floats |
| D=3, emb=128 | 10,880 | 0.054 | Good quality |
| D=3, emb=32 | 2,720 | 0.027 | Surprisingly good — emb_dim reduction works |
| D=2, emb=48 | 1,008 | ~0.25 (stuck) | D_BOTTLE=2 doesn't have enough spatial resolution |
| D=2, emb=10 | 210 | ~0.25 (stuck) | Not enough info |
| D=1, emb=40 | 200 | ~0.27 (stuck) | Not enough info |

**Key finding**: D_BOTTLE=3 (85 nodes) is the minimum depth that provides enough spatial resolution for ancestor-indexed conditioning. Below that, the model can't learn. The emb_dim can be reduced significantly (128→32 with minimal loss).

### Target regime

- **Latent floats**: 100–500 (currently the gap between 210 (fails) and 2720 (works) is unexplored)
- **Val loss target**: as low as possible, ideally below 0.10
- **The latent must be standalone**: decoder + latent embeddings must reconstruct without the encoder or GT tree

## Setup

1. **The script**: `compressed_bottleneck_ae_ultra.py` — this is the file you modify.
2. **Read these files for context**:
   - `compressed_bottleneck_ae_ultra.py` — the training script (model, data gen, training loop, plotting)
   - `compressed_bottleneck_ae.py` — the base version (reference, do not modify)
   - `20260217-QuadVAE-Autoencoder.py` — the proven full AE (reference architecture)
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
4. **Initialize results.tsv**: Create `results.tsv` with the header row.
5. **Confirm and go**.

## Submitting GPU Jobs (SLURM)

All experiments run on the Tufts HPC cluster via SLURM. Here's how to submit and monitor:

### Submit a job

```bash
sbatch \
    --job-name=<experiment-name> \
    --nodes=1 --ntasks=4 \
    --time=08:00:00 \
    --partition=gpu \
    --mem=40G \
    --qos=normal \
    --constraint=a100 \
    --gres=gpu:a100:1 \
    --output="slurm_<experiment>.out" \
    --error="slurm_<experiment>.err" \
    --wrap="
        source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
        module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
        export PYTHONPATH=\"/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:\$PYTHONPATH\"
        export PYTHONUNBUFFERED=1
        cd /cluster/tufts/paralab/tawal01/Octree-Refine-Net
        python3 -u compressed_bottleneck_ae_ultra.py --d_bottle <D> --emb_dim <E> --hidden <H>
    "
```

### Monitor a job

```bash
# Check job status
squeue -u $USER -o "%.10i %.12j %.2t %.10M %R" --noheader

# Watch live output (unbuffered thanks to PYTHONUNBUFFERED=1 and python3 -u)
tail -f slurm_<experiment>.out

# Check for crashes
cat slurm_<experiment>.err | tail -20

# Extract best val loss from output
grep "best=" slurm_<experiment>.out | tail -1

# Cancel a job
scancel <jobid>
```

### Key SLURM notes
- Always use `export PYTHONUNBUFFERED=1` and `python3 -u` — without these, stdout is buffered and you won't see logs for minutes.
- Jobs take ~5-15 min to start depending on cluster load.
- You can run 2-3 GPU jobs simultaneously (one per available A100 node).
- Output goes to `slurm_<name>.out`, errors to `slurm_<name>.err`.

## What you CAN modify

Everything in `compressed_bottleneck_ae_ultra.py` is fair game:

- **D_BOTTLE** (bottleneck depth) — currently 1-4, sets number of latent nodes
- **emb_dim** (embedding dimension per node) — determines info per node
- **hidden** (internal network width) — decoder capacity, doesn't affect latent size
- **Architecture changes**:
  - Ancestor-indexed conditioning strategy
  - QuadConv kernel design
  - Child generation (child_head)
  - Skip connection design
  - Positional encoding (Fourier frequencies)
  - Adding attention mechanisms
  - Learned codebook / VQ-VAE style quantization
  - Multi-scale skip strategies
- **Training changes**:
  - Learning rate, schedule, optimizer
  - Phase 1/Phase 2 step counts
  - Loss weights (split_weight)
  - Gradient clipping
  - Data augmentation
  - Curriculum learning (start with simple k1,k2, increase complexity)

## What you CANNOT modify

- `compressed_bottleneck_ae.py` (the base reference)
- `20260217-QuadVAE-Autoencoder.py` (the proven full AE reference)
- The data generation logic (sin functions, quadtree refinement, Morton codes)
- The evaluation metric (MSE on leaf values)

## The goal

**Get the lowest val_loss with a latent between 100–500 floats.**

The latent size formula: `latent_floats = sum(4^d for d in range(D_BOTTLE+1)) * emb_dim`

| D_BOTTLE | Nodes | emb_dim for 100 floats | emb_dim for 500 floats |
|---|---|---|---|
| 1 | 5 | 20 | 100 |
| 2 | 21 | 5 | 24 |
| 3 | 85 | 2 | 6 |

Given that D_BOTTLE=2 has failed to learn so far, the most promising directions are:
1. **Make D_BOTTLE=2 work** — this would unlock 100-500 float latents. Ideas: stronger decoder, attention-based conditioning instead of ancestor-indexed, curriculum training, larger hidden dim.
2. **Push D_BOTTLE=3 lower** — with emb_dim=2-6 (170-510 floats). Very small embeddings but more spatial nodes.
3. **Hybrid approaches** — e.g., D_BOTTLE=2 with a learned upsampling to create "virtual" depth-3 embeddings.

## Output format

The training script prints per-step logs like:
```
[P1] Step  100 | val=0.140479 split=0.096662 total=0.188892 | k1=1 k2=3 leaves=8704 | compress=85/11605 | best=0.149540@98 patience=2/300
```

Extract key metrics:
```bash
grep "best=" slurm_<name>.out | tail -1    # best total loss
grep "val=" slurm_<name>.out | tail -1      # latest val loss
```

Plots are saved every 100 steps to `plots/<output_dir>/`:
- `samples_stepNNNN.png` — GT vs Predicted vs Error Map vs MSE-by-depth
- `loss_curve_stepNNNN.png` — Value/Split/Total loss curves
- `infer_stepNNNN.png` — GT vs Teacher-forced vs Autoregressive standalone

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	d_bottle	emb_dim	latent_floats	best_val_loss	best_total_loss	status	description
```

Example:
```
commit	d_bottle	emb_dim	latent_floats	best_val_loss	best_total_loss	status	description
a1b2c3d	3	32	2720	0.015	0.027	keep	baseline d3e32
b2c3d4e	3	6	510	0.045	0.068	keep	reduced emb_dim to 6
c3d4e5f	2	24	504	0.250	0.300	discard	d_bottle=2 still stuck
d4e5f6g	2	24	504	0.120	0.180	keep	d_bottle=2 with attention conditioning
```

## The experiment loop

LOOP FOREVER:

1. Look at `results.tsv` and the current best configuration
2. Pick an experiment: modify `compressed_bottleneck_ae_ultra.py` with a new idea
3. `git commit -am "experiment: <description>"`
4. Submit the SLURM job (see submission template above)
5. Monitor: check `slurm_<name>.out` periodically for progress
6. Wait for the job to reach at least step 500 (Phase 1) — check `grep "best=" slurm_<name>.out | tail -1`
7. If val_loss improved → status=keep, advance the branch
8. If val_loss is worse → status=discard, `git reset --hard HEAD~1`
9. Record results in `results.tsv`
10. Go to step 1

### Experiment strategy

Start with the most promising low-hanging fruit:
1. First, establish baseline: D=3 emb=6 (510 floats) — does tiny emb_dim work at D=3?
2. Then try D=3 emb=4 (340 floats) and D=3 emb=2 (170 floats)
3. If those fail, try architectural improvements to make small emb_dim work:
   - Increase hidden dim (256 or 512) to give the decoder more capacity
   - Add attention in the decoder
   - Use a learned upsampling module at the bottleneck boundary
   - Curriculum training: start with simple k1,k2=1,2 then increase
4. If D=3 hits a wall, try making D=2 work with architectural changes
5. Try hybrid: D=2 nodes but with a learned "virtual expansion" to depth 3

### Parallelism

You can submit 2-3 experiments simultaneously on different GPU nodes. Use different output files:
```bash
# Experiment A
--output="slurm_expA.out" --error="slurm_expA.err"

# Experiment B (different config, same script)
--output="slurm_expB.out" --error="slurm_expB.err"
```

### Patience

- Each experiment needs at least 500 steps to judge. Don't kill early.
- If a run is clearly stuck after 300 steps (val_loss not moving), you can cancel early.
- Phase 2 (autoregressive fine-tuning) is bonus — judge primarily on Phase 1 results.

### NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep or away. You are autonomous. If you run out of ideas, think harder — re-read the architecture, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
