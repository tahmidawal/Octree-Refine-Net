#!/bin/bash
# run_inr.slurm
# Smart launcher: reuses an existing GPU node if available, else submits a new SLURM job.
# AE-only mode: only runs INR-Autoencoder.py (no NM-ROM downstream step).
#
# Usage:
#   bash run_inr.slurm [--gpu <jobid>]
#
#   --gpu <jobid>   Use the node from this specific SLURM job ID (falls back to new job if dead)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH_EXTRA="/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages"

# ── Parse flags ─────────────────────────────────────────────────────────────
SPECIFIC_JOBID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     SPECIFIC_JOBID="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Shared remote command ────────────────────────────────────────────────────
remote_cmd() {
    cat << REMOTE
        source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
        module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
        export PYTHONPATH="$PYTHONPATH_EXTRA:\$PYTHONPATH"
        cd "$SCRIPT_DIR"
        echo "=== Training Heat AE ==="
        python3 train_heat_ae.py
REMOTE
}

run_local() {
    source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
    module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
    export PYTHONPATH="$PYTHONPATH_EXTRA:$PYTHONPATH"
    cd "$SCRIPT_DIR"
    echo "=== Training Heat AE ==="
    python3 train_heat_ae.py
}

submit_new_job() {
    echo ">> No GPU node found — submitting new SLURM job..."
    sbatch \
        --job-name=Heat3D-AE \
        --nodes=1 --ntasks=4 \
        --time=08:00:00 \
        --partition=gpu \
        --mem=40G \
        --qos=normal \
        --constraint=a100 \
        --gres=gpu:a100:1 \
        --output="$SCRIPT_DIR/slurm.out" \
        --error="$SCRIPT_DIR/slurm.err" \
        --wrap="
            source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
            module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
            export PYTHONPATH=\"$PYTHONPATH_EXTRA:\$PYTHONPATH\"
            cd \"$SCRIPT_DIR\"
            echo '=== Training Heat AE ==='
            python3 train_heat_ae.py
        "
    echo ">> Watch progress: tail -f $SCRIPT_DIR/training.log"
}

ssh_to_node() {
    local node="$1"
    echo ">> SSHing to GPU node: $node..."
    ssh -o StrictHostKeyChecking=no "$node" bash -s <<< "$(remote_cmd)"
}

# ── Resolve GPU node ─────────────────────────────────────────────────────────
if [[ -n "$SPECIFIC_JOBID" ]]; then
    # User specified a job ID — check if it's still running
    NODE_FROM_JOB=$(squeue -j "$SPECIFIC_JOBID" -o "%.T %N" --noheader 2>/dev/null \
        | awk '$1=="RUNNING" {print $2}')
    if [[ -n "$NODE_FROM_JOB" ]]; then
        echo ">> Using node from job $SPECIFIC_JOBID: $NODE_FROM_JOB"
        ssh_to_node "$NODE_FROM_JOB"
    else
        echo ">> Job $SPECIFIC_JOBID is not running (dead or invalid) — submitting new job..."
        submit_new_job
    fi
elif [[ -n "$SLURM_JOB_ID" ]]; then
    echo ">> Running inside SLURM job $SLURM_JOB_ID on $(hostname)"
    run_local
else
    # Auto-detect any running GPU node
    GPU_NODE=$(squeue -u "$USER" -o "%.T %.R %N" --noheader 2>/dev/null \
        | awk '$1=="RUNNING" && $2!="(None)" {print $3}' \
        | while read node; do
            echo "$node" | grep -qE "gpu|s1cmp|cc[0-9]gpu" && echo "$node" && break
          done)
    if [[ -n "$GPU_NODE" ]]; then
        echo ">> Found running GPU node: $GPU_NODE — SSHing directly..."
        ssh_to_node "$GPU_NODE"
    else
        submit_new_job
    fi
fi
