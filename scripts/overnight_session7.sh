#!/bin/bash
# Session 7 — overnight unattended ablation suite on AutoDL.
#
# Runs:
#   1. λ ∈ {0.0, 0.1, 0.3, 1.0, 3.0, 10.0} at 10k steps each   (~60 min total)
#   2. λ=1.0 at 30k steps                                       (~30 min)
#
# Total wall-clock: ~90 min on RTX 5090 with bf16.
#
# Launch:
#   cd /root/eeg-slm
#   nohup bash scripts/overnight_session7.sh > /root/session7.log 2>&1 &
#   disown
#   tail -f /root/session7.log    # to watch; Ctrl-C just detaches the tail
#
# Then close SSH safely. The job runs to completion regardless.
#
# Morning workflow:
#   1. SSH back in, check the log: tail -100 /root/session7.log
#   2. Grep for "ALL DONE" — if present, all 7 runs completed
#   3. rsync runs/ back to M1: see Session 6 / AUTODL.md §6

set -u  # error on unset vars (but not -e — we want subsequent runs to proceed
        # even if one fails)

cd "$(dirname "$0")/.."   # cd to repo root no matter where launched from

REPO=$(pwd)
COMMON_ARGS=(
    --steps 10000
    --subjects $(seq 1 50)
    --runs 4 8 12
    --batch-size 64
    --num-workers 4
    --bf16
    --device cuda
    --no-plot
)

banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "  $(date)"
    echo "================================================================"
}

run_lambda() {
    local LAM=$1
    banner "λ = $LAM   (10000 steps, batch=64, subjects 1-50)"
    python scripts/04_train.py \
        "${COMMON_ARGS[@]}" \
        --sigreg-weight "$LAM" \
        --out "runs/s7-lambda-${LAM}" \
        || echo ">>> FAILED for λ=$LAM"
}

banner "Session 7 overnight ablation — started"
echo "  Host:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# 1. λ ablation
for LAM in 0.0 0.1 0.3 1.0 3.0 10.0; do
    run_lambda "$LAM"
done

# 2. Longer training at default λ
banner "Longer training: λ=1.0, 30000 steps"
python scripts/04_train.py \
    --steps 30000 \
    --subjects $(seq 1 50) \
    --runs 4 8 12 \
    --batch-size 64 \
    --num-workers 4 \
    --bf16 \
    --device cuda \
    --no-plot \
    --sigreg-weight 1.0 \
    --out runs/s7-50subj-30k \
    || echo ">>> FAILED for 30k step run"

banner "ALL DONE"
echo "Checkpoints in runs/s7-*/"
ls -la runs/s7-*/model_final.pt 2>/dev/null || echo "(no checkpoints found?)"
