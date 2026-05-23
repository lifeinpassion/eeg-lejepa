#!/bin/bash
# Extra GPU experiments for the revision (run on the PRO 6000 / 5090).
#
# Produces three things; ALL probing is done afterward on the laptop:
#   (A) §4.5 transfer: a 22-channel matched re-pretrain (subjects 51-100) so the
#       clean headline encoder's channel-padded transfer can be compared to a
#       channel-matched encoder on BCI-IV-2a.
#   (B) Capacity isolation: the 7M large model trained leakage-free at M=256
#       (vs the existing M=1024 run) to test whether the original "large model
#       regresses" result was a SIGReg-projection artifact, not capacity.
#   (C) Multi-seed capacity: base + large at seeds 43 and 44 (seed 42 already
#       exists as clean-s89-30k / clean-large-89-30k) for a 3-seed comparison.
#
# Each run is skipped if its checkpoint already exists (idempotent / resumable).
#
#   bash scripts/run_extra_experiments.sh
#
set -u
cd "$(dirname "$0")/.."

COMMON=(--runs 4 8 12 --batch-size 64 --num-workers 4 --bf16 --device cuda --no-plot)

pretrain() {
    local out="$1"; shift
    if [ -f "$out/model_final.pt" ]; then echo "    skip (exists): $out"; return 0; fi
    echo "    training -> $out"
    python scripts/04_train.py "${COMMON[@]}" "$@" --out "$out"
}

echo "==> (A) Transfer: 22-channel matched re-pretrain (subjects 51-100, M=1024)"
pretrain runs/clean-s51-100-22ch-10k \
    --steps 10000 --subjects $(seq 51 100) --channel-subset bci-iv-2a \
    --sigreg-weight 1.0 --sigreg-slices 1024

echo "==> (B) Capacity isolation: large model leakage-free at M=256 (89 subj, 30k)"
pretrain runs/clean-large-89-30k-m256 \
    --steps 30000 --subjects $(seq 21 109) --model-size large \
    --lr 5e-4 --warmup-steps 300 --sigreg-weight 1.0 --sigreg-slices 256

echo "==> (C) Multi-seed capacity: base + large at seeds 43, 44 (M=1024, 89 subj, 30k)"
for SEED in 43 44; do
    pretrain "runs/clean-s89-30k-s${SEED}" \
        --steps 30000 --subjects $(seq 21 109) --sigreg-weight 1.0 \
        --sigreg-slices 1024 --seed "$SEED"
    pretrain "runs/clean-large-89-30k-s${SEED}" \
        --steps 30000 --subjects $(seq 21 109) --model-size large \
        --lr 5e-4 --warmup-steps 300 --sigreg-weight 1.0 \
        --sigreg-slices 1024 --seed "$SEED"
done

echo ""
echo "==> DONE. Pull these checkpoints back, then probe on the laptop:"
echo "      clean-s51-100-22ch-10k     (transfer matched)"
echo "      clean-large-89-30k-m256    (capacity isolation)"
echo "      clean-s89-30k-s43/44, clean-large-89-30k-s43/44  (capacity seeds)"
