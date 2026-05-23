#!/bin/bash
# Leakage-free end-to-end pipeline for EEG-LeJEPA.
#
# WHY THIS EXISTS
# ---------------
# The original headline checkpoint (runs/s7-lambda-1.0) was pretrained with
# `--subjects $(seq 1 50)` (see scripts/overnight_session7.sh) but every probe
# evaluates on subjects 1-20. The 20 eval subjects were therefore INSIDE the SSL
# pretraining set, which inflates the cross-subject linear-probe numbers and
# contradicts the paper's claim that subjects 1-20 are "strictly held out".
#
# This script reproduces every EEGMMIDB result with a strictly DISJOINT split:
#     pretraining subjects  in  {21..109}     (never 1-20)
#     evaluation subjects   =   {1..20}        (never pretrained on)
#
# USAGE
#   bash scripts/run_clean_pipeline.sh             # full: train + probe + figs
#   bash scripts/run_clean_pipeline.sh --train-only# GPU box: only the 04_train runs
#   bash scripts/run_clean_pipeline.sh --minimal   # headline + probe + significance
#
# Each pretraining run is SKIPPED if its model_final.pt already exists, so the
# script is safe to re-run / resume. NEITHER path needs BCI-IV-2a; the
# cross-dataset transfer experiment is handled separately by
# scripts/07_probe_bci_iv_2a.py.
#
# Recommended split of labor: run `--train-only` on the GPU box, pull the ~12 MB
# checkpoints back, then run the probe/significance/figure steps on the laptop
# (which has a current src/ and supervised_loso.csv).
set -u
cd "$(dirname "$0")/.."

MODE="full"
case "${1:-}" in
  --minimal)    MODE="minimal" ;;
  --train-only) MODE="train-only" ;;
esac
echo "==> mode: $MODE"

EVAL="$(seq 1 20)"          # held-out evaluation subjects
# --sigreg-slices 1024 matches the paper's stated M=1024 (original runs used the
# 256 default, so paper text and checkpoints disagreed; this makes them consistent).
TRAIN_ARGS=(--runs 4 8 12 --batch-size 64 --num-workers 4 --bf16 --device cuda \
            --no-plot --sigreg-slices 1024)

# Skip a pretraining run if its checkpoint already exists (idempotent / resumable).
pretrain() {
    local out="$1"; shift
    if [ -f "$out/model_final.pt" ]; then
        echo "    skip (already trained): $out"
        return 0
    fi
    echo "    training -> $out"
    python scripts/04_train.py "${TRAIN_ARGS[@]}" "$@" --out "$out"
}

echo "==> 1. Headline checkpoint: subjects 51-100 (disjoint from 1-20)"
pretrain runs/clean-s51-100-10k --steps 10000 --subjects $(seq 51 100) --sigreg-weight 1.0

if [ "$MODE" = "full" ] || [ "$MODE" = "train-only" ]; then
    echo "==> 2. Scaling operating points (all pretraining subjects in 21-109)"
    pretrain runs/clean-s3-200  --steps 200   --subjects $(seq 51 53)  --sigreg-weight 1.0
    pretrain runs/clean-s20-1k  --steps 1000  --subjects $(seq 51 70)  --sigreg-weight 1.0
    pretrain runs/clean-s20-5k  --steps 5000  --subjects $(seq 51 70)  --sigreg-weight 1.0
    pretrain runs/clean-s89-10k --steps 10000 --subjects $(seq 21 109) --sigreg-weight 1.0
    pretrain runs/clean-s89-30k --steps 30000 --subjects $(seq 21 109) --sigreg-weight 1.0

    echo "==> 3. Capacity test: 7M-param model, same disjoint corpus"
    pretrain runs/clean-large-89-30k --steps 30000 --subjects $(seq 21 109) \
        --model-size large --lr 5e-4 --warmup-steps 300 --sigreg-weight 1.0

    echo "==> 4. lambda ablation on the clean headline corpus (subjects 51-100)"
    for LAM in 0.0 0.1 0.3 1.0 3.0 10.0; do
        pretrain "runs/clean-lambda-${LAM}" --steps 10000 --subjects $(seq 51 100) \
            --sigreg-weight "$LAM"
    done
fi

if [ "$MODE" = "train-only" ]; then
    echo ""
    echo "==> TRAINING DONE. Pull the checkpoints back, then on the laptop run:"
    echo "      scripts/11_perfold_dump.py / 09_supervised_baseline.py /"
    echo "      12_significance.py / 08_per_fold_figure.py / 13_scaling_figure.py"
    exit 0
fi

echo "==> 5. Per-fold probe dump on the clean headline checkpoint (eval subj 1-20)"
python scripts/11_perfold_dump.py \
    --ckpt runs/clean-s51-100-10k/model_final.pt \
    --subjects $EVAL --out runs/perfold_probe_clean.csv

echo "==> 6. Supervised-from-scratch baseline (already leakage-free; per-fold labels)"
python scripts/09_supervised_baseline.py --task left_right       --subjects $EVAL
python scripts/09_supervised_baseline.py --task rest_vs_activity --subjects $EVAL

echo "==> 7. Significance tests"
python scripts/12_significance.py \
    --perfold runs/perfold_probe_clean.csv \
    --supervised runs/supervised_loso.csv \
    --out-csv runs/significance_tests.csv \
    --out-tex runs/significance_table.tex

echo "==> 8. Figures"
python scripts/08_per_fold_figure.py --ckpt runs/clean-s51-100-10k/model_final.pt --subjects $EVAL
if [ "$MODE" = "full" ]; then
    python scripts/13_scaling_figure.py --subjects $EVAL
fi

echo ""
echo "==> DONE ($MODE)."
