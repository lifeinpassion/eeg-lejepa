"""Dump per-fold (per-subject) LOSO probe results to a tidy CSV.

Unlike scripts/06_probe_sweep.py (which saves only the mean +/- std per
checkpoint), this writes ONE ROW PER (task, source, held-out subject) for both
the pretrained checkpoint and the random-initialized baseline. That per-fold
detail is what the paired significance tests in scripts/12_significance.py
need.

Output columns:
    task, source, subject, pre_acc, pre_auc, rand_acc, rand_auc

IMPORTANT (leakage): the checkpoint passed via --ckpt must have been pretrained
on subjects DISJOINT from --subjects. The paper evaluates on subjects 1-20, so
the checkpoint must be pretrained on subjects 51-100 (see
scripts/run_clean_pipeline.sh). This script prints a reminder but cannot verify
the pretraining split from the checkpoint alone.

Usage:
    python scripts/11_perfold_dump.py \\
        --ckpt runs/clean-s51-100-10k/model_final.pt \\
        --subjects $(seq 1 20) \\
        --out runs/perfold_probe_clean.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from rich.console import Console

from eeg_slm.data import PreprocessingConfig
from eeg_slm.eval import (
    build_motor_imagery_dataset,
    extract_features_jepa,
    linear_probe_loso_from_features,
)
from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()

TASKS = ("left_right", "rest_vs_activity")
SOURCES = ("encoder_mean", "predictor_mean", "both_mean")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    p.add_argument("--out", type=Path, default=Path("runs/perfold_probe_clean.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])
    console.print(f"[bold]Device:[/bold] {device}  [bold]Checkpoint:[/bold] {args.ckpt}")
    console.print("[yellow]Reminder:[/yellow] checkpoint must be pretrained on subjects "
                  "DISJOINT from the eval set to avoid leakage.")

    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )

    state = torch.load(args.ckpt, map_location="cpu")
    for k, v in state.items():
        if "patch_embed.weight" in k:
            in_ch, embed_dim = v.shape[1], v.shape[0]
            break
    is_large = embed_dim == 256

    def _make_cfg(n_channels: int) -> EEGLeJEPAConfig:
        c = EEGLeJEPAConfig.large() if is_large else EEGLeJEPAConfig.base()
        c.encoder.n_channels = n_channels
        c.encoder.patch_size = 40
        return c

    torch.manual_seed(cfg["training"]["seed"] + 1)  # match 06/08 random baseline
    rand_model = EEGLeJEPA(_make_cfg(in_ch))
    pre_model = EEGLeJEPA(_make_cfg(in_ch))
    pre_model.load_state_dict(state, strict=False)

    rows: list[dict] = []
    for task in TASKS:
        console.print(f"\n[bold]Loading[/bold] task={task}")
        ds = build_motor_imagery_dataset(
            subjects=args.subjects, data_root=cfg["paths"]["data_root"],
            preprocessing=pp, task=task,
        )
        console.print(f"  {ds.summary()}")
        for source in SOURCES:
            feats_pre = extract_features_jepa(pre_model, ds.X, source=source,
                                              device=device, batch_size=8)
            feats_rand = extract_features_jepa(rand_model, ds.X, source=source,
                                               device=device, batch_size=8)
            res_pre = linear_probe_loso_from_features(feats_pre, ds.y, ds.subject_ids)
            res_rand = linear_probe_loso_from_features(feats_rand, ds.y, ds.subject_ids)
            # Index random folds by subject so alignment is exact.
            rand_acc = dict(zip(res_rand.fold_subjects, res_rand.fold_accuracies))
            rand_auc = dict(zip(res_rand.fold_subjects, res_rand.fold_aucs))
            for s, a, au in zip(res_pre.fold_subjects, res_pre.fold_accuracies,
                                res_pre.fold_aucs):
                rows.append({
                    "task": task, "source": source, "subject": int(s),
                    "pre_acc": a, "pre_auc": au,
                    "rand_acc": rand_acc.get(s, float("nan")),
                    "rand_auc": rand_auc.get(s, float("nan")),
                })
            console.print(f"  {source:14s}: pre {res_pre.mean_accuracy:.3f} / "
                          f"rand {res_rand.mean_accuracy:.3f}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    console.print(f"\n[green]Saved per-fold CSV:[/green] {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
