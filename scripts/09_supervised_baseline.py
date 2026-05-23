"""Supervised-from-scratch baseline on EEGMMIDB motor imagery.

For each LOSO fold on subjects 1-20, train a fresh EEGClassifier (same encoder
architecture as our SSL model + linear classification head) end-to-end with
cross-entropy. Reports per-fold accuracy + AUC.

This isolates the SSL contribution: comparison vs the linear-probe-on-frozen-
SIGReg-encoder pipeline. If supervised matches or beats SSL+probe, SSL adds
little; if SSL+probe matches supervised at much less downstream training cost,
that's a clean argument for the pretraining recipe.

Runs both tasks: left_right and rest_vs_activity.

Usage:
    python scripts/09_supervised_baseline.py
    python scripts/09_supervised_baseline.py --steps 1000 --batch-size 32
    python scripts/09_supervised_baseline.py --task rest_vs_activity --subjects $(seq 1 20)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from eeg_slm.data import PreprocessingConfig
from eeg_slm.eval import build_motor_imagery_dataset
from eeg_slm.eval.motor_imagery import RUNS_MOTOR_IMAGERY_LEFT_RIGHT
from eeg_slm.models import EEGClassifier, EEGClassifierConfig
from eeg_slm.training import SupervisedTrainConfig, supervised_loso
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    p.add_argument("--tasks", nargs="+", default=["left_right", "rest_vs_activity"],
                   choices=["left_right", "rest_vs_activity"])
    p.add_argument("--steps", type=int, default=500,
                   help="Supervised training steps per LOSO fold")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("runs/supervised_loso.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])
    console.print(f"[bold]Device:[/bold] {device}")

    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )

    train_cfg = SupervisedTrainConfig(
        n_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for task in args.tasks:
        console.print(f"\n[bold]Task: {task}[/bold]")
        ds = build_motor_imagery_dataset(
            subjects=args.subjects,
            data_root=cfg["paths"]["data_root"],
            preprocessing=pp,
            runs=tuple(RUNS_MOTOR_IMAGERY_LEFT_RIGHT),
            task=task,
        )
        console.print(f"  {ds.summary()}")
        n_classes = int(ds.y.max()) + 1
        console.print(f"  {n_classes} classes; supervised LOSO with "
                      f"{args.steps} steps × batch={args.batch_size} per fold")

        # Build model_factory closure (fresh model per fold)
        n_channels = ds.X.shape[1]
        def _make_model():
            mcfg = EEGClassifierConfig(
                encoder=EEGClassifierConfig().encoder,  # base encoder config
                n_classes=n_classes,
            )
            mcfg.encoder.n_channels = n_channels
            mcfg.encoder.patch_size = 40
            return EEGClassifier(mcfg)

        # Param count (sanity check, before LOSO)
        sample_model = _make_model()
        n_params = sample_model.num_parameters
        console.print(f"  EEGClassifier: {n_params/1e6:.2f}M params per fold")
        del sample_model

        result = supervised_loso(
            model_factory=_make_model,
            X=ds.X, y=ds.y, subject_ids=ds.subject_ids,
            cfg=train_cfg, device=device, seed=cfg["training"]["seed"],
        )

        console.print(f"  → {result.summary()}")

        # Collect per-fold rows
        for i, subj in enumerate(result.fold_subjects):
            all_rows.append({
                "task": task,
                "subject": subj,
                "accuracy": result.fold_accuracies[i],
                "auc": result.fold_aucs[i],
            })
        # Summary row per task
        summary_rows.append({
            "task": task,
            "mean_acc": result.mean_accuracy,
            "std_acc": result.std_accuracy,
            "mean_auc": result.mean_auc,
            "chance": result.chance,
            "n_folds": len(result.fold_accuracies),
            "above_70": sum(1 for a in result.fold_accuracies if a >= 0.70),
            "above_60": sum(1 for a in result.fold_accuracies if a >= 0.60),
            "below_chance": sum(1 for a in result.fold_accuracies if a < result.chance),
        })

    # Save per-fold CSV
    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    console.print(f"\n[green]Saved per-fold CSV:[/green] {args.out}")

    # Summary table
    tbl = Table(title=f"Supervised-from-scratch LOSO baseline "
                      f"(n_steps={args.steps}, batch={args.batch_size})",
                show_header=True, header_style="bold")
    tbl.add_column("Task")
    tbl.add_column("Acc ± std", justify="right")
    tbl.add_column("AUC", justify="right")
    tbl.add_column("Above 70%", justify="right")
    tbl.add_column("Below chance", justify="right")
    for r in summary_rows:
        tbl.add_row(
            r["task"],
            f"{r['mean_acc']:.3f} ± {r['std_acc']:.3f}",
            f"{r['mean_auc']:.3f}",
            f"{r['above_70']}/{r['n_folds']}",
            f"{r['below_chance']}/{r['n_folds']}",
        )
    console.print()
    console.print(tbl)

    # Comparison hint vs the published SSL+probe numbers
    console.print()
    console.print("[bold]For comparison — SSL+probe (s7-lambda-1.0) headline numbers:[/bold]")
    console.print("  left_right       (both_mean):     0.657 ± 0.046  (AUC 0.711)")
    console.print("  rest_vs_activity (predictor_mean): 0.711 ± 0.089  (AUC 0.778)")


if __name__ == "__main__":
    main()
