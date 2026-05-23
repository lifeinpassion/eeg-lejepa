"""Generate the per-fold variance figure for the paper's headline result.

Runs the LOSO probe on the best checkpoint (s7-lambda-1.0) with full per-fold
detail captured, plots all 20 subjects' accuracies as a bar chart, overlays the
random-init baseline + chance line + mean line. Preempts the reviewer question
"is this result driven by a few easy subjects?"

Two panels: left = left_right MI, right = rest_vs_activity. Both use the
best feature source per task (both_mean for left_right, predictor_mean for
rest_vs_activity, per Sessions 6-7).

Output: runs/per_fold_figure.png

Usage:
    python scripts/08_per_fold_figure.py \\
        --ckpt runs/s7-lambda-1.0/model_final.pt \\
        --subjects $(seq 1 20)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich.console import Console

from eeg_slm.data import PreprocessingConfig
from eeg_slm.eval import (
    build_motor_imagery_dataset,
    extract_features_jepa,
    linear_probe_loso_from_features,
)
from eeg_slm.eval.motor_imagery import RUNS_MOTOR_IMAGERY_LEFT_RIGHT
from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()

# (task, source): under the leakage-free split, encoder_mean is the strongest
# AND only consistently significant source on both tasks; the predictor-derived
# sources lost their (leakage-inflated) advantage. See scripts/12_significance.py.
PANEL_CONFIGS = (
    ("left_right",       "encoder_mean"),
    ("rest_vs_activity", "encoder_mean"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("runs/s7-lambda-1.0/model_final.pt"))
    p.add_argument("--subjects", type=int, nargs="+",
                   default=list(range(1, 21)))
    p.add_argument("--out", type=Path, default=Path("runs/per_fold_figure.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])
    console.print(f"[bold]Device:[/bold] {device}")
    console.print(f"[bold]Checkpoint:[/bold] {args.ckpt.name}")

    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )

    # Build model configs (need shapes that match the checkpoint)
    state = torch.load(args.ckpt, map_location="cpu")
    for k, v in state.items():
        if "patch_embed.weight" in k:
            in_ch = v.shape[1]
            embed_dim = v.shape[0]
            break
    is_large = embed_dim == 256
    console.print(f"  detected: {in_ch}-channel, embed_dim={embed_dim}, "
                  f"{'large' if is_large else 'base'}")

    def _make_cfg(n_channels: int) -> EEGLeJEPAConfig:
        c = EEGLeJEPAConfig.large() if is_large else EEGLeJEPAConfig.base()
        c.encoder.n_channels = n_channels
        c.encoder.patch_size = 40
        return c

    # Build models once (reused across tasks)
    torch.manual_seed(cfg["training"]["seed"] + 1)
    rand_model = EEGLeJEPA(_make_cfg(in_ch))
    pre_model = EEGLeJEPA(_make_cfg(in_ch))
    pre_model.load_state_dict(state, strict=False)

    # Collect per-fold results for each (task, source) panel
    panel_data: list[dict] = []
    for task, source in PANEL_CONFIGS:
        console.print(f"\n[bold]Loading dataset[/bold] task={task}")
        ds = build_motor_imagery_dataset(
            subjects=args.subjects,
            data_root=cfg["paths"]["data_root"],
            preprocessing=pp,
            runs=tuple(RUNS_MOTOR_IMAGERY_LEFT_RIGHT),
            task=task,
        )
        console.print(f"  {ds.summary()}")

        console.print(f"[bold]Probing[/bold] source={source}")
        feats_pre = extract_features_jepa(
            pre_model, ds.X, source=source, device=device, batch_size=8,
        )
        feats_rand = extract_features_jepa(
            rand_model, ds.X, source=source, device=device, batch_size=8,
        )
        res_pre = linear_probe_loso_from_features(feats_pre, ds.y, ds.subject_ids)
        res_rand = linear_probe_loso_from_features(feats_rand, ds.y, ds.subject_ids)

        panel_data.append({
            "task": task,
            "source": source,
            "fold_subjects": res_pre.fold_subjects,
            "pre_accs": res_pre.fold_accuracies,
            "rand_accs": res_rand.fold_accuracies,
            "chance": res_pre.chance,
            "pre_mean": res_pre.mean_accuracy,
            "rand_mean": res_rand.mean_accuracy,
            "pre_std": res_pre.std_accuracy,
            "pre_auc": res_pre.mean_auc,
        })

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, panel in zip(axes, panel_data):
        # Sort folds by pretrained accuracy (ascending) for readability
        order = np.argsort(panel["pre_accs"])
        subjects_sorted = [panel["fold_subjects"][i] for i in order]
        pre_sorted = [panel["pre_accs"][i] for i in order]
        rand_sorted = [panel["rand_accs"][i] for i in order]

        x = np.arange(len(subjects_sorted))
        bar_w = 0.4
        ax.bar(x - bar_w / 2, pre_sorted, bar_w,
               color="C0", alpha=0.85, label=f"Pretrained ({panel['source']})")
        ax.bar(x + bar_w / 2, rand_sorted, bar_w,
               color="C7", alpha=0.6, label="Random init")

        # Horizontal reference lines
        ax.axhline(panel["chance"], linestyle=":", color="gray", linewidth=1,
                   label=f"chance = {panel['chance']:.2f}")
        ax.axhline(panel["pre_mean"], linestyle="--", color="C0", linewidth=1.2,
                   alpha=0.7, label=f"pretrained mean = {panel['pre_mean']:.3f}")

        ax.set_xticks(x)
        ax.set_xticklabels([f"s{s:02d}" for s in subjects_sorted],
                          rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("LOSO accuracy")
        ax.set_title(
            f"{panel['task']}\n"
            f"acc = {panel['pre_mean']:.3f} ± {panel['pre_std']:.3f}  "
            f"(AUC = {panel['pre_auc']:.3f}; Δ vs random = "
            f"{(panel['pre_mean'] - panel['rand_mean']) * 100:+.1f} pp)"
        )
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Per-subject LOSO accuracy on 20 EEGMMIDB subjects — "
        f"{args.ckpt.parent.name}",
        fontweight="bold",
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    console.print(f"\n[green]Saved figure:[/green] {args.out}")

    # Also print a compact textual summary
    console.print()
    for panel in panel_data:
        n_above_mean = sum(1 for a in panel["pre_accs"] if a > panel["pre_mean"])
        n_below_chance = sum(1 for a in panel["pre_accs"] if a < panel["chance"])
        n_above_60 = sum(1 for a in panel["pre_accs"] if a >= 0.60)
        n_above_70 = sum(1 for a in panel["pre_accs"] if a >= 0.70)
        console.print(
            f"[bold]{panel['task']}/{panel['source']}:[/bold] "
            f"mean={panel['pre_mean']:.3f} ± {panel['pre_std']:.3f}, "
            f"{n_above_70}/20 above 70%, {n_above_60}/20 above 60%, "
            f"{n_below_chance}/20 below chance"
        )


if __name__ == "__main__":
    main()
