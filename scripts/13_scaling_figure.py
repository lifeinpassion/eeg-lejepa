"""Generate the scaling-law figure (the figure the paper marks as TODO).

Probes each scaling-operating-point checkpoint on EEGMMIDB rest-vs-activity
(all three feature sources, best source taken), and plots best-source probe AUC
against an explicitly-defined compute axis:

    x = total window-presentations = n_steps * batch_size

(i.e. how many 4-second EEG windows the optimizer consumed, counting repeats).
This is a clean, reproducible definition; the caption states it verbatim. The
horizontal "saturation" reading then follows from where the curve flattens.

A separate off-curve marker shows the 2.4x-larger-capacity model at matched
compute, illustrating that the plateau is data-bound, not capacity-bound.

LEAKAGE NOTE: every operating point below pretrains on subjects DISJOINT from
the eval set (eval = subjects 1-20; pretraining drawn from 21-109). This is the
corrected, leakage-free scaling experiment. The original paper's "109-subject"
points included subjects 1-20 in pretraining; the clean analog uses the 89
subjects 21-109.

Output: runs/scaling_figure.png, runs/scaling_points.csv

Usage:
    python scripts/13_scaling_figure.py --subjects $(seq 1 20)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
SOURCES = ("encoder_mean", "predictor_mean", "both_mean")


@dataclass
class Point:
    label: str
    ckpt: str        # directory under runs/
    n_subjects: int
    n_steps: int
    batch: int = 64
    is_large: bool = False

    @property
    def exposures(self) -> int:
        return self.n_steps * self.batch


# Clean (disjoint-split) operating points. Edit ckpt names to match your run dirs
# produced by scripts/run_clean_pipeline.sh.
OPERATING_POINTS = [
    Point("3 subj / 200 steps",     "clean-s3-200",      3,     200),
    Point("20 subj / 1k steps",     "clean-s20-1k",      20,   1000),
    Point("20 subj / 5k steps",     "clean-s20-5k",      20,   5000),
    Point("50 subj / 10k steps",    "clean-s51-100-10k", 50,  10000),
    Point("89 subj / 10k steps",    "clean-s89-10k",     89,  10000),
    Point("89 subj / 30k steps",    "clean-s89-30k",     89,  30000),
]
LARGE_POINT = Point("7M params / 89 subj / 30k", "clean-large-89-30k", 89, 30000,
                    is_large=True)


def _probe_best_auc(ckpt_path: Path, ds, cfg, device) -> tuple[float, float, str]:
    state = torch.load(ckpt_path, map_location="cpu")
    embed_dim = next(v.shape[0] for k, v in state.items() if "patch_embed.weight" in k)
    in_ch = next(v.shape[1] for k, v in state.items() if "patch_embed.weight" in k)
    c = EEGLeJEPAConfig.large() if embed_dim == 256 else EEGLeJEPAConfig.base()
    c.encoder.n_channels = in_ch
    c.encoder.patch_size = 40
    model = EEGLeJEPA(c)
    model.load_state_dict(state, strict=False)
    best = (-1.0, -1.0, "")
    for src in SOURCES:
        feats = extract_features_jepa(model, ds.X, source=src, device=device, batch_size=8)
        res = linear_probe_loso_from_features(feats, ds.y, ds.subject_ids)
        if res.mean_auc > best[0]:
            best = (res.mean_auc, res.mean_accuracy, src)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    ap.add_argument("--out", type=Path, default=Path("runs/scaling_figure.png"))
    ap.add_argument("--out-csv", type=Path, default=Path("runs/scaling_points.csv"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])

    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )
    console.print("[bold]Loading rest-vs-activity eval set[/bold] "
                  f"(subjects {args.subjects[0]}-{args.subjects[-1]})")
    ds = build_motor_imagery_dataset(subjects=args.subjects,
                                     data_root=cfg["paths"]["data_root"],
                                     preprocessing=pp, task="rest_vs_activity")
    console.print(f"  {ds.summary()}")

    rows = []
    for pt in OPERATING_POINTS + [LARGE_POINT]:
        ckpt = Path("runs") / pt.ckpt / "model_final.pt"
        if not ckpt.exists():
            console.print(f"[yellow]skip (missing): {ckpt}[/yellow]")
            continue
        auc, acc, src = _probe_best_auc(ckpt, ds, cfg, device)
        rows.append({"label": pt.label, "ckpt": pt.ckpt, "n_subjects": pt.n_subjects,
                     "n_steps": pt.n_steps, "exposures": pt.exposures,
                     "best_auc": auc, "best_acc": acc, "best_source": src,
                     "is_large": pt.is_large})
        console.print(f"  {pt.label:28s} exposures={pt.exposures:>9,d}  "
                      f"AUC={auc:.3f} acc={acc:.3f} ({src})")

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Plot
    base = df[~df["is_large"]].sort_values("exposures")
    large = df[df["is_large"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(base["exposures"], base["best_auc"], "o-", color="C0",
            label="base (2.86M params)")
    for _, r in base.iterrows():
        ax.annotate(r["label"], (r["exposures"], r["best_auc"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7)
    if len(large):
        ax.scatter(large["exposures"], large["best_auc"], marker="D", s=70,
                   color="C3", zorder=5, label="large (7M params)")
        for _, r in large.iterrows():
            ax.annotate(r["label"], (r["exposures"], r["best_auc"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7, color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("Total window-presentations  (n_steps x batch, log scale)")
    ax.set_ylabel("Best-source rest-vs-activity LOSO AUC")
    ax.set_title("Compact-regime scaling: AUC vs compute (leakage-free, eval subj 1-20 held out)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    console.print(f"\n[green]Saved:[/green] {args.out}\n[green]Saved:[/green] {args.out_csv}")


if __name__ == "__main__":
    main()
