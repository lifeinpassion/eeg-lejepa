"""Plot training curves from a CSVLogger output.

Reusable helper; called by scripts/04_train.py and can also be run standalone:

    python scripts/_plot_train.py runs/eeg-lejepa-dev/train_log.csv [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_log(csv_path: str | Path, out_path: str | Path) -> None:
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # 1. Losses
    ax = axes[0, 0]
    ax.plot(df["step"], df["total_loss"], label="total", color="black", linewidth=1.5)
    ax.plot(df["step"], df["pred_loss"], label="pred", color="C0", alpha=0.8)
    ax.plot(df["step"], df["sigreg_loss"], label="sigreg", color="C3", alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Training losses")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_yscale("log")

    # 2. Embedding distribution stats
    ax = axes[0, 1]
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.plot(df["step"], df["emb_abs_mean"], label="|mean| (target 0)", color="C2")
    ax.plot(df["step"], df["emb_std"], label="std (target 1)", color="C1")
    ax.plot(df["step"], df["emb_offdiag_abs"], label="off-diag |cov| (target 0)", color="C4")
    ax.set_xlabel("step"); ax.set_ylabel("value")
    ax.set_title("Embedding distribution stats")
    ax.legend(); ax.grid(alpha=0.3)

    # 3. Gradient norm
    ax = axes[1, 0]
    ax.plot(df["step"], df["grad_norm"], color="C5")
    ax.set_xlabel("step"); ax.set_ylabel("L2 grad norm")
    ax.set_title("Gradient norm (clipped at 1.0)")
    ax.grid(alpha=0.3)

    # 4. Learning rate
    ax = axes[1, 1]
    ax.plot(df["step"], df["lr"], color="C7")
    ax.set_xlabel("step"); ax.set_ylabel("lr")
    ax.set_title("Learning rate schedule")
    ax.grid(alpha=0.3)

    fig.suptitle(f"EEGLeJEPA pretraining — {csv_path.name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/_plot_train.py <csv> [out.png]")
        sys.exit(1)
    csv = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else csv.with_suffix(".png")
    plot_training_log(csv, out)
    print(f"Saved plot: {out}")
