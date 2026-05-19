"""Plot λ sweep results from the Session 7 probe_sweep CSV.

Produces a 2-panel figure: left = left_right task, right = rest_vs_activity.
Each panel shows accuracy and AUC vs λ for each feature source, with random-init
baselines as horizontal reference lines.

Usage:
    python scripts/_plot_lambda_sweep.py runs/probe_sweep.csv [out.png]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def lambda_from_name(name: str) -> float | None:
    m = re.search(r"lambda-([\d.]+)", name)
    return float(m.group(1)) if m else None


def plot_sweep(csv_path: str | Path, out_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    df["lambda"] = df["checkpoint"].apply(lambda_from_name)
    lam = df.dropna(subset=["lambda"]).sort_values("lambda")

    tasks = ["left_right", "rest_vs_activity"]
    sources = ["encoder_mean", "predictor_mean", "both_mean"]
    colors = {"encoder_mean": "C0", "predictor_mean": "C1", "both_mean": "C2"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    for j, task in enumerate(tasks):
        sub = lam[lam["task"] == task]
        # Top row: accuracy
        ax = axes[0, j]
        for src in sources:
            ss = sub[sub["source"] == src].sort_values("lambda")
            ax.plot(ss["lambda"], ss["pre_acc"], "o-",
                    color=colors[src], label=src, linewidth=2, markersize=8)
            # Random baseline as horizontal line
            rand_val = ss["rand_acc"].iloc[0] if len(ss) else None
            if rand_val is not None:
                ax.axhline(rand_val, linestyle=":", color=colors[src], alpha=0.5)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xticks([0, 0.1, 0.3, 1.0, 3.0, 10.0])
        ax.set_xticklabels(["0", "0.1", "0.3", "1.0", "3.0", "10"])
        ax.set_ylabel("LOSO accuracy")
        ax.set_title(f"{task}")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="lower center", fontsize=9)

        # Bottom row: AUC
        ax = axes[1, j]
        for src in sources:
            ss = sub[sub["source"] == src].sort_values("lambda")
            ax.plot(ss["lambda"], ss["pre_auc"], "o-",
                    color=colors[src], label=src, linewidth=2, markersize=8)
            rand_val = ss["rand_auc"].iloc[0] if len(ss) else None
            if rand_val is not None:
                ax.axhline(rand_val, linestyle=":", color=colors[src], alpha=0.5)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xticks([0, 0.1, 0.3, 1.0, 3.0, 10.0])
        ax.set_xticklabels(["0", "0.1", "0.3", "1.0", "3.0", "10"])
        ax.set_xlabel("SIGReg weight λ")
        ax.set_ylabel("LOSO AUC")
        ax.grid(alpha=0.3)

    fig.suptitle("Session 7 λ ablation — 50 subj / 10k steps / batch 64 (RTX 5090)\n"
                 "Solid lines: pretrained.  Dotted: random init baseline.",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/_plot_lambda_sweep.py <csv> [out.png]")
        sys.exit(1)
    csv = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else csv.with_name("lambda_sweep.png")
    plot_sweep(csv, out)
