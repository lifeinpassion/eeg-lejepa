"""CONTAMINATED PREVIEW (no GPU needed).

Demonstrates the significance + figure machinery using ONLY real numbers already
in runs/ (probe_sweep.csv, supervised_loso.csv) plus the logged scaling AUCs.

NOTHING here is the final result. The headline checkpoint used here
(s7-lambda-1.0) was pretrained on subjects 1-50, which OVERLAPS the eval set
1-20 (subject leakage). Re-run scripts/run_clean_pipeline.sh on a GPU for the
real, leakage-free numbers.

Outputs:
  runs/significance_PREVIEW.csv         (headline gaps, summary-stat CIs)
  runs/significance_supervised_real.csv (REAL paired tests on supervised folds)
  runs/scaling_figure_PREVIEW.png       (scaling curve from logged AUCs)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# import the validated stat functions from 12_significance.py
spec = importlib.util.spec_from_file_location("sig", ROOT / "scripts" / "12_significance.py")
sig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sig)

T_975_DF19 = 2.093  # Student-t 0.975 quantile, df = n-1 = 19  (n = 20 LOSO folds)
BEST = {"left_right": "both_mean", "rest_vs_activity": "predictor_mean"}


def headline_summary_ci() -> pd.DataFrame:
    """95% CI on the pretrained accuracy from the REAL reported mean+std (n=20).

    This is a summary-statistics approximation (per-fold vectors aren't stored in
    probe_sweep.csv, so the exact paired permutation test can't run yet). It uses
    only the real logged mean/std and answers: does the pretrained accuracy's CI
    exclude the random-init mean and chance (0.5)?
    """
    ps = pd.read_csv(RUNS / "probe_sweep.csv")
    ps = ps[ps.checkpoint == "s7-lambda-1.0"]
    rows = []
    n = 20
    half = T_975_DF19 / np.sqrt(n)
    for task, src in BEST.items():
        r = ps[(ps.task == task) & (ps.source == src)].iloc[0]
        ci_lo = r.pre_acc - half * r.pre_acc_std
        ci_hi = r.pre_acc + half * r.pre_acc_std
        rows.append({
            "task": task, "source": src,
            "pre_acc": round(r.pre_acc, 4), "pre_std": round(r.pre_acc_std, 4),
            "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
            "rand_acc": round(r.rand_acc, 4),
            "gap_vs_random_pp": round((r.pre_acc - r.rand_acc) * 100, 1),
            "CI_excludes_random": bool(ci_lo > r.rand_acc),
            "CI_excludes_chance": bool(ci_lo > 0.5),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "significance_PREVIEW.csv", index=False)
    return df


def supervised_real_tests() -> pd.DataFrame:
    """REAL paired significance tests on the only real per-fold data we have."""
    sup = pd.read_csv(RUNS / "supervised_loso.csv")
    rows = []
    # (a) supervised accuracy vs chance, per task (one-sample sign-flip on acc-0.5)
    for task in ["left_right", "rest_vs_activity"]:
        acc = sup[sup.task == task].sort_values("subject")["accuracy"].to_numpy()
        d = acc - 0.5
        lo, hi = sig.bootstrap_ci_mean(d, seed=0)
        p, exact = sig.paired_permutation_p(d, seed=0)
        rows.append({"test": f"supervised {task} vs chance", "n": len(acc),
                     "mean_diff_pp": round(d.mean() * 100, 1),
                     "ci95_pp": f"[{lo*100:+.1f},{hi*100:+.1f}]",
                     "perm_p": round(p, 4), "exact": exact,
                     "wilcoxon_p": round(sig.wilcoxon_signed_rank_p(d), 4),
                     "cohen_dz": round(sig.cohen_dz(d), 2)})
    # (b) paired: supervised rest vs left_right (same subjects)
    lr = sup[sup.task == "left_right"].sort_values("subject")["accuracy"].to_numpy()
    rv = sup[sup.task == "rest_vs_activity"].sort_values("subject")["accuracy"].to_numpy()
    s = sig.summarize_pair(rv, lr)
    rows.append({"test": "supervised rest minus left_right", "n": s["n"],
                 "mean_diff_pp": round(s["mean_diff"] * 100, 1),
                 "ci95_pp": f"[{s['ci95_lo']*100:+.1f},{s['ci95_hi']*100:+.1f}]",
                 "perm_p": round(s["perm_p"], 4), "exact": s["perm_exact"],
                 "wilcoxon_p": round(s["wilcoxon_p"], 4),
                 "cohen_dz": round(s["cohen_dz"], 2)})
    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "significance_supervised_real.csv", index=False)
    return df


def scaling_preview() -> None:
    """Scaling curve from the REAL logged best-source rest-vs-activity AUCs.

    Points combine PAPER_OUTLINE's scaling table (small-scale runs, M1 batch=8)
    with probe_sweep.csv (50/109-subject AutoDL runs, batch=64). x = steps*batch.
    """
    pts = [
        # label, n_subj, steps, batch, best_auc(rest_vs_activity), large?
        ("3 subj / 200",     3,    200,  8,  0.692, False),
        ("20 subj / 1k",     20,  1000,  8,  0.694, False),
        ("20 subj / 5k",     20,  5000,  8,  0.740, False),
        ("50 subj / 10k",    50, 10000, 64,  0.778, False),
        ("109 subj / 10k",  109, 10000, 64,  0.756, False),
        ("109 subj / 30k",  109, 30000, 64,  0.766, False),
        ("7M / 109 / 30k",  109, 30000, 64,  0.748, True),
    ]
    df = pd.DataFrame(pts, columns=["label", "n_subj", "steps", "batch", "auc", "large"])
    df["exposures"] = df.steps * df.batch
    base = df[~df.large].sort_values("exposures")
    large = df[df.large]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(base.exposures, base.auc, "o-", color="C0", label="base (2.86M)")
    for _, r in base.iterrows():
        ax.annotate(r.label, (r.exposures, r.auc), textcoords="offset points",
                    xytext=(6, -11), fontsize=7)
    ax.scatter(large.exposures, large.auc, marker="D", s=70, color="C3",
               zorder=5, label="large (7M)")
    for _, r in large.iterrows():
        ax.annotate(r.label, (r.exposures, r.auc), textcoords="offset points",
                    xytext=(6, 7), fontsize=7, color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("Total window-presentations  (steps x batch, log scale)")
    ax.set_ylabel("Best-source rest-vs-activity LOSO AUC")
    ax.set_title("PREVIEW (contaminated split) — scaling from logged AUCs")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.text(0.02, 0.96, "PREVIEW — eval subjects were in pretraining (leakage).\n"
                        "Regenerate with scripts/13_scaling_figure.py after GPU rerun.",
            transform=ax.transAxes, fontsize=7, va="top", color="C3",
            bbox=dict(boxstyle="round", fc="white", ec="C3", alpha=0.8))
    fig.tight_layout()
    fig.savefig(RUNS / "scaling_figure_PREVIEW.png", dpi=140)
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("CONTAMINATED PREVIEW — illustrative only; eval subjects leaked into")
    print("pretraining (s7-lambda-1.0 trained on subjects 1-50, eval on 1-20).")
    print("=" * 70)

    print("\n[1] Headline gaps — 95% CI from REAL reported mean+/-std (n=20, t df=19)")
    print("    (summary-stat approximation; exact paired test needs the GPU rerun)\n")
    print(headline_summary_ci().to_string(index=False))

    print("\n[2] REAL paired significance tests on the supervised per-fold data")
    print("    (runs/supervised_loso.csv is genuine per-fold data, leakage-free)\n")
    print(supervised_real_tests().to_string(index=False))

    scaling_preview()
    print("\n[3] Saved scaling preview figure: runs/scaling_figure_PREVIEW.png")
    print("\nSaved: runs/significance_PREVIEW.csv, runs/significance_supervised_real.csv")


if __name__ == "__main__":
    main()
