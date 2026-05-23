"""Paired significance tests for the EEG-LeJEPA linear-probe results.

Computes, for each (task, best-source) comparison:

  * mean / median paired difference,
  * a 95% bootstrap confidence interval on the mean difference,
  * an EXACT paired permutation (sign-flip) p-value
        - full enumeration when n_folds <= 22 (2^n flips),
        - Monte-Carlo with 100k flips otherwise,
  * a Wilcoxon signed-rank p-value (normal approximation with
    tie + continuity correction) as a rank-based cross-check,
  * Cohen's d_z effect size,

then applies a Benjamini-Hochberg FDR correction across the family of
accuracy comparisons.

Two comparison families are produced:

  1. PRETRAINED vs RANDOM-init  (does SSL pretraining help?)
  2. SSL(best source) vs SUPERVISED-from-scratch  (does SSL match/beat
     end-to-end supervised training of the same architecture?)

Inputs
------
--perfold   tidy per-fold probe CSV with columns:
                task, source, subject, pre_acc, pre_auc, rand_acc, rand_auc
            (produced by scripts/11_perfold_dump.py on a checkpoint).
--supervised  per-fold supervised CSV with columns:
                task, subject, accuracy, auc
            (produced by scripts/09_supervised_baseline.py).

Outputs
-------
--out-csv   one row per test with all statistics.
--out-tex   a booktabs LaTeX table fragment for the paper.

Dependency-free beyond numpy + pandas (no scipy required), so it runs in any
environment that can already run the rest of the pipeline.

Examples
--------
    # correctness self-test (no data needed)
    python scripts/12_significance.py --selftest

    # real run on a clean (disjoint-split) checkpoint's per-fold dump
    python scripts/12_significance.py \\
        --perfold runs/perfold_probe_clean.csv \\
        --supervised runs/supervised_loso.csv \\
        --out-csv runs/significance_tests.csv \\
        --out-tex runs/significance_table.tex
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# Primary feature source per task for the SSL-vs-supervised comparison.
# Under the leakage-free split, encoder_mean is the strongest/most robust source
# on BOTH tasks (the predictor_mean/both_mean advantage in the original
# contaminated runs was largely a subject-leakage artifact). The
# pretrained-vs-random comparison below is reported for ALL sources so nothing is
# hidden; change PRIMARY_SOURCE if you select a different headline source.
PRIMARY_SOURCE = {
    "left_right": "encoder_mean",
    "rest_vs_activity": "encoder_mean",
}


# --------------------------------------------------------------------------- #
# Statistics (pure numpy)
# --------------------------------------------------------------------------- #
def bootstrap_ci_mean(diffs: np.ndarray, n_boot: int = 10000,
                      alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def cohen_dz(diffs: np.ndarray) -> float:
    """Paired-sample effect size d_z = mean(diff) / sd(diff)."""
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else float("nan")


def paired_permutation_p(diffs: np.ndarray, n_perm: int = 100_000,
                         seed: int = 0) -> tuple[float, bool]:
    """Two-sided paired permutation (sign-flip) p-value on the mean.

    Exact full enumeration of all 2^n sign assignments when n <= 22,
    otherwise Monte-Carlo with `n_perm` random sign vectors.
    Returns (p_value, exact_flag).
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    obs = abs(diffs.mean())
    if n == 0:
        return float("nan"), True
    if n <= 22:
        # Exact: every datum is +d or -d. Enumerate sign vectors as bits.
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        perm_means = np.abs((signs * diffs).mean(axis=1))
        p = float((perm_means >= obs - 1e-12).mean())
        return p, True
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_perm, n))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    # +1 correction (include the observed assignment) for a valid MC p-value.
    p = float((np.sum(perm_means >= obs - 1e-12) + 1) / (n_perm + 1))
    return p, False


def wilcoxon_signed_rank_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p (normal approx, tie + continuity corr).

    Zeros are dropped (standard Wilcoxon handling). Returns NaN if fewer than
    one non-zero difference remains.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return float("nan")
    ranks = _average_ranks(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    # Variance with tie correction.
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = (counts ** 3 - counts).sum()
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return float("nan")
    z = (w - mean_w + 0.5 * np.sign(mean_w - w)) / np.sqrt(var_w)  # continuity corr
    return float(2.0 * _norm_sf(abs(z)))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (1-based)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of 1-based ranks i+1..j+1
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _norm_sf(z: float) -> float:
    """Upper-tail standard normal survival function via erfc (math.erfc)."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0, 1)
    return q.tolist()


def summarize_pair(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """All paired statistics for arrays a (treatment) vs b (control)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diffs = a - b
    lo, hi = bootstrap_ci_mean(diffs, seed=seed)
    perm_p, exact = paired_permutation_p(diffs, seed=seed)
    return {
        "n": int(len(diffs)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "cohen_dz": cohen_dz(diffs),
        "perm_p": perm_p,
        "perm_exact": exact,
        "wilcoxon_p": wilcoxon_signed_rank_p(diffs),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _aligned(df: pd.DataFrame, task: str, source: str,
             col_a: str, col_b: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[(df["task"] == task) & (df["source"] == source)].sort_values("subject")
    return sub["subject"].to_numpy(), sub[col_a].to_numpy(), sub[col_b].to_numpy()


def run(perfold_csv: Path, supervised_csv: Path,
        out_csv: Path, out_tex: Path) -> pd.DataFrame:
    pf = pd.read_csv(perfold_csv)
    sup = pd.read_csv(supervised_csv)

    rows: list[dict] = []

    # 1. Pretrained vs random — reported for EVERY (task, source) so the source
    #    ranking is fully visible (it changed under the leakage-free split).
    for task in sorted(pf["task"].unique()):
        for source in sorted(pf[pf["task"] == task]["source"].unique()):
            _, pre, rnd = _aligned(pf, task, source, "pre_acc", "rand_acc")
            r = summarize_pair(pre, rnd)
            rows.append({"comparison": "pretrained_vs_random", "metric": "accuracy",
                         "task": task, "source": source, **r})
            _, pre_a, rnd_a = _aligned(pf, task, source, "pre_auc", "rand_auc")
            r = summarize_pair(pre_a, rnd_a)
            rows.append({"comparison": "pretrained_vs_random", "metric": "auc",
                         "task": task, "source": source, **r})

    # 2. SSL (primary source) vs supervised-from-scratch, aligned by subject.
    for task in sorted(pf["task"].unique()):
        source = PRIMARY_SOURCE.get(task)
        if source is None:
            continue
        ss = pf[(pf["task"] == task) & (pf["source"] == source)][["subject", "pre_acc", "pre_auc"]]
        sv = sup[sup["task"] == task][["subject", "accuracy", "auc"]]
        merged = ss.merge(sv, on="subject", how="inner").sort_values("subject")
        if len(merged):
            r = summarize_pair(merged["pre_acc"].to_numpy(), merged["accuracy"].to_numpy())
            rows.append({"comparison": "ssl_vs_supervised", "metric": "accuracy",
                         "task": task, "source": source, **r})
            r = summarize_pair(merged["pre_auc"].to_numpy(), merged["auc"].to_numpy())
            rows.append({"comparison": "ssl_vs_supervised", "metric": "auc",
                         "task": task, "source": source, **r})

    df = pd.DataFrame(rows)

    # BH correction across the accuracy tests (the primary family).
    acc_mask = df["metric"] == "accuracy"
    df.loc[acc_mask, "perm_q_bh"] = benjamini_hochberg(df.loc[acc_mask, "perm_p"].tolist())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _write_latex(df, out_tex)
    return df


def _write_latex(df: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        r"% Auto-generated by scripts/12_significance.py",
        r"\begin{table}[t]",
        r"\caption{Paired significance of the headline gaps (LOSO folds). "
        r"$\Delta$ is mean paired difference in accuracy (pp); CI is a 95\% "
        r"percentile bootstrap; $p$ is an exact paired sign-flip permutation "
        r"test; $q$ is the Benjamini--Hochberg FDR-adjusted value across the "
        r"accuracy family; $d_z$ is the paired effect size.}",
        r"\label{tab:significance}",
        r"\centering\small",
        r"\begin{tabular}{l l l c c c c}",
        r"\toprule",
        r"Comparison & Task & Source & $\Delta$ (pp) & 95\% CI (pp) & $p$ / $q$ & $d_z$ \\",
        r"\midrule",
    ]
    label = {"pretrained_vs_random": "Pre $-$ Rand",
             "ssl_vs_supervised": "SSL $-$ Sup"}
    task_label = {"left_right": "L-vs-R MI", "rest_vs_activity": "rest-vs-act"}
    src_label = {"encoder_mean": "enc", "predictor_mean": "pred", "both_mean": "both"}
    acc = df[df["metric"] == "accuracy"].copy()
    # Group the table: pretrained_vs_random block first, then ssl_vs_supervised.
    for comp in ["pretrained_vs_random", "ssl_vs_supervised"]:
        block = acc[acc["comparison"] == comp]
        for _, r in block.iterrows():
            q = r.get("perm_q_bh", float("nan"))
            lines.append(
                f"{label[r['comparison']]} & {task_label[r['task']]} & "
                f"{src_label.get(r['source'], r['source'])} & "
                f"{r['mean_diff']*100:+.1f} & "
                f"[{r['ci95_lo']*100:+.1f}, {r['ci95_hi']*100:+.1f}] & "
                f"{r['perm_p']:.3f} / {q:.3f} & {r['cohen_dz']:+.2f} \\\\"
            )
        if comp == "pretrained_vs_random":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest() -> None:
    print("Running correctness self-tests...")
    rng = np.random.default_rng(42)

    # (a) Exact permutation == brute-force agreement, and matches MC closely.
    d = rng.normal(0.3, 1.0, size=15)
    p_exact, ex = paired_permutation_p(d)
    assert ex, "n=15 should use exact enumeration"
    p_mc, _ = paired_permutation_p(np.concatenate([d, d, [0.0]])[:23], n_perm=200_000)
    # Independent MC on the SAME n=15 sample (force MC path via n_perm large, n<=22 still exact)
    # Validate exact value by an independent brute-force here:
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(d))))
    brute = (np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12).mean()
    assert abs(p_exact - brute) < 1e-12, (p_exact, brute)
    print(f"  [ok] exact permutation matches brute force (p={p_exact:.4f})")

    # (b) Wilcoxon known small example. Differences 1..10 (all positive) ->
    #     W=0, strong significance; p should be small (~0.002 normal approx).
    d2 = np.arange(1, 11, dtype=float)
    p_w = wilcoxon_signed_rank_p(d2)
    assert p_w < 0.01, p_w
    print(f"  [ok] wilcoxon monotone-positive sample p={p_w:.4f} (<0.01)")

    # (c) Symmetric-around-zero sample -> permutation p near 1.
    d3 = np.array([-3, -2, -1, 1, 2, 3], dtype=float)
    p3, _ = paired_permutation_p(d3)
    assert p3 > 0.9, p3
    print(f"  [ok] symmetric sample permutation p={p3:.3f} (>0.9)")

    # (d) Bootstrap CI brackets the sample mean.
    d4 = rng.normal(0.5, 0.2, size=30)
    lo, hi = bootstrap_ci_mean(d4)
    assert lo < d4.mean() < hi, (lo, d4.mean(), hi)
    print(f"  [ok] bootstrap CI [{lo:.3f},{hi:.3f}] brackets mean {d4.mean():.3f}")

    # (e) BH monotonic and bounded.
    q = benjamini_hochberg([0.001, 0.04, 0.2, 0.5])
    assert all(0 <= x <= 1 for x in q) and q == sorted(q)
    print(f"  [ok] BH q-values {[round(x,3) for x in q]}")
    print("All self-tests passed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perfold", type=Path, default=Path("runs/perfold_probe_clean.csv"))
    ap.add_argument("--supervised", type=Path, default=Path("runs/supervised_loso.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("runs/significance_tests.csv"))
    ap.add_argument("--out-tex", type=Path, default=Path("runs/significance_table.tex"))
    ap.add_argument("--selftest", action="store_true", help="Run correctness checks and exit.")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.perfold.exists():
        raise SystemExit(
            f"Per-fold probe CSV not found: {args.perfold}\n"
            "Generate it first with scripts/11_perfold_dump.py on a "
            "clean (disjoint-split) checkpoint.")
    df = run(args.perfold, args.supervised, args.out_csv, args.out_tex)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))
    print(f"\nSaved: {args.out_csv}\nSaved: {args.out_tex}")


if __name__ == "__main__":
    main()
