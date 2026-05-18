"""Linear-probe evaluation: multiple feature sources × pretrained vs random.

For each combination of (init, source) we run leave-one-subject-out
cross-validation on EEGMMIDB motor-imagery left vs right (runs 4, 8, 12).

Feature sources:
  encoder_mean    : mean-pool encoder embeddings (no temporal context — Session 4 v1)
  predictor_mean  : mean-pool predictor hidden states (with causal temporal context)
  both_mean       : concat encoder_mean and predictor_mean

The predictor-based sources should be substantially better for sequence-level
tasks like motor imagery, because the encoder by design has zero cross-time
context (per-patch independent) while the predictor is a causal Transformer
over the patch sequence.

Run:

    python scripts/05_linear_probe.py \\
        --ckpt runs/lambda-1.0/model_final.pt \\
        --subjects 1 2 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.table import Table

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

DEFAULT_SOURCES = ("encoder_mean", "predictor_mean", "both_mean")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to the pretrained EEGLeJEPA state_dict (.pt)")
    p.add_argument("--subjects", type=int, nargs="+", required=True,
                   help="Subject IDs for the probe (≥2 for LOSO)")
    p.add_argument("--runs", type=int, nargs="*",
                   default=list(RUNS_MOTOR_IMAGERY_LEFT_RIGHT),
                   help="Run codes (default: 4 8 12 — left/right fist imagery)")
    p.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES),
                   choices=["encoder_mean", "predictor_mean", "both_mean",
                            "predictor_last", "predictor_concat"])
    p.add_argument("--C", type=float, default=1.0,
                   help="LogisticRegression inverse regularization strength")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])
    console.print(f"[bold]Device:[/bold] {device}")

    # 1. Build the labeled motor-imagery dataset
    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )
    console.print(f"[bold]Loading motor-imagery dataset[/bold] "
                  f"(subjects={args.subjects}, runs={args.runs})")
    ds = build_motor_imagery_dataset(
        subjects=args.subjects,
        data_root=cfg["paths"]["data_root"],
        preprocessing=pp,
        runs=tuple(args.runs),
    )
    console.print(f"  {ds.summary()}")

    # 2. Two models: pretrained and random-init
    model_cfg = EEGLeJEPAConfig()
    model_cfg.encoder.n_channels = ds.X.shape[1]
    model_cfg.encoder.patch_size = 40

    console.print(f"\n[bold]Loading pretrained checkpoint:[/bold] [cyan]{args.ckpt}[/cyan]")
    pre_model = EEGLeJEPA(model_cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = pre_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        console.print(f"  [yellow]missing[/yellow]: {missing[:3]}{'...' if len(missing)>3 else ''}")

    torch.manual_seed(cfg["training"]["seed"] + 1)
    rand_model = EEGLeJEPA(model_cfg)  # fresh init

    # 3. Run probe for each (init, source) pair
    results: dict[tuple[str, str], object] = {}
    for init_name, model in [("pretrained", pre_model), ("random", rand_model)]:
        for source in args.sources:
            console.print(f"  probing [bold]{init_name}[/bold] / {source} ...")
            feats = extract_features_jepa(
                model=model, X=ds.X, source=source, device=device, batch_size=8,
            )
            res = linear_probe_loso_from_features(feats, ds.y, ds.subject_ids, C=args.C)
            results[(init_name, source)] = res

    # 4. Comparison table
    table = Table(title=f"Linear-probe LOSO accuracy on {len(args.subjects)} subjects",
                  show_header=True, header_style="bold")
    table.add_column("Feature source")
    table.add_column("Pretrained acc", justify="right")
    table.add_column("Random acc", justify="right")
    table.add_column("Δ (pp)", justify="right")
    table.add_column("Pretrained AUC", justify="right")

    chance = next(iter(results.values())).chance

    for source in args.sources:
        pre = results[("pretrained", source)]
        rnd = results[("random", source)]
        delta = (pre.mean_accuracy - rnd.mean_accuracy) * 100
        delta_str = f"[green]+{delta:.1f}[/green]" if delta > 2 else (
            f"[red]{delta:.1f}[/red]" if delta < -2 else f"[yellow]{delta:+.1f}[/yellow]"
        )
        table.add_row(
            source,
            f"{pre.mean_accuracy:.3f} ±{pre.std_accuracy:.3f}",
            f"{rnd.mean_accuracy:.3f} ±{rnd.std_accuracy:.3f}",
            delta_str,
            f"{pre.mean_auc:.3f}",
        )
    table.add_row("[dim]chance[/dim]", f"[dim]{chance:.3f}[/dim]", f"[dim]{chance:.3f}[/dim]", "", "0.500")
    console.print()
    console.print(table)

    # 5. Per-fold detail for the BEST pretrained source
    best_src = max(args.sources, key=lambda s: results[("pretrained", s)].mean_accuracy)
    best = results[("pretrained", best_src)]
    console.print(f"\n[bold]Per-fold detail — pretrained / {best_src}:[/bold]")
    console.print("  " + best.summary())

    # 6. Verdict
    best_delta = max(
        (results[("pretrained", s)].mean_accuracy - results[("random", s)].mean_accuracy) * 100
        for s in args.sources
    )
    if best_delta > 3.0:
        console.print(f"\n[bold green]Pretraining helped[/bold green] "
                      f"by +{best_delta:.1f} pp on the best feature source.")
    elif best_delta > 0:
        console.print(f"\n[bold yellow]Marginal gain[/bold yellow] (+{best_delta:.1f} pp); "
                      f"need more subjects to be confident.")
    else:
        console.print(f"\n[bold red]No benefit yet[/bold red] (best Δ = {best_delta:+.1f} pp).")


if __name__ == "__main__":
    main()
