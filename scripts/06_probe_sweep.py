"""Probe multiple checkpoints in one go — sweep summary + CSV export.

Loops the standard linear probe over every checkpoint matching --ckpts (glob
expansion supported), on each --tasks task, with each --sources feature source,
all against the SAME randomly-initialized baseline (so Δ values are comparable
across checkpoints).

Saves a CSV with one row per (checkpoint, task, source), and prints a pretty
per-task summary table to the console.

Example:

    python scripts/06_probe_sweep.py \\
        --ckpts 'runs/s7-lambda-*/model_final.pt' \\
        --subjects $(seq 1 20)

Outputs: runs/probe_sweep.csv plus per-task tables on stdout.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
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
from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Checkpoint paths or globs (e.g. 'runs/s7-lambda-*/model_final.pt')")
    p.add_argument("--subjects", type=int, nargs="+", required=True,
                   help="Subject IDs for the LOSO probe (≥2)")
    p.add_argument("--tasks", nargs="+", default=["left_right", "rest_vs_activity"],
                   choices=["left_right", "rest_vs_activity"])
    p.add_argument("--sources", nargs="+",
                   default=["encoder_mean", "predictor_mean", "both_mean"],
                   choices=["encoder_mean", "predictor_mean", "both_mean",
                            "predictor_last", "predictor_concat"])
    p.add_argument("--out", type=Path, default=Path("runs/probe_sweep.csv"))
    return p.parse_args()


def _expand_ckpts(patterns: list[str]) -> list[Path]:
    """Expand globs (shell may have already done so; either way, dedupe + sort)."""
    out: list[str] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        out.extend(matches if matches else [pat])
    # Dedupe, drop missing files
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        if Path(p).exists():
            unique.append(Path(p))
        else:
            console.print(f"[yellow]Skipping missing: {p}[/yellow]")
    return unique


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])

    ckpts = _expand_ckpts(args.ckpts)
    if not ckpts:
        console.print(f"[red]No checkpoints found matching {args.ckpts}[/red]")
        return
    console.print(f"[bold]Found {len(ckpts)} checkpoints[/bold]  (device: {device})")
    for c in ckpts:
        console.print(f"  {c.parent.name}")

    # Preprocessing config (shared)
    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )

    # Load datasets once per task (the expensive part — MNE preprocessing)
    datasets: dict[str, object] = {}
    for task in args.tasks:
        console.print(f"\n[bold]Loading dataset[/bold] task={task}")
        ds = build_motor_imagery_dataset(
            subjects=args.subjects,
            data_root=cfg["paths"]["data_root"],
            preprocessing=pp,
            task=task,
        )
        datasets[task] = ds
        console.print(f"  {ds.summary()}")

    # Build the per-size model configs we'll need.
    # Checkpoint architecture is inferred from the state_dict keys/shapes
    # because we don't store config alongside the checkpoint (TODO: add that).
    sample_ds = datasets[args.tasks[0]]

    def _make_cfg(size: str) -> EEGLeJEPAConfig:
        cfg = EEGLeJEPAConfig.large() if size == "large" else EEGLeJEPAConfig.base()
        cfg.encoder.n_channels = sample_ds.X.shape[1]
        cfg.encoder.patch_size = 40
        return cfg

    def _infer_size_from_state_dict(state: dict) -> str:
        """Sniff the encoder.patch_embed weight shape to detect base (192) vs large (256)."""
        for k, v in state.items():
            if "patch_embed.weight" in k:
                return "large" if v.shape[0] == 256 else "base"
        return "base"

    model_cfg = _make_cfg("base")  # for the random-init baseline

    # Random-init baseline (one per task × source; reuse across all checkpoints)
    console.print("\n[bold]Random-init baseline[/bold]")
    torch.manual_seed(cfg["training"]["seed"] + 1)  # match 05_linear_probe.py
    rand_model = EEGLeJEPA(model_cfg)
    rand_cache: dict[tuple[str, str], object] = {}
    for task in args.tasks:
        for src in args.sources:
            feats = extract_features_jepa(rand_model, datasets[task].X,
                                          source=src, device=device, batch_size=8)
            res = linear_probe_loso_from_features(feats, datasets[task].y,
                                                  datasets[task].subject_ids)
            rand_cache[(task, src)] = res
            console.print(f"  random / {task:18s} / {src:14s}: "
                          f"acc={res.mean_accuracy:.3f}  auc={res.mean_auc:.3f}")

    # Probe every checkpoint
    rows: list[dict] = []
    for ckpt in ckpts:
        ckpt_name = ckpt.parent.name
        console.print(f"\n[bold]Probing[/bold] {ckpt_name}")
        state = torch.load(ckpt, map_location="cpu")
        size = _infer_size_from_state_dict(state)
        ckpt_cfg = _make_cfg(size)
        model = EEGLeJEPA(ckpt_cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            console.print(f"  [yellow]missing {len(missing)} / unexpected {len(unexpected)}[/yellow]")
        elif size == "large":
            console.print(f"  [dim](detected large architecture)[/dim]")
        for task in args.tasks:
            for src in args.sources:
                feats = extract_features_jepa(model, datasets[task].X,
                                              source=src, device=device, batch_size=8)
                res = linear_probe_loso_from_features(feats, datasets[task].y,
                                                      datasets[task].subject_ids)
                rand = rand_cache[(task, src)]
                delta_pp = (res.mean_accuracy - rand.mean_accuracy) * 100
                delta_auc = res.mean_auc - rand.mean_auc
                rows.append({
                    "checkpoint": ckpt_name,
                    "task": task,
                    "source": src,
                    "pre_acc": res.mean_accuracy,
                    "pre_acc_std": res.std_accuracy,
                    "pre_auc": res.mean_auc,
                    "rand_acc": rand.mean_accuracy,
                    "rand_auc": rand.mean_auc,
                    "delta_pp": delta_pp,
                    "delta_auc": delta_auc,
                })
                console.print(
                    f"  {task:18s} / {src:14s}: acc={res.mean_accuracy:.3f} "
                    f"(Δ {delta_pp:+5.1f}pp)  auc={res.mean_auc:.3f} "
                    f"(Δ {delta_auc:+.3f})"
                )

    # Save CSV
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    console.print(f"\n[green]Saved CSV:[/green] {args.out}")

    # Pretty per-task summary tables
    for task in args.tasks:
        t = Table(title=f"task = {task}", show_header=True, header_style="bold")
        t.add_column("checkpoint")
        for src in args.sources:
            t.add_column(f"{src}\nacc / AUC / Δpp", justify="right")
        for ckpt in ckpts:
            ckpt_name = ckpt.parent.name
            row_strs = [ckpt_name]
            for src in args.sources:
                r = next(x for x in rows
                         if x["checkpoint"] == ckpt_name
                         and x["task"] == task
                         and x["source"] == src)
                delta_color = ("[green]" if r["delta_pp"] > 3 else
                               "[red]" if r["delta_pp"] < -3 else "[yellow]")
                cell = (f"{r['pre_acc']:.3f} / {r['pre_auc']:.3f} / "
                        f"{delta_color}{r['delta_pp']:+5.1f}[/]")
                row_strs.append(cell)
            t.add_row(*row_strs)
        # Add random baseline as bottom row
        rand_row = ["random"]
        for src in args.sources:
            rand = rand_cache[(task, src)]
            rand_row.append(f"[dim]{rand.mean_accuracy:.3f} / {rand.mean_auc:.3f} / —[/]")
        t.add_row(*rand_row)
        console.print()
        console.print(t)


if __name__ == "__main__":
    main()
