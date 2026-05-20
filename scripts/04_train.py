"""End-to-end pretraining: load EEG, build EEGLeJEPA, train for N steps, plot.

Run from project root:

    python scripts/04_train.py             # default: 300 steps, batch 8
    python scripts/04_train.py --steps 100 # quick smoke test
    make train                              # via Makefile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console

from eeg_slm.data import (
    EEGTensorDataset,
    PreprocessingConfig,
    build_eegmmidb_pretraining_tensor,
)
from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.training import TrainConfig, train
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--steps", type=int, default=None, help="Override n_steps from config")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--subjects", type=int, nargs="*", default=None,
                   help="Override config subject list")
    p.add_argument("--runs", type=int, nargs="*", default=None,
                   help="Override config run list (e.g. --runs 4 8 12 to use MI-only data)")
    p.add_argument("--out", type=Path, default=Path("runs/eeg-lejepa-dev"),
                   help="Output directory for logs + checkpoints")
    p.add_argument("--model-size", choices=["base", "large"], default="base",
                   help="Architecture preset: base ≈ 2.86M params (default), large ≈ 7M params")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="Override LR-warmup step count (default from config; bump for larger models)")
    p.add_argument("--channel-subset", default=None,
                   help="Restrict EEGMMIDB pretraining to a channel subset. Either the literal "
                        "'bci-iv-2a' (22 channels matching BCI Competition IV Dataset 2a), or a "
                        "comma-separated list of channel names. Default uses all 64 channels.")
    p.add_argument("--bf16", action="store_true", help="Enable bf16 autocast")
    p.add_argument("--no-plot", action="store_true", help="Skip the final plot")
    p.add_argument("--device", default=None,
                   help="Override device (cuda / mps / cpu). Default reads config (auto).")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader num_workers. Default 0 on M1, 4 on CUDA (auto).")
    p.add_argument("--sigreg-weight", type=float, default=None,
                   help="Override SIGReg λ (default 1.0 per Session 3 calibration)")
    p.add_argument("--sigreg-slices", type=int, default=None,
                   help="Override SIGReg num_slices (default 256; paper default 1024)")
    p.add_argument("--predictor-depth", type=int, default=None,
                   help="Override predictor depth (default 4; try 2 to reduce shortcut capacity)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(args.device if args.device else cfg["training"]["device"])

    subjects = args.subjects if args.subjects is not None else cfg["dataset"]["subjects"]
    runs = args.runs if args.runs is not None else cfg["dataset"]["runs"]

    # 1. Data
    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )
    # Resolve --channel-subset
    channel_subset: list[str] | None = None
    if args.channel_subset is not None:
        if args.channel_subset.lower() in ("bci-iv-2a", "bci_iv_2a", "bci_iv2a"):
            from eeg_slm.data.bci_iv_2a import BCI_IV_2A_EEG_CHANNELS
            channel_subset = list(BCI_IV_2A_EEG_CHANNELS)
        else:
            channel_subset = [c.strip() for c in args.channel_subset.split(",") if c.strip()]

    console.print(f"[bold]Building pretraining tensor[/bold] from "
                  f"subjects={subjects}, runs={runs}")
    if channel_subset is not None:
        console.print(f"  channel-subset: {len(channel_subset)} channels — "
                      f"{channel_subset[:5]}{' ...' if len(channel_subset) > 5 else ''}")
    X = build_eegmmidb_pretraining_tensor(
        subjects=subjects, runs=runs,
        data_root=cfg["paths"]["data_root"], preprocessing=pp,
        channel_subset=channel_subset,
    )
    console.print(f"  → {X.shape} (epochs, channels, samples)")
    dataset = EEGTensorDataset(X)

    # 2. DataLoader
    from torch.utils.data import DataLoader
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 8)
    # Default num_workers: 4 for CUDA (multi-GPU dataloading worth the fork cost),
    # 0 elsewhere (M1 MPS / CPU — fork overhead exceeds parallel speedup at our scale).
    default_workers = 4 if device == "cuda" else 0
    num_workers = args.num_workers if args.num_workers is not None else default_workers
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    console.print(f"  DataLoader: batch={batch_size}, num_workers={num_workers}, "
                  f"pin_memory={device == 'cuda'}")

    # 3. Model
    model_cfg = (EEGLeJEPAConfig.large() if args.model_size == "large"
                 else EEGLeJEPAConfig.base())
    model_cfg.encoder.n_channels = dataset.n_channels
    model_cfg.encoder.patch_size = 40
    model_cfg.sigreg_num_slices = args.sigreg_slices if args.sigreg_slices is not None else 256
    if args.sigreg_weight is not None:
        model_cfg.sigreg_weight = args.sigreg_weight
    if args.predictor_depth is not None:
        model_cfg.predictor.depth = args.predictor_depth
    model = EEGLeJEPA(model_cfg)
    console.print(
        f"  sigreg_weight={model_cfg.sigreg_weight}, "
        f"num_slices={model_cfg.sigreg_num_slices}, "
        f"predictor.depth={model_cfg.predictor.depth}"
    )
    console.print(f"[bold]Model:[/bold] EEGLeJEPA "
                  f"({model.num_parameters['total']/1e6:.2f}M params total)")

    # 4. Train
    train_cfg = TrainConfig(
        n_steps=args.steps if args.steps is not None else cfg["training"].get("n_steps", 300),
        batch_size=batch_size,
        learning_rate=args.lr if args.lr is not None else cfg["training"].get("learning_rate", 1e-3),
        weight_decay=cfg["training"].get("weight_decay", 0.05),
        warmup_steps=args.warmup_steps if args.warmup_steps is not None
                     else cfg["training"].get("warmup_steps", 30),
        grad_clip=cfg["training"].get("grad_clip", 1.0),
        log_every=cfg["training"].get("log_every", 10),
        use_bf16=args.bf16,
        seed=cfg["training"]["seed"],
        output_dir=args.out,
    )
    result = train(model, loader, train_cfg, device=device)

    # 5. Plot
    if not args.no_plot:
        try:
            # _plot_train.py lives next to this script; add its dir to sys.path so
            # we can import it without making scripts/ a real package.
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _plot_train import plot_training_log
            plot_path = args.out / "train_curves.png"
            plot_training_log(result["csv"], plot_path)
            console.print(f"[green]Saved plot:[/green] {plot_path}")
        except Exception as e:
            console.print(f"[yellow]Plot skipped:[/yellow] {e}")


if __name__ == "__main__":
    main()
