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
    p.add_argument("--out", type=Path, default=Path("runs/eeg-lejepa-dev"),
                   help="Output directory for logs + checkpoints")
    p.add_argument("--bf16", action="store_true", help="Enable bf16 autocast")
    p.add_argument("--no-plot", action="store_true", help="Skip the final plot")
    p.add_argument("--sigreg-weight", type=float, default=None,
                   help="Override SIGReg λ (default 0.1 per paper; try 1.0 if collapsing)")
    p.add_argument("--sigreg-slices", type=int, default=None,
                   help="Override SIGReg num_slices (default 256; paper default 1024)")
    p.add_argument("--predictor-depth", type=int, default=None,
                   help="Override predictor depth (default 4; try 2 to reduce shortcut capacity)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])

    subjects = args.subjects if args.subjects is not None else cfg["dataset"]["subjects"]
    runs = cfg["dataset"]["runs"]

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
    console.print(f"[bold]Building pretraining tensor[/bold] from "
                  f"subjects={subjects}, runs={runs}")
    X = build_eegmmidb_pretraining_tensor(
        subjects=subjects, runs=runs,
        data_root=cfg["paths"]["data_root"], preprocessing=pp,
    )
    console.print(f"  → {X.shape} (epochs, channels, samples)")
    dataset = EEGTensorDataset(X)

    # 2. DataLoader
    from torch.utils.data import DataLoader
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 8)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0,  # M1: keep 0 to avoid fork overhead; on AutoDL bump to 4
        drop_last=True,
    )

    # 3. Model
    model_cfg = EEGLeJEPAConfig()
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
        warmup_steps=cfg["training"].get("warmup_steps", 30),
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
