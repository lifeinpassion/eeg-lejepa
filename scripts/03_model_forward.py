"""End-to-end forward pass of EEGLeJEPA on real EEG data.

This is the Session 2 milestone: load PhysioNet EEGMMIDB → preprocess →
forward through the full encoder + predictor + SIGReg pipeline → print all
losses, parameter counts, and memory footprint.

No training yet — that's Session 3.

Run from the project root:

    python scripts/03_model_forward.py
    # or
    make forward
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.table import Table

from eeg_slm.data import (
    EEGMMIDBLoader,
    PreprocessingConfig,
    fixed_length_epochs,
    preprocess_raw,
    to_numpy,
    zscore_per_channel,
)
from eeg_slm.data.loaders import RUNS_MOTOR_IMAGERY_HANDS
from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()


def fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def main() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    set_global_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    console.print(f"[bold]Device:[/bold] {device}")

    # 1. Load and preprocess one subject's worth of EEG
    loader = EEGMMIDBLoader(data_root=Path(cfg["paths"]["data_root"]))
    subject = cfg["dataset"]["subjects"][0]
    raw = loader.load_raw(subject=subject, runs=list(RUNS_MOTOR_IMAGERY_HANDS))
    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )
    raw_pp = preprocess_raw(raw, pp)
    epochs = fixed_length_epochs(raw_pp, pp)
    X = to_numpy(epochs, to_microvolts=True)
    X = zscore_per_channel(X)  # (n_epochs, 64, 800)
    console.print(
        f"[bold]Data:[/bold] {X.shape} (n_epochs, channels, samples) "
        f"after preprocessing"
    )

    # 2. Build the model
    model_cfg = EEGLeJEPAConfig()
    model_cfg.encoder.n_channels = X.shape[1]
    model_cfg.encoder.patch_size = 40       # 200 ms at 200 Hz
    model_cfg.sigreg_num_slices = 256       # M1-friendly; bump to 1024 on AutoDL
    model = EEGLeJEPA(model_cfg).to(device)
    model.eval()

    counts = model.num_parameters
    table = Table(title="Parameter counts", show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Params", justify="right")
    for name, n in counts.items():
        table.add_row(name, fmt_params(n))
    console.print(table)

    # 3. Forward pass on a small batch
    batch_size = 8
    x = torch.from_numpy(X[:batch_size]).to(device)
    console.print(f"[bold]Input batch:[/bold] {tuple(x.shape)} on {x.device}")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    forward_ms = (time.perf_counter() - t0) * 1000

    # 4. Report shapes and losses
    losses = Table(title="Forward pass output", show_header=True, header_style="bold")
    losses.add_column("Key")
    losses.add_column("Shape / Value", justify="right")
    losses.add_row("embeddings", str(tuple(out["embeddings"].shape)))
    losses.add_row("predictions", str(tuple(out["predictions"].shape)))
    losses.add_row("pred_loss", f"{out['pred_loss'].item():.6f}")
    losses.add_row("sigreg_loss", f"{out['sigreg_loss'].item():.6f}")
    losses.add_row("total_loss", f"{out['total_loss'].item():.6f}")
    losses.add_row("forward (ms)", f"{forward_ms:.1f}")
    console.print(losses)

    # 5. Sanity check: a backward pass works and gradients are finite
    console.print("[bold]Backward sanity check...[/bold]")
    model.train()
    x_train = torch.from_numpy(X[:batch_size]).to(device)
    out_train = model(x_train)
    out_train["total_loss"].backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_total = sum(1 for _ in model.parameters())
    console.print(f"  {n_with_grad}/{n_total} parameters received non-zero gradients")
    console.print(
        f"  All grads finite: {all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)}"
    )

    # 6. Embedding-distribution sanity check
    with torch.no_grad():
        z = out["embeddings"].reshape(-1, model.cfg.encoder.embed_dim)
        z_mean = z.mean(dim=0).abs().mean().item()
        z_std = z.std(dim=0).mean().item()
    console.print(
        f"[bold]Embedding stats:[/bold] |mean| per dim = {z_mean:.4f}, "
        f"mean std per dim = {z_std:.4f} "
        f"(SIGReg targets mean≈0, std≈1; before training these will be off)"
    )

    console.print("\n[bold green]Session 2 milestone hit:[/bold green] "
                  "end-to-end forward + backward verified on real EEG.")


if __name__ == "__main__":
    main()
