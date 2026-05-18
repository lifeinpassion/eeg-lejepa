"""Profile EEGLeJEPA forward and backward passes on the active device.

Runs:
  - 3 warm-up forwards (kernel JIT compilation)
  - 10 timed forwards (no_grad)
  - 10 timed forward+backward steps

Uses proper MPS synchronization for accurate wall-clock timing. On a clean
M1 with deterministic mode OFF, the warm forward should be in the tens of
milliseconds; if it's still seconds, something else is wrong.
"""

from __future__ import annotations

import statistics
import time

import torch
from rich.console import Console
from rich.table import Table

from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_block(fn, device: str) -> float:
    synchronize(device)
    t0 = time.perf_counter()
    fn()
    synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    set_global_seed(42, deterministic=False)
    device = get_device("auto")
    console.print(f"[bold]Device:[/bold] {device}  "
                  f"[dim](deterministic mode: OFF)[/dim]")

    model = EEGLeJEPA(EEGLeJEPAConfig(sigreg_num_slices=256)).to(device)
    n_params = model.num_parameters["total"]
    console.print(f"[bold]Model:[/bold] EEGLeJEPA, {n_params/1e6:.2f}M params")

    # Synthetic input — same shape as real preprocessed EEG
    x = torch.randn(8, 64, 800, device=device)
    console.print(f"[bold]Input:[/bold] {tuple(x.shape)} on {x.device}")

    # Warm-up
    console.print("\n[bold]Warming up (3 forwards)...[/bold]")
    model.eval()
    warmup_times = []
    for i in range(3):
        ms = time_block(lambda: model(x), device)
        warmup_times.append(ms)
        console.print(f"  warmup forward {i}: {ms:.1f} ms")

    # Timed forward
    n = 10
    fwd_times = []
    for _ in range(n):
        with torch.no_grad():
            ms = time_block(lambda: model(x), device)
        fwd_times.append(ms)

    # Timed forward + backward
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fb_times = []
    for _ in range(n):
        def step():
            optim.zero_grad()
            out = model(x)
            out["total_loss"].backward()
            optim.step()
        ms = time_block(step, device)
        fb_times.append(ms)

    # Report
    table = Table(title=f"Timing summary ({n} runs each, after warmup)",
                  show_header=True, header_style="bold")
    table.add_column("Phase")
    table.add_column("Median (ms)", justify="right")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    table.add_row("Forward (no_grad)",
                  f"{statistics.median(fwd_times):.1f}",
                  f"{min(fwd_times):.1f}",
                  f"{max(fwd_times):.1f}")
    table.add_row("Forward + backward + step",
                  f"{statistics.median(fb_times):.1f}",
                  f"{min(fb_times):.1f}",
                  f"{max(fb_times):.1f}")
    console.print()
    console.print(table)

    console.print(
        f"\n[bold]Throughput estimate:[/bold] "
        f"{1000.0 / statistics.median(fb_times) * 8:.0f} epochs/sec "
        f"(at batch=8). For a 300-step training run: "
        f"~{300 * statistics.median(fb_times) / 1000:.0f} seconds wall-clock."
    )


if __name__ == "__main__":
    main()
