"""Training loop for EEGLeJEPA pretraining.

Functional rather than class-based — easier to test, less ceremony. The `train`
function runs `n_steps` of gradient updates over an iterable DataLoader,
logging metrics every `log_every` steps and saving checkpoints every
`ckpt_every` steps.

Optimizer: AdamW. Schedule: cosine with linear warmup. Gradient clipping by L2
norm. Optional bf16 autocast (off by default on M1 — MPS bf16 is not always
faster for our scale and can interact awkwardly with `torch.sort`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader

from eeg_slm.training.logger import CSVLogger, StepTimer, embedding_stats
from eeg_slm.training.schedules import cosine_with_warmup

console = Console()


@dataclass
class TrainConfig:
    n_steps: int = 300
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 30
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    ckpt_every: int = 0          # 0 = save only at the end
    use_bf16: bool = False
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("runs/run-default"))


def _iter_forever(loader: Iterable) -> Iterable:
    """Cycle a dataloader indefinitely. Used so step count drives termination."""
    while True:
        for batch in loader:
            yield batch


def train(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    device: str = "cpu",
) -> dict:
    """Pretrain `model` (an EEGLeJEPA) on `loader` for `cfg.n_steps` updates.

    Returns a dict summarizing the run (path to CSV log + final checkpoint).
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.output_dir / "train_log.csv"
    ckpt_path = cfg.output_dir / "model_final.pt"

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = cosine_with_warmup(
        optimizer,
        n_warmup_steps=cfg.warmup_steps,
        n_total_steps=cfg.n_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )

    autocast_ctx = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if cfg.use_bf16 and device in ("cuda", "mps")
        else _NullCtx()
    )

    iterator = iter(_iter_forever(loader))
    timer = StepTimer()
    timer.tick()  # zero out

    console.print(f"[bold]Training[/bold] {cfg.n_steps} steps "
                  f"(bs={cfg.batch_size}, lr={cfg.learning_rate}) on {device}")
    console.print(f"  Output dir: {cfg.output_dir}")

    with CSVLogger(csv_path) as logger:
        for step in range(cfg.n_steps):
            batch = next(iterator).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                out = model(batch)
                loss = out["total_loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.grad_clip
            ).item()
            optimizer.step()
            scheduler.step()

            if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
                stats = embedding_stats(out["embeddings"])
                wall_dt = timer.tick()
                row = {
                    "step": step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "total_loss": out["total_loss"].item(),
                    "pred_loss": out["pred_loss"].item(),
                    "sigreg_loss": out["sigreg_loss"].item(),
                    "grad_norm": grad_norm,
                    "wall_dt_s": wall_dt,
                    **stats,
                }
                logger.log(row)
                console.print(
                    f"  step {step:4d} | "
                    f"loss {row['total_loss']:.4f} "
                    f"(pred {row['pred_loss']:.4f}, sig {row['sigreg_loss']:.3f}) | "
                    f"|mean|={row['emb_abs_mean']:.3f} std={row['emb_std']:.3f} "
                    f"off-diag={row['emb_offdiag_abs']:.4f} | "
                    f"gn {grad_norm:.3f} | "
                    f"lr {row['lr']:.2e} | "
                    f"{wall_dt:.1f}s"
                )

            if cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
                torch.save(model.state_dict(), cfg.output_dir / f"model_step_{step}.pt")

    torch.save(model.state_dict(), ckpt_path)
    console.print(f"[green]Done.[/green] Final checkpoint: {ckpt_path}")
    return {"csv": str(csv_path), "ckpt": str(ckpt_path)}


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *args): return False
