"""Learning-rate schedules."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    n_warmup_steps: int,
    n_total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Linear warmup then cosine decay to `min_lr_ratio * base_lr`.

    Standard schedule for small transformer pretraining. Linear warmup over
    the first `n_warmup_steps`, then cosine from 1.0 to `min_lr_ratio` over
    the remaining steps.
    """
    if n_warmup_steps < 0 or n_warmup_steps > n_total_steps:
        raise ValueError(
            f"n_warmup_steps ({n_warmup_steps}) must be in [0, {n_total_steps}]."
        )

    def lr_lambda(step: int) -> float:
        if step < n_warmup_steps:
            return float(step) / max(1, n_warmup_steps)
        progress = (step - n_warmup_steps) / max(1, n_total_steps - n_warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
