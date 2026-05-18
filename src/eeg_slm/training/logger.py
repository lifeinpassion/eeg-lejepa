"""CSV + console logging for training runs.

CSVLogger writes one row per call, with a header inferred from the first row's
keys. Subsequent rows must use the same keys. Designed to be tail-able during
long runs.

`embedding_stats` computes the SIGReg-relevant diagnostics in a single pass.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class CSVLogger:
    """Append-only CSV logger. Header is locked on the first log call."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._fields: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._fh = self.path.open("w", newline="")
            self._fields = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        # Ensure consistent fields — drop unknown, fill missing with empty
        out = {k: row.get(k, "") for k in self._fields}
        self._writer.writerow(out)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


@torch.no_grad()
def embedding_stats(embeddings: Tensor) -> dict[str, float]:
    """Compute SIGReg-relevant diagnostics on (..., D) embeddings.

    Returns
    -------
    dict with keys:
        emb_abs_mean    : per-dim mean's absolute value, averaged across dims.
                          Target → 0. Indicates whether SIGReg is centering.
        emb_std         : per-dim std, averaged across dims.
                          Target → 1. Indicates whether SIGReg is scaling.
        emb_norm_mean   : average L2 norm of an embedding.
                          For isotropic N(0, I^D), should be ≈ sqrt(D).
        emb_offdiag_abs : average absolute value of off-diagonal entries of the
                          empirical covariance matrix. Target → 0. Indicates
                          decorrelation.
    """
    flat = embeddings.reshape(-1, embeddings.shape[-1]).float()
    n, d = flat.shape
    mean = flat.mean(dim=0)                           # (D,)
    centered = flat - mean
    std = centered.std(dim=0, unbiased=False)          # (D,)
    # Empirical covariance (D x D). For n=B*T~few hundred and D~192 this is small.
    cov = centered.t() @ centered / max(1, n - 1)      # (D, D)
    diag_mask = torch.eye(d, device=cov.device, dtype=torch.bool)
    offdiag = cov[~diag_mask].abs().mean()
    return {
        "emb_abs_mean": mean.abs().mean().item(),
        "emb_std": std.mean().item(),
        "emb_norm_mean": flat.norm(dim=-1).mean().item(),
        "emb_offdiag_abs": offdiag.item(),
    }


class StepTimer:
    """Wall-clock timer that tracks per-step throughput."""

    def __init__(self) -> None:
        self._t_last = time.perf_counter()

    def tick(self) -> float:
        """Return seconds since the last tick (or construction), and reset."""
        now = time.perf_counter()
        dt = now - self._t_last
        self._t_last = now
        return dt
