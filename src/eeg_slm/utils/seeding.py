"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed
        Integer seed to set.
    deterministic
        If True, force deterministic algorithms in PyTorch where possible.
        This may slow training and is not guaranteed to be bit-exact across
        devices (especially MPS), but it's still useful for debugging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def get_device(prefer: str = "auto") -> str:
    """Return the best available device string.

    Parameters
    ----------
    prefer
        One of "auto", "cuda", "mps", "cpu". "auto" picks the first
        available in the order cuda > mps > cpu.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for device detection.") from exc

    if prefer != "auto":
        return prefer

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
