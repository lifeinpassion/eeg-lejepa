"""Smoke tests for the training package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eeg_slm.data.dataset import EEGTensorDataset
from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.training.logger import CSVLogger, embedding_stats
from eeg_slm.training.schedules import cosine_with_warmup
from eeg_slm.training.trainer import TrainConfig, train


def test_cosine_with_warmup_shape() -> None:
    """LR goes from 0 → base over warmup, then base → min_lr_ratio*base via cosine."""
    optim = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = cosine_with_warmup(optim, n_warmup_steps=10, n_total_steps=100, min_lr_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(optim.param_groups[0]["lr"])
        optim.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.0, abs=1e-6)
    assert lrs[10] == pytest.approx(1.0, abs=1e-3)
    assert lrs[-1] == pytest.approx(0.1, abs=1e-2)
    # Should be monotonically non-decreasing through warmup
    for i in range(1, 10):
        assert lrs[i] >= lrs[i - 1] - 1e-9


def test_csv_logger_writes_and_reads(tmp_path: Path) -> None:
    log_path = tmp_path / "log.csv"
    with CSVLogger(log_path) as logger:
        logger.log({"step": 0, "loss": 1.5})
        logger.log({"step": 1, "loss": 1.4})
        logger.log({"step": 2, "loss": 1.3})
    content = log_path.read_text().strip().splitlines()
    assert content[0] == "step,loss"
    assert content[-1] == "2,1.3"


def test_embedding_stats_on_isotropic_gaussian() -> None:
    """For genuine N(0, I), abs_mean ≈ 0, std ≈ 1, off-diag ≈ 0."""
    torch.manual_seed(0)
    z = torch.randn(2048, 64)
    s = embedding_stats(z)
    assert s["emb_abs_mean"] < 0.05
    assert 0.9 < s["emb_std"] < 1.1
    assert s["emb_offdiag_abs"] < 0.05


def test_train_runs_and_loss_decreases() -> None:
    """A tiny end-to-end run should produce a CSV and a decreasing total_loss.

    Uses synthetic Gaussian-like input so we don't need EEG data here.
    """
    torch.manual_seed(0)
    # 64 synthetic "epochs"
    X = np.random.RandomState(0).randn(64, 64, 800).astype(np.float32)
    dataset = EEGTensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    model_cfg = EEGLeJEPAConfig()
    model_cfg.encoder.n_channels = 64
    model_cfg.encoder.patch_size = 40
    model_cfg.sigreg_num_slices = 32  # tiny for speed
    model_cfg.predictor.depth = 2     # tiny for speed
    model = EEGLeJEPA(model_cfg)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig(
            n_steps=30, batch_size=8, learning_rate=3e-3,
            warmup_steps=5, log_every=5,
            output_dir=Path(td),
        )
        result = train(model, loader, cfg, device="cpu")
        # CSV exists and is non-empty
        assert Path(result["csv"]).exists()
        # Final checkpoint exists
        assert Path(result["ckpt"]).exists()
        # Loss should generally decrease (compare first vs last logged step)
        import pandas as pd
        df = pd.read_csv(result["csv"])
        assert df["total_loss"].iloc[-1] < df["total_loss"].iloc[0], (
            f"Loss did not decrease: first={df['total_loss'].iloc[0]:.4f}, "
            f"last={df['total_loss'].iloc[-1]:.4f}"
        )
