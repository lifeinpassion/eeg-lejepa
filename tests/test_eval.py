"""Smoke tests for the eval package — no network, no real EEG required."""

from __future__ import annotations

import warnings

import numpy as np
import torch

from eeg_slm.eval.linear_probe import (
    LinearProbeResult,
    extract_features,
    extract_features_jepa,
    linear_probe_loso,
)
from eeg_slm.models import EEGEncoder, EEGLeJEPA, EEGLeJEPAConfig, EncoderConfig

# Silence sklearn 1.8 deprecation warning about n_jobs (we no longer pass it)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def test_extract_features_shape() -> None:
    enc = EEGEncoder(EncoderConfig(n_channels=64, patch_size=40, embed_dim=192))
    X = np.random.RandomState(0).randn(12, 64, 800).astype(np.float32)
    feats = extract_features(enc, X, device="cpu", batch_size=4, pool="mean")
    assert feats.shape == (12, 192)
    assert feats.dtype == np.float32


def test_extract_features_max_pool() -> None:
    enc = EEGEncoder(EncoderConfig(n_channels=64, patch_size=40, embed_dim=192))
    X = np.random.RandomState(1).randn(4, 64, 800).astype(np.float32)
    feats = extract_features(enc, X, device="cpu", batch_size=2, pool="max")
    assert feats.shape == (4, 192)


def test_loso_with_separable_synthetic() -> None:
    """If features are linearly separable, LOSO accuracy should be near 1.0."""
    torch.manual_seed(0)
    enc = EEGEncoder(EncoderConfig(n_channels=64, patch_size=40, embed_dim=192))

    # Synthesize a strongly separable problem: half the inputs have a constant
    # offset that survives the encoder (modulo BN). 4 subjects × 6 epochs each.
    rng = np.random.RandomState(7)
    X = rng.randn(24, 64, 800).astype(np.float32) * 0.1
    y = np.array([0, 0, 0, 1, 1, 1] * 4, dtype=np.int64)
    subject_ids = np.repeat(np.arange(4), 6)
    # Inject a class-correlated pattern on a few channels
    X[y == 1, :8, :] += 3.0

    res = linear_probe_loso(
        encoder=enc, X=X, y=y, subject_ids=subject_ids, device="cpu",
    )
    assert isinstance(res, LinearProbeResult)
    assert len(res.fold_accuracies) == 4
    # With strong signal, even a random-init encoder should easily beat chance
    assert res.mean_accuracy > 0.7, (
        f"LOSO accuracy {res.mean_accuracy:.3f} unexpectedly low for a separable signal."
    )


def test_loso_chance_baseline_recorded() -> None:
    """Result should record the actual class-balance chance level."""
    torch.manual_seed(0)
    enc = EEGEncoder(EncoderConfig(n_channels=64, patch_size=40, embed_dim=192))
    rng = np.random.RandomState(11)
    X = rng.randn(16, 64, 800).astype(np.float32)
    # Imbalanced: 12 zeros, 4 ones → chance = 0.75
    y = np.array([0]*12 + [1]*4, dtype=np.int64)
    subject_ids = np.array([0]*8 + [1]*8, dtype=np.int64)
    res = linear_probe_loso(enc, X, y, subject_ids, device="cpu")
    assert abs(res.chance - 0.75) < 1e-9, f"chance was {res.chance}, expected 0.75"
