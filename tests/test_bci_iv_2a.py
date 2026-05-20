"""Smoke tests for the BCI-IV-2a data module — no network, no real GDF needed."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_slm.data.bci_iv_2a import (
    BCI_IV_2A_EEG_CHANNELS,
    _detect_unit_scale,
    channel_intersection_indices,
    restrict_to_bci_channels,
)


def test_module_imports() -> None:
    """All public symbols importable."""
    from eeg_slm.data.bci_iv_2a import (   # noqa: F401
        BCIIV2aLoader,
        SUBJECT_IDS,
        SESSIONS,
        BCI_IV_2A_LABELS,
        build_bci_iv_2a_dataset,
        load_motor_imagery_subject,
    )


def test_eeg_channel_set_size() -> None:
    """BCI-IV-2a has exactly 22 EEG channels."""
    assert len(BCI_IV_2A_EEG_CHANNELS) == 22
    # Each channel appears once
    assert len(set(BCI_IV_2A_EEG_CHANNELS)) == 22


def test_eeg_channels_are_uppercase() -> None:
    """Channel names are normalized to uppercase for clean cross-dataset comparison."""
    for ch in BCI_IV_2A_EEG_CHANNELS:
        assert ch == ch.upper(), f"'{ch}' is not uppercase"


def test_detect_unit_scale_microvolts() -> None:
    """EEG data at µV scale (typical ±50-200) should be scaled by 1e-6 to get volts."""
    rng = np.random.default_rng(0)
    X_uv = rng.normal(0, 20, size=(22, 1000))  # ~±60 µV at 3σ
    scale = _detect_unit_scale(X_uv)
    assert scale == 1e-6


def test_detect_unit_scale_volts() -> None:
    """Data already in volts (tiny values) should not be rescaled."""
    rng = np.random.default_rng(1)
    X_v = rng.normal(0, 2e-5, size=(22, 1000))  # ~±6e-5 V at 3σ
    scale = _detect_unit_scale(X_v)
    assert scale == 1.0


def test_channel_intersection_indices_basic() -> None:
    """Returns indices in target order, missing channels raise."""
    source = ["FZ", "CZ", "PZ", "FC1", "CP1", "POZ"]
    target = ["CZ", "FZ", "POZ"]
    idx = channel_intersection_indices(source, target)
    assert idx == [1, 0, 5]


def test_channel_intersection_indices_missing_raises() -> None:
    """A channel in target but not in source raises a clear ValueError."""
    source = ["FZ", "CZ", "PZ"]
    target = ["CZ", "FZ", "OZ"]   # OZ missing
    with pytest.raises(ValueError, match="Channels not in source"):
        channel_intersection_indices(source, target)


def test_channel_intersection_indices_dot_normalization() -> None:
    """EEGMMIDB's 'FP1.' should match BCI-IV-2a's 'FP1' style channels."""
    source = ["FP1.", "FZ.", "C3."]
    target = ["FZ", "FP1"]
    idx = channel_intersection_indices(source, target)
    assert idx == [1, 0]


def test_restrict_to_bci_channels_selects_subset() -> None:
    """Round-trip: restrict a 64-channel array to BCI-IV-2a's 22, in BCI order."""
    # 64 fake channel names — must contain all 22 BCI-IV-2a channels
    extra = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10",
             "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10",
             "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10",
             "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
             "R1", "R2"]
    source = list(BCI_IV_2A_EEG_CHANNELS) + extra
    assert len(source) == 64

    X = np.arange(2 * 64 * 5, dtype=np.float32).reshape(2, 64, 5)
    X22 = restrict_to_bci_channels(X, source)
    assert X22.shape == (2, 22, 5)
    # The first 22 channels of `source` are exactly BCI_IV_2A_EEG_CHANNELS in order,
    # so X22 should equal X[:, :22, :].
    np.testing.assert_array_equal(X22, X[:, :22, :])


def test_eegmmidb_intersection_full() -> None:
    """Sanity check that all 22 BCI-IV-2a EEG channels are present in
    EEGMMIDB's 10-10 layout (the assumption behind cross-dataset transfer)."""
    eegmmidb_channels = [
        "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
        "FP1", "FPZ", "FP2",
        "AF7", "AF3", "AFZ", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FT8", "T7", "T8", "T9", "T10",
        "TP7", "TP8",
        "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POZ", "PO4", "PO8",
        "O1", "OZ", "O2", "IZ",
    ]
    assert len(eegmmidb_channels) == 64
    # Should not raise — every BCI-IV-2a channel is in EEGMMIDB
    indices = channel_intersection_indices(eegmmidb_channels, BCI_IV_2A_EEG_CHANNELS)
    assert len(indices) == 22
    assert len(set(indices)) == 22  # all unique
