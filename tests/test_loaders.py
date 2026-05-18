"""Smoke tests for the data layer.

These tests are intentionally minimal — they verify the modules import cleanly
and configs parse without errors. Network-dependent tests (actual download)
are gated behind an environment variable so CI runs are fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


def test_package_imports() -> None:
    import eeg_slm  # noqa: F401
    from eeg_slm import data  # noqa: F401
    from eeg_slm.data import loaders, preprocessing  # noqa: F401
    from eeg_slm.utils import seeding  # noqa: F401


def test_default_config_parses() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert "dataset" in cfg
    assert cfg["dataset"]["name"] == "eegmmidb"
    assert isinstance(cfg["dataset"]["subjects"], list)


def test_preprocessing_config_defaults() -> None:
    from eeg_slm.data.preprocessing import PreprocessingConfig

    pc = PreprocessingConfig()
    assert pc.bandpass_low_hz < pc.bandpass_high_hz
    assert pc.epoch_length_s > 0


def test_seeding_works_without_torch_or_numpy() -> None:
    from eeg_slm.utils.seeding import set_global_seed

    # Should not raise even if optional libraries are missing
    set_global_seed(42, deterministic=False)


@pytest.mark.skipif(
    os.environ.get("EEG_SLM_RUN_NETWORK_TESTS") != "1",
    reason="Network test (download); set EEG_SLM_RUN_NETWORK_TESTS=1 to enable",
)
def test_download_one_subject(tmp_path: Path) -> None:
    """Actual download of one (subject, run) pair. Slow; gated by env var."""
    from eeg_slm.data.loaders import EEGMMIDBLoader

    loader = EEGMMIDBLoader(data_root=tmp_path / "raw")
    loader.download_subjects(subjects=[1], runs=[3])
    raw = loader.load_raw(subject=1, runs=[3])
    assert raw.info["nchan"] >= 32
