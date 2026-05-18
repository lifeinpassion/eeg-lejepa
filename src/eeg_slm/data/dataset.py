"""PyTorch Dataset over preprocessed EEG epochs.

For Phase 1 scale (~3-20 subjects, ~1k-10k epochs total), we hold the entire
preprocessed corpus in memory as one (N, C, T) float32 tensor. This is fast,
simple, and sidesteps disk I/O bottlenecks during M1 development. When we
scale to TUH-EEG on AutoDL, we'll replace this with a memory-mapped or
on-the-fly variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg_slm.data.loaders import EEGMMIDBLoader
from eeg_slm.data.preprocessing import (
    PreprocessingConfig,
    fixed_length_epochs,
    preprocess_raw,
    to_numpy,
    zscore_per_channel,
)


@dataclass
class EEGTensorDataset(Dataset):
    """In-memory Dataset over a (N, C, T) tensor of preprocessed EEG epochs.

    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor of shape (N, C, T), float32 preferred.
        Should already be preprocessed (filtered, referenced, resampled,
        epoched) and ideally per-channel z-scored.
    """

    X: torch.Tensor

    def __post_init__(self) -> None:
        if isinstance(self.X, np.ndarray):
            self.X = torch.from_numpy(self.X)
        if self.X.dtype != torch.float32:
            self.X = self.X.float()
        if self.X.dim() != 3:
            raise ValueError(f"Expected (N, C, T), got shape {tuple(self.X.shape)}.")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        return self.X.shape[2]


def build_eegmmidb_pretraining_tensor(
    subjects: list[int],
    runs: list[int],
    data_root: str | Path,
    preprocessing: PreprocessingConfig,
    to_microvolts: bool = True,
    zscore: bool = True,
) -> np.ndarray:
    """Load + preprocess + epoch + (optionally) z-score multiple subjects.

    Returns a single (N_total_epochs, C, T) float32 array, suitable for
    wrapping in `EEGTensorDataset`.
    """
    loader = EEGMMIDBLoader(data_root=Path(data_root))
    pieces: list[np.ndarray] = []
    for s in subjects:
        raw = loader.load_raw(subject=s, runs=runs)
        raw_pp = preprocess_raw(raw, preprocessing)
        epochs = fixed_length_epochs(raw_pp, preprocessing)
        X = to_numpy(epochs, to_microvolts=to_microvolts)
        if zscore:
            X = zscore_per_channel(X)
        pieces.append(X)
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
