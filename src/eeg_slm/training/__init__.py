"""Training utilities for eeg-slm."""

from eeg_slm.training.logger import CSVLogger, embedding_stats
from eeg_slm.training.schedules import cosine_with_warmup
from eeg_slm.training.supervised import SupervisedTrainConfig, supervised_loso
from eeg_slm.training.trainer import TrainConfig, train

__all__ = [
    "CSVLogger",
    "embedding_stats",
    "cosine_with_warmup",
    "SupervisedTrainConfig",
    "supervised_loso",
    "TrainConfig",
    "train",
]
