"""Downstream evaluation utilities for eeg-slm.

Phase 1 evaluation = linear probe on labeled EEG tasks. We freeze the encoder,
pool token embeddings to a single fixed-size feature vector per epoch, and
train a simple linear classifier (sklearn LogisticRegression by default) on
top. Comparing the same probe on (a) a pretrained encoder vs (b) a randomly
initialized encoder tells us whether SIGReg pretraining produced useful
representations.
"""

from eeg_slm.eval.linear_probe import (
    FeatureSource,
    LinearProbeResult,
    extract_features,
    extract_features_jepa,
    linear_probe_loso,
    linear_probe_loso_from_features,
)
from eeg_slm.eval.motor_imagery import (
    EEGMMIDB_MI_LABELS,
    EEGMMIDB_REST_VS_ACTIVITY_LABELS,
    Task,
    build_motor_imagery_dataset,
)

__all__ = [
    "FeatureSource",
    "LinearProbeResult",
    "extract_features",
    "extract_features_jepa",
    "linear_probe_loso",
    "linear_probe_loso_from_features",
    "EEGMMIDB_MI_LABELS",
    "EEGMMIDB_REST_VS_ACTIVITY_LABELS",
    "Task",
    "build_motor_imagery_dataset",
]
