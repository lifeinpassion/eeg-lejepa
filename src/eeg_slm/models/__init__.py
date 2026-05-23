"""Model architectures for eeg-slm.

Phase 1: EEGLeJEPA — small JEPA-style predictive world model for EEG with SIGReg.
"""

from eeg_slm.models.classifier import EEGClassifier, EEGClassifierConfig
from eeg_slm.models.encoder import EEGEncoder, EncoderConfig, PerPatchMLP
from eeg_slm.models.jepa import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.models.predictor import EEGPredictor, PredictorConfig
from eeg_slm.models.sigreg import (
    SIGReg,
    cramer_von_mises_normal,
    cramer_von_mises_normal_batch,
)
from eeg_slm.models.transformer import (
    MultiHeadAttention,
    TransformerBlock,
    sinusoidal_position_embeddings,
)

__all__ = [
    "EEGClassifier",
    "EEGClassifierConfig",
    "EEGEncoder",
    "EncoderConfig",
    "PerPatchMLP",
    "EEGPredictor",
    "PredictorConfig",
    "SIGReg",
    "cramer_von_mises_normal",
    "cramer_von_mises_normal_batch",
    "TransformerBlock",
    "MultiHeadAttention",
    "sinusoidal_position_embeddings",
    "EEGLeJEPA",
    "EEGLeJEPAConfig",
]
