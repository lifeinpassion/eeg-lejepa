"""EEGLeJEPA — the full Joint-Embedding Predictive Architecture for EEG.

Wraps the encoder, predictor, and SIGReg into a single nn.Module whose forward
returns all losses needed for training:

    L_total = L_pred + λ · L_sigreg

where
    L_pred   = MSE(predictor(z)[:, :-1], encoder(x)[:, 1:])
    L_sigreg = mean over slices of CvM(z @ random unit vectors)

Per LeJEPA / LeWorldModel:
- No stop-gradient
- No EMA target encoder
- No teacher-student split
- Single regularization weight λ (default 0.1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eeg_slm.models.encoder import EEGEncoder, EncoderConfig
from eeg_slm.models.predictor import EEGPredictor, PredictorConfig
from eeg_slm.models.sigreg import SIGReg


@dataclass
class EEGLeJEPAConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    sigreg_weight: float = 0.1
    sigreg_num_slices: int = 256


class EEGLeJEPA(nn.Module):
    """End-to-end EEG JEPA: encoder + causal predictor + SIGReg regularizer."""

    def __init__(self, cfg: EEGLeJEPAConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or EEGLeJEPAConfig()
        # Enforce matched embed_dim between encoder and predictor
        if cfg.encoder.embed_dim != cfg.predictor.embed_dim:
            raise ValueError(
                f"encoder.embed_dim ({cfg.encoder.embed_dim}) != "
                f"predictor.embed_dim ({cfg.predictor.embed_dim})"
            )
        self.cfg = cfg
        self.encoder = EEGEncoder(cfg.encoder)
        self.predictor = EEGPredictor(cfg.predictor)
        self.sigreg = SIGReg(
            embed_dim=cfg.encoder.embed_dim,
            num_slices=cfg.sigreg_num_slices,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass returning embeddings, predictions, and all losses.

        Parameters
        ----------
        x : (B, C, T) — raw EEG, already preprocessed and z-scored.

        Returns
        -------
        dict with keys:
            embeddings  : (B, T_patches, D) encoder output
            predictions : (B, T_patches, D) predictor output (position t predicts t+1)
            pred_loss   : scalar — MSE on shifted target
            sigreg_loss : scalar — SIGReg on flattened embeddings
            total_loss  : scalar — pred_loss + λ * sigreg_loss
        """
        embeddings = self.encoder(x)              # (B, T, D)
        predictions = self.predictor(embeddings)  # (B, T, D)

        # Teacher-forced next-step prediction in raw embedding space
        # predictions[:, t] is the predicted embedding at time t+1
        pred_loss = F.mse_loss(predictions[:, :-1], embeddings[:, 1:])

        # SIGReg on all (batch, time) embeddings flattened
        sigreg_loss = self.sigreg(embeddings)

        total_loss = pred_loss + self.cfg.sigreg_weight * sigreg_loss

        return {
            "embeddings": embeddings,
            "predictions": predictions,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "total_loss": total_loss,
        }

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Inference-only: return encoder embeddings (B, T_patches, D)."""
        self.eval()
        return self.encoder(x)

    @property
    def num_parameters(self) -> dict[str, int]:
        return {
            "encoder": self.encoder.num_parameters,
            "predictor": self.predictor.num_parameters,
            "sigreg": sum(p.numel() for p in self.sigreg.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
