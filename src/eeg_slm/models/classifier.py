"""Classification head on top of EEGEncoder for supervised baselines.

Used to isolate the SSL contribution: train the same encoder architecture
end-to-end with class labels (cross-entropy) instead of via SIGReg + linear
probe. Comparison gap = how much SSL pretraining actually buys us.

Architecture:
    EEGEncoder (same as SSL) → mean-pool over patches → Linear(D, n_classes)

NB: no predictor module, no SIGReg, no JEPA loss. This is a pure supervised
classifier of the same parameter scale as our SSL encoder (the predictor's
~1.8M params are absent here, so it's ~1.1M for the base encoder).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor, nn

from eeg_slm.models.encoder import EEGEncoder, EncoderConfig


@dataclass
class EEGClassifierConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    n_classes: int = 2


class EEGClassifier(nn.Module):
    """End-to-end supervised classifier: EEGEncoder + mean-pool + linear head."""

    def __init__(self, cfg: EEGClassifierConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or EEGClassifierConfig()
        self.cfg = cfg
        self.encoder = EEGEncoder(cfg.encoder)
        self.head = nn.Linear(cfg.encoder.embed_dim, cfg.n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, T) raw EEG. Returns logits (B, n_classes)."""
        z = self.encoder(x)            # (B, n_patches, D)
        pooled = z.mean(dim=1)         # (B, D) — mean over patches
        return self.head(pooled)       # (B, n_classes)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
