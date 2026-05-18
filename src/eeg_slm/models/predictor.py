"""EEG predictor — autoregressive next-embedding prediction.

Takes a sequence of per-patch encoder embeddings (z_1, ..., z_T) and predicts
the next-step embedding sequence (ẑ_2, ..., ẑ_{T+1}) via a small causal
Transformer. Trained to minimize MSE against the encoder's own outputs at the
shifted positions:

    L_pred = MSE(predictor(z)[:, :-1], encoder(x)[:, 1:])

Per LeJEPA / LeWorldModel: NO stop-gradient, NO EMA, NO teacher network.
Gradients flow through both the prediction side and the target side from a
single shared encoder. SIGReg prevents the trivial-solution collapse that
would otherwise occur.

The predictor also ends in BatchNorm1d (not LayerNorm), matching the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from eeg_slm.models.transformer import TransformerBlock


@dataclass
class PredictorConfig:
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1


class EEGPredictor(nn.Module):
    """Causal Transformer that predicts the next-step embedding.

    Forward
    -------
    z : (B, T, D) — sequence of encoder embeddings
    returns : (B, T, D) — predicted embeddings, where position t is the prediction
              of z_{t+1} given (z_1, ..., z_t). Use predictions[:, :-1] against
              z[:, 1:] for the loss.
    """

    def __init__(self, cfg: PredictorConfig | None = None, **overrides) -> None:
        super().__init__()
        if cfg is None:
            cfg = PredictorConfig()
        if overrides:
            cfg = PredictorConfig(**{**cfg.__dict__, **overrides})
        self.cfg = cfg

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                is_causal=True,
            )
            for _ in range(cfg.depth)
        ])

        # CRITICAL: BatchNorm, NOT LayerNorm. Matches encoder convention.
        self.projector_bn = nn.BatchNorm1d(cfg.embed_dim)

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, T, D). Returns predictions (B, T, D)."""
        if z.dim() != 3:
            raise ValueError(f"Expected (B, T, D), got shape {tuple(z.shape)}.")
        x = z
        for block in self.blocks:
            x = block(x)
        b, t, d = x.shape
        x = self.projector_bn(x.reshape(b * t, d)).reshape(b, t, d)
        return x

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
