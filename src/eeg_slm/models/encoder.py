"""EEG encoder — per-patch independent embedding.

Each non-overlapping (channels × patch_size) time-slice is encoded *independently*
to a single d-dim embedding. The encoder MUST NOT mix information across patches —
if it did, the encoder embedding z_t would leak information about z_{t+1}, and the
LeJEPA-style next-token prediction objective would collapse.

The encoder is therefore:

    PatchEmbed (Conv1d, kernel=stride=patch_size)   ← all-channel time-slice → d-dim
        ↓
    sinusoidal positional embedding                 ← time index encoding
        ↓
    per-patch MLP layers                            ← Linear/GELU only; NO cross-patch op
        ↓
    BatchNorm1d projector                           ← critical: NOT LayerNorm
                                                       (LayerNorm would force per-sample
                                                       unit norm and destroy SIGReg)

References
----------
Balestriero & LeCun (2025), LeJEPA: Provable and Scalable Self-Supervised Learning
    Without the Heuristics. arXiv:2511.08544.
LeCun et al. (2026), LeWorldModel: Stable End-to-End Joint-Embedding Predictive
    Architecture from Pixels. arXiv:2603.19312.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from eeg_slm.models.transformer import sinusoidal_position_embeddings


@dataclass
class EncoderConfig:
    n_channels: int = 64
    patch_size: int = 40
    embed_dim: int = 192
    mlp_depth: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    add_position_embed: bool = True


class PerPatchMLP(nn.Module):
    """A small MLP applied independently to each token along the embed_dim.

    Because nn.Linear is applied to the last dimension only, this is genuinely
    per-patch — no cross-time leakage by construction.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([
                nn.LayerNorm(embed_dim),  # internal LN is fine; only the FINAL norm matters for SIGReg
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, embed_dim),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D). Returns (B, T, D)."""
        return x + self.net(x)  # residual around the whole MLP stack


class EEGEncoder(nn.Module):
    """Per-patch independent encoder for multi-channel EEG.

    Forward
    -------
    x : (B, C, T) — raw EEG, C channels × T samples
    returns : (B, n_patches, embed_dim) — one embedding per non-overlapping
              patch of length patch_size

    The output passes through BatchNorm1d (not LayerNorm) so SIGReg can
    meaningfully constrain the embedding distribution.
    """

    def __init__(self, cfg: EncoderConfig | None = None, **overrides) -> None:
        super().__init__()
        if cfg is None:
            cfg = EncoderConfig()
        if overrides:
            cfg = EncoderConfig(**{**cfg.__dict__, **overrides})
        self.cfg = cfg

        # Conv1d patch embed: each output position is a linear projection of
        # one (n_channels, patch_size) slice.
        self.patch_embed = nn.Conv1d(
            in_channels=cfg.n_channels,
            out_channels=cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

        self.per_patch_mlp = PerPatchMLP(
            embed_dim=cfg.embed_dim,
            depth=cfg.mlp_depth,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

        # CRITICAL: BatchNorm, NOT LayerNorm. See module docstring.
        self.projector_bn = nn.BatchNorm1d(cfg.embed_dim)

        # Cache for positional embeddings (lazily created per device/seq-len)
        self._pos_cache: dict[tuple[int, torch.device], Tensor] = {}

    def _get_pos(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (seq_len, device)
        if key not in self._pos_cache:
            self._pos_cache[key] = sinusoidal_position_embeddings(
                seq_len, self.cfg.embed_dim, device=device
            )
        return self._pos_cache[key].to(dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, T) raw EEG. Returns (B, n_patches, embed_dim)."""
        if x.dim() != 3:
            raise ValueError(f"Expected (B, C, T), got shape {tuple(x.shape)}.")
        if x.shape[1] != self.cfg.n_channels:
            raise ValueError(
                f"Expected {self.cfg.n_channels} channels, got {x.shape[1]}."
            )
        if x.shape[2] % self.cfg.patch_size != 0:
            raise ValueError(
                f"Sequence length {x.shape[2]} not divisible by patch size {self.cfg.patch_size}."
            )

        # (B, C, T) → (B, D, T/patch_size) → (B, T/patch_size, D)
        z = self.patch_embed(x).transpose(1, 2)

        if self.cfg.add_position_embed:
            z = z + self._get_pos(z.shape[1], z.device, z.dtype)

        # Per-patch MLP (no cross-time mixing)
        z = self.per_patch_mlp(z)

        # BatchNorm over the embed_dim, computed across the (B * T) flattened batch
        # Equivalent to: flatten (B, T, D) → (B*T, D), BatchNorm1d, reshape back.
        b, t, d = z.shape
        z = self.projector_bn(z.reshape(b * t, d)).reshape(b, t, d)
        return z

    def n_patches(self, seq_len: int) -> int:
        return seq_len // self.cfg.patch_size

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
