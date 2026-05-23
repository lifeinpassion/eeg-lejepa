"""Smoke tests for encoder, predictor, and full EEGLeJEPA wrapper.

Properties verified:
  - Encoder output has the expected (B, n_patches, embed_dim) shape.
  - Encoder is genuinely per-patch (permuting input along time should
    permute outputs identically — proves no cross-time mixing).
  - Predictor preserves (B, T, D) shape.
  - EEGLeJEPA forward returns all five expected keys with finite losses.
  - Full forward + backward produces finite gradients across all parameters.
  - Parameter counts are within the documented budget.
"""

from __future__ import annotations

import torch

from eeg_slm.models import (
    EEGEncoder,
    EEGLeJEPA,
    EEGLeJEPAConfig,
    EEGPredictor,
    EncoderConfig,
    PredictorConfig,
)


def _make_input(b: int = 2, c: int = 64, t: int = 800) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(b, c, t)


def test_encoder_output_shape() -> None:
    enc = EEGEncoder(EncoderConfig(n_channels=64, patch_size=40, embed_dim=192))
    enc.eval()
    x = _make_input(b=2, c=64, t=800)
    z = enc(x)
    assert z.shape == (2, 20, 192), f"Got {z.shape}"


def test_encoder_is_per_patch_no_cross_time_leakage() -> None:
    """Permuting input patches along the time axis must permute outputs identically.

    If any cross-patch mixing existed (e.g. an accidental attention layer),
    permuting time-windows would change the values, not just their order.
    """
    cfg = EncoderConfig(
        n_channels=64, patch_size=40, embed_dim=192, add_position_embed=False,
    )
    enc = EEGEncoder(cfg)
    enc.eval()
    x = _make_input(b=1, c=64, t=800)
    z = enc(x)  # (1, 20, 192)

    # Build a permutation that swaps time-windows of size patch_size
    perm = torch.randperm(20)
    x_perm = torch.cat([x[:, :, p * 40:(p + 1) * 40] for p in perm.tolist()], dim=-1)
    z_perm = enc(x_perm)

    # z_perm should equal z indexed by the permutation
    expected = z[:, perm, :]
    assert torch.allclose(z_perm, expected, atol=1e-5), (
        "Encoder is mixing information across time-patches — there must NOT "
        "be any cross-patch attention or convolution in the encoder."
    )


def test_predictor_output_shape() -> None:
    pred = EEGPredictor(PredictorConfig(embed_dim=192, depth=4, num_heads=4))
    pred.eval()
    z = torch.randn(2, 20, 192)
    out = pred(z)
    assert out.shape == z.shape


def test_jepa_forward_returns_all_keys() -> None:
    model = EEGLeJEPA(EEGLeJEPAConfig())
    model.eval()
    x = _make_input(b=2, c=64, t=800)
    out = model(x)
    for key in ("embeddings", "predictions", "pred_loss", "sigreg_loss", "total_loss"):
        assert key in out, f"Missing key '{key}' in JEPA output."
    assert out["embeddings"].shape == (2, 20, 192)
    assert out["predictions"].shape == (2, 20, 192)
    for k in ("pred_loss", "sigreg_loss", "total_loss"):
        assert out[k].dim() == 0, f"{k} should be scalar, got shape {out[k].shape}"
        assert torch.isfinite(out[k]), f"{k} is not finite: {out[k]}"


def test_compact_preset_matches_seizure_detector_tokenisation() -> None:
    """compact() at 22 ch / patch 64 must yield the eeg-seizure detector's 256/64/128 shapes."""
    cfg = EEGLeJEPAConfig.compact()
    cfg.encoder.n_channels = 22
    cfg.encoder.patch_size = 64
    assert cfg.encoder.embed_dim == 128 and cfg.predictor.embed_dim == 128
    model = EEGLeJEPA(cfg)
    model.eval()
    x = _make_input(b=2, c=22, t=1024)            # 4 s @ 256 Hz
    out = model(x)
    assert out["embeddings"].shape == (2, 16, 128)   # 1024 / 64 = 16 patches
    assert out["predictions"].shape == (2, 16, 128)


def test_jepa_backward_yields_finite_gradients() -> None:
    model = EEGLeJEPA(EEGLeJEPAConfig(sigreg_num_slices=32))  # small for speed
    model.train()
    x = _make_input(b=2, c=64, t=800)
    out = model(x)
    out["total_loss"].backward()

    n_with_grad = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += 1
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient on {name}."
        if p.grad.abs().sum() > 0:
            n_with_grad += 1
    # Most parameters should receive non-zero gradients on a single backward pass.
    assert n_with_grad >= int(0.8 * n_total), (
        f"Only {n_with_grad}/{n_total} params got non-zero grads — likely a wiring bug."
    )


def test_parameter_count_within_budget() -> None:
    """Phase 1 prototype should be ~1-5M params total."""
    model = EEGLeJEPA(EEGLeJEPAConfig())
    counts = model.num_parameters
    assert 500_000 < counts["total"] < 10_000_000, (
        f"Parameter count {counts['total']:,} is outside the documented 0.5M-10M budget."
    )


def test_jepa_encode_inference_only() -> None:
    """encode() should return just embeddings, with no_grad."""
    model = EEGLeJEPA(EEGLeJEPAConfig())
    x = _make_input(b=1, c=64, t=800)
    z = model.encode(x)
    assert z.shape == (1, 20, 192)
    assert not z.requires_grad
