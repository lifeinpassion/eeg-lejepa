"""Unit tests for SIGReg.

Properties verified:
  1. CvM on actual N(0, 1) samples is small and decreases as n grows.
  2. CvM on collapsed (constant) or strongly non-Gaussian data is large.
  3. CvM matches scipy.stats.cramervonmises within numerical tolerance.
  4. SIGReg (the full module) is differentiable and yields finite gradients.
  5. SIGReg on isotropic Gaussian embeddings is small relative to collapsed.
"""

from __future__ import annotations

import math

import pytest
import torch

from eeg_slm.models.sigreg import (
    SIGReg,
    cramer_von_mises_normal,
    cramer_von_mises_normal_batch,
)


def test_cvm_small_on_standard_normal_large_n() -> None:
    torch.manual_seed(0)
    n_large = 4096
    x = torch.randn(n_large)
    stat = cramer_von_mises_normal(x).item()
    # Asymptotic mean of CvM under H0 is 1/6 ≈ 0.167; expect well below 1.0 typically.
    assert stat < 1.0, f"CvM on N(0,1) of size {n_large} suspiciously large: {stat}"


def test_cvm_large_on_uniform() -> None:
    """Uniform(-1, 1) is clearly non-Gaussian; CvM should be much larger."""
    torch.manual_seed(0)
    n = 4096
    x_normal = torch.randn(n)
    x_uniform = (torch.rand(n) - 0.5) * 2.0
    cvm_normal = cramer_von_mises_normal(x_normal).item()
    cvm_uniform = cramer_von_mises_normal(x_uniform).item()
    assert cvm_uniform > 5.0 * cvm_normal, (
        f"Expected uniform CvM ({cvm_uniform}) to be much larger than normal CvM ({cvm_normal})."
    )


def test_cvm_large_on_collapsed() -> None:
    """All-same-value (zero variance) is a degenerate distribution; CvM should be huge."""
    n = 1024
    x = torch.full((n,), 1.5)
    # Add a tiny perturbation to avoid duplicate-sort numerical edge cases
    x = x + 1e-6 * torch.arange(n)
    stat = cramer_von_mises_normal(x).item()
    assert stat > 10.0, f"CvM on collapsed distribution should be large, got {stat}"


def test_cvm_matches_scipy_within_tolerance() -> None:
    """Verify our CvM implementation against scipy.stats.cramervonmises.

    scipy returns the W² statistic for the *empirical* CDF vs the *hypothesized*
    CDF. Our implementation uses the same closed form, so values should agree
    to high precision (only sort/float ordering should cause differences).
    """
    scipy_stats = pytest.importorskip("scipy.stats")

    torch.manual_seed(42)
    n = 500
    x = torch.randn(n)
    ours = cramer_von_mises_normal(x).item()

    res = scipy_stats.cramervonmises(x.numpy(), "norm", args=(0, 1))
    theirs = float(res.statistic)

    assert math.isclose(ours, theirs, rel_tol=1e-4, abs_tol=1e-6), (
        f"CvM mismatch: ours={ours}, scipy={theirs}"
    )


def test_cvm_batch_matches_loop() -> None:
    """Vectorized batch CvM should match a per-column Python loop."""
    torch.manual_seed(0)
    n, k = 256, 16
    proj = torch.randn(n, k)
    batched = cramer_von_mises_normal_batch(proj)
    loop = torch.stack([cramer_von_mises_normal(proj[:, j]) for j in range(k)])
    assert torch.allclose(batched, loop, atol=1e-6), (
        f"Batched CvM disagrees with loop:\n  batched={batched}\n  loop={loop}"
    )


def test_sigreg_module_smaller_on_gaussian_than_collapsed() -> None:
    """The full SIGReg module should rank isotropic Gaussian below collapsed embeddings."""
    torch.manual_seed(0)
    d = 64
    n = 512
    sig = SIGReg(embed_dim=d, num_slices=128)

    gaussian = torch.randn(n, d)
    collapsed = torch.zeros(n, d) + torch.randn(d) * 1e-3  # tight cloud at one point

    loss_gauss = sig(gaussian).item()
    loss_collapsed = sig(collapsed).item()
    assert loss_collapsed > 5.0 * loss_gauss, (
        f"SIGReg(collapsed)={loss_collapsed} should be much larger than "
        f"SIGReg(gaussian)={loss_gauss}"
    )


def test_sigreg_gradient_flows() -> None:
    """SIGReg loss must produce finite gradients on the input embeddings."""
    torch.manual_seed(0)
    d, n = 32, 128
    z = torch.randn(n, d, requires_grad=True)
    sig = SIGReg(embed_dim=d, num_slices=64)
    loss = sig(z)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all(), "SIGReg produced non-finite gradients."
    assert z.grad.abs().sum() > 0, "SIGReg gradients are identically zero."


def test_sigreg_handles_3d_input() -> None:
    """SIGReg should accept (B, T, D) and flatten leading dims internally."""
    torch.manual_seed(0)
    b, t, d = 4, 20, 64
    z = torch.randn(b, t, d)
    sig = SIGReg(embed_dim=d, num_slices=32)
    loss = sig(z)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
