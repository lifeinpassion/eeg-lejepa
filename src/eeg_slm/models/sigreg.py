"""SIGReg — Sketched Isotropic Gaussian Regularization.

Forces the distribution of a batch of embeddings to be isotropic N(0, I) via
the Cramér-Wold theorem: if every 1-D random projection of a high-dimensional
distribution is N(0, 1), then the high-dimensional distribution is N(0, I).

We sample `num_slices` random unit vectors on the sphere S^{D-1}, project the
embeddings onto each, and measure the deviation of each 1-D projection from
N(0, 1) using a univariate goodness-of-fit test. The total loss is the mean
over slices.

For the univariate test we implement Cramér-von Mises (CvM):

    W² = 1/(12n) + Σ_{i=1}^n [F(x_(i)) - (2i - 1)/(2n)]²

where x_(i) is the i-th order statistic and F is the standard normal CDF.
CvM is closed-form, differentiable through `torch.sort` and `torch.erf`, and
verifiable against `scipy.stats.cramervonmises`.

The paper recommends Epps-Pulley (characteristic-function-based). We default to
CvM for clarity; Epps-Pulley is a clean follow-up.

References
----------
Balestriero & LeCun (2025), LeJEPA. arXiv:2511.08544.
Cramér (1928), On the composition of elementary errors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

_SQRT_2 = math.sqrt(2.0)


def _standard_normal_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF via erf. Differentiable everywhere."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT_2))


def cramer_von_mises_normal(x: Tensor) -> Tensor:
    """Cramér-von Mises statistic against the standard normal N(0, 1).

    Parameters
    ----------
    x : Tensor of shape (n,) — one 1-D sample of size n.

    Returns
    -------
    scalar Tensor — the CvM statistic W². Smaller = closer to N(0, 1).
    """
    if x.dim() != 1:
        raise ValueError(f"Expected 1-D tensor, got shape {tuple(x.shape)}.")
    n = x.shape[0]
    sorted_x, _ = torch.sort(x)
    i = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    cdf = _standard_normal_cdf(sorted_x)
    return 1.0 / (12.0 * n) + ((cdf - (2.0 * i - 1.0) / (2.0 * n)) ** 2).sum()


def cramer_von_mises_normal_batch(projections: Tensor) -> Tensor:
    """Vectorized CvM applied independently to each column.

    Parameters
    ----------
    projections : Tensor of shape (n, k) — n samples × k independent 1-D tests.

    Returns
    -------
    Tensor of shape (k,) — CvM statistic per column.
    """
    if projections.dim() != 2:
        raise ValueError(f"Expected (n, k) tensor, got shape {tuple(projections.shape)}.")
    n, _ = projections.shape
    sorted_proj, _ = torch.sort(projections, dim=0)  # sort each column independently
    i = torch.arange(
        1, n + 1, dtype=projections.dtype, device=projections.device
    ).unsqueeze(-1)  # (n, 1) broadcasts over k
    cdf = _standard_normal_cdf(sorted_proj)  # (n, k)
    return 1.0 / (12.0 * n) + ((cdf - (2.0 * i - 1.0) / (2.0 * n)) ** 2).sum(dim=0)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization loss module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality D of the embeddings to regularize.
    num_slices : int
        Number of random 1-D projections per forward pass (default 1024).
        Paper notes this is not a sensitive hyperparameter — anywhere from
        512 to 2048 works well. We default to 256 for M1 dev speed; bump to
        1024 on AutoDL for production training.
    test : {"cvm"}
        Which univariate Gaussianity test to apply on each slice. Currently
        only Cramér-von Mises is implemented; Epps-Pulley is a planned addition.
    resample_each_call : bool
        If True (default), draw fresh random projections on every forward pass.
        Matches the paper's stochastic slicing. If False, reuse a single fixed
        set of projections (useful for debugging or reproducibility experiments).

    Notes
    -----
    The minimum achievable value of CvM under perfect N(0, 1) is 1/(12n) plus
    finite-sample sampling noise; for n=256 that floor is ~3.3e-4 (per slice).
    The loss is therefore never exactly zero — what matters is whether it
    *decreases* with training.
    """

    def __init__(
        self,
        embed_dim: int,
        num_slices: int = 256,
        test: str = "cvm",
        resample_each_call: bool = True,
    ) -> None:
        super().__init__()
        if test != "cvm":
            raise NotImplementedError(f"Test '{test}' not implemented. Only 'cvm' is available.")
        self.embed_dim = embed_dim
        self.num_slices = num_slices
        self.test = test
        self.resample_each_call = resample_each_call

        if not resample_each_call:
            # Fixed projections — registered as buffer so device-moves work.
            self.register_buffer("_fixed_directions", self._sample_directions(embed_dim, num_slices))
        else:
            self._fixed_directions = None  # type: ignore[assignment]

    @staticmethod
    def _sample_directions(embed_dim: int, num_slices: int, device=None, dtype=None) -> Tensor:
        """Sample `num_slices` unit vectors uniformly on the sphere S^{D-1}.

        Uses the standard trick: draw from N(0, I), normalize to unit length.
        """
        directions = torch.randn(embed_dim, num_slices, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
        return directions

    def forward(self, embeddings: Tensor) -> Tensor:
        """Compute SIGReg loss on a batch of embeddings.

        Parameters
        ----------
        embeddings : Tensor of shape (..., D)
            Any leading dims are flattened to a single batch dim before projection.
            Typically (B, T, D) from an encoder, flattened to (B*T, D).

        Returns
        -------
        scalar Tensor — mean univariate-test statistic across all slices.
        """
        if embeddings.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embed_dim}, got {embeddings.shape[-1]}."
            )
        flat = embeddings.reshape(-1, self.embed_dim)  # (N, D)

        if self.resample_each_call:
            directions = self._sample_directions(
                self.embed_dim, self.num_slices, device=flat.device, dtype=flat.dtype
            )
        else:
            directions = self._fixed_directions.to(dtype=flat.dtype)  # type: ignore[union-attr]

        projections = flat @ directions  # (N, K)
        per_slice = cramer_von_mises_normal_batch(projections)  # (K,)
        return per_slice.mean()
