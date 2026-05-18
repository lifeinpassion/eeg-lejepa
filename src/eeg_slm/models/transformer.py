"""Reusable transformer building blocks.

A small, dependency-free pre-norm Transformer block with optional causal masking.
Used by both the encoder (non-causal) and predictor (causal) in EEGLeJEPA.

References
----------
Vaswani et al. (2017), Attention Is All You Need.
Xiong et al. (2020), On Layer Normalization in the Transformer Architecture
    (pre-norm > post-norm for training stability at small scale).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_position_embeddings(
    seq_len: int, embed_dim: int, device: torch.device | None = None
) -> Tensor:
    """Standard sinusoidal positional embeddings (Vaswani et al. 2017).

    Returns a (seq_len, embed_dim) tensor.
    """
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, embed_dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(seq_len, embed_dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with optional causal mask.

    Uses torch.nn.functional.scaled_dot_product_attention (PyTorch >= 2.0)
    which dispatches to flash-attention on CUDA and a stable kernel on MPS.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        """x: (B, N, D). Returns (B, N, D)."""
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → MHA → residual → LN → MLP → residual.

    Set `is_causal=True` when constructing the block to make every forward pass causal,
    or pass it per-call via `forward(x, is_causal=...)`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.is_causal_default = is_causal

    def forward(self, x: Tensor, is_causal: bool | None = None) -> Tensor:
        causal = self.is_causal_default if is_causal is None else is_causal
        x = x + self.attn(self.norm1(x), is_causal=causal)
        x = x + self.mlp(self.norm2(x))
        return x
