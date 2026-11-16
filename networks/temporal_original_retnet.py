"""
temporal_original_retnet.py
===========================

Temporal (inter-frame) module following the ORIGINAL Retentive Network
(Sun et al., 2023), parallel representation, used as a comparison baseline
for the proposed bidirectional retention encoder.

Differences from the proposed module (everything else is kept identical):
    1. no softmax: the decay-weighted retention scores are used directly
    2. causal decay: each frame only attends to itself and preceding frames
    3. feed-forward network: original RetNet FFN, gelu(X W_1) W_2, hidden size 2d

Shared with the proposed module (to isolate the effect of 1-3):
    pre-norm blocks, RoPE on Q/K, fixed multi-scale decay rates
    gamma_h = 1 - 2^(-5-h), per-head GroupNorm, swish output gate, W_O,
    value dimension d_v = d (the original RetNet uses d_v = 2d).

Multi-scale retention, for each head h (parallel representation):
    Q = RoPE(X W_Q),  K = RoPE(X W_K) * d_h^(-1/2),  V = X W_V
    D_h[n, m] = gamma_h^(n - m)  if n >= m,  0 otherwise
    S_h = (Q_h K_h^T) . D_h
    S_h = S_h / max(sum_m |S_h[n, m]|, 1)          (stability normalization,
                                                     official implementation)
    head_h = S_h V_h
    MSR(X) = ( swish(X W_G) . GroupNorm(Concat(head_1..head_H)) ) W_O

Block (pre-norm):
    Y  = MSR(LN(X)) + X
    X' = FFN(LN(Y)) + Y,   FFN(X) = gelu(X W_1) W_2

Input / output: (batch, seq_len, dim) -> (batch, seq_len, dim)
Drop-in replacement for BidirectionalRetentionEncoder.
"""
from __future__ import annotations   # allows "int | None" on Python 3.8 (JetPack 5)

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotary position embedding (same as in the proposed module)
# --------------------------------------------------------------------------- #
class RotaryPositionEmbedding(nn.Module):
    """RoPE form of RetNet's xPos rotation  Q_n e^{i n theta},  K_m e^{-i m theta}."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for rotary embedding"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)      # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)                # (seq_len, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq_len, head_dim); cos, sin: (seq_len, head_dim)
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


# --------------------------------------------------------------------------- #
# Original (causal, softmax-free) multi-scale retention
# --------------------------------------------------------------------------- #
class OriginalMultiScaleRetention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.use_rotary = use_rotary

        # bias-free projections, d_v = d (same sizes as the proposed module)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)    # swish gate
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # per-head normalization (one group per head)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim)
        self.dropout = nn.Dropout(dropout)

        if use_rotary:
            self.rotary = RotaryPositionEmbedding(self.head_dim)

        # fixed multi-scale decay rates, gamma_h = 1 - 2^(-5-h), h = 0..H-1
        gammas = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())
        self.register_buffer("gammas", gammas)

    def _causal_decay(self, seq_len: int, device, dtype) -> torch.Tensor:
        """D[h, n, m] = gamma_h^(n-m) for n >= m, 0 otherwise.  Shape (H, L, L)."""
        idx = torch.arange(seq_len, device=device)
        n_minus_m = (idx[:, None] - idx[None, :]).to(dtype)               # (L, L)
        causal = (n_minus_m >= 0).to(dtype)
        log_gamma = torch.log(self.gammas.to(device=device, dtype=dtype))  # (H,)
        # clamp the exponent of future positions to 0 before exp to avoid overflow;
        # they are zeroed by the causal mask anyway
        decay = torch.exp(log_gamma[:, None, None] * n_minus_m.clamp(min=0)[None])
        return decay * causal[None]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:         (batch, seq_len, dim)
        attn_mask: optional (batch, seq_len), 1 = valid frame, 0 = padding
        returns    (batch, seq_len, dim)
        """
        b, l, d = x.shape
        h, hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(b, l, h, hd).transpose(1, 2)    # (b, h, l, hd)
        k = self.k_proj(x).view(b, l, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, l, h, hd).transpose(1, 2)

        if self.use_rotary:
            cos, sin = self.rotary(l, x.device, x.dtype)
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)
        k = k * self.scaling                                     # key scaling

        scores = torch.matmul(q, k.transpose(-1, -2))            # (b, h, l, l)
        scores = scores * self._causal_decay(l, x.device, x.dtype)[None]   # no softmax

        if attn_mask is not None:                                # zero padded keys
            scores = scores * attn_mask[:, None, None, :].to(scores.dtype)

        # stability normalization of the official implementation
        scores = scores / scores.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        scores = self.dropout(scores)

        out = torch.matmul(scores, v)                            # (b, h, l, hd)
        out = out.transpose(1, 2).reshape(b, l, d)               # (b, l, d)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        out = F.silu(self.g_proj(x)) * out                       # swish gate
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
# Original RetNet feed-forward network: gelu(X W_1) W_2, hidden size 2d
# --------------------------------------------------------------------------- #
class RetNetFeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 2 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Block and encoder
# --------------------------------------------------------------------------- #
class OriginalRetNetBlock(nn.Module):
    """Pre-norm block:  Y = MSR(LN(X)) + X,   X' = FFN(LN(Y)) + Y."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_rotary: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.retention = OriginalMultiScaleRetention(dim, num_heads, dropout, use_rotary)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = RetNetFeedForward(dim, ffn_hidden_dim, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.retention(self.norm1(x), attn_mask=attn_mask) + x
        return self.ffn(self.norm2(y)) + y


class OriginalRetNetEncoder(nn.Module):
    """
    Stack of L original RetNet blocks.  (batch, seq_len, dim) -> (batch, seq_len, dim)

    Same constructor arguments as BidirectionalRetentionEncoder, so it can replace
    it directly; `causal` and `ffn_type` are accepted for compatibility but ignored,
    since this module is always causal and always uses the RetNet GELU FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_rotary: bool = True,
        causal: bool = True,      # ignored (always causal)
        ffn_type: str = "gelu",   # ignored (always RetNet FFN)
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                OriginalRetNetBlock(dim, num_heads, ffn_hidden_dim, dropout, use_rotary)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x
