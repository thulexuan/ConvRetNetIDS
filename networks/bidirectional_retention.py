"""
bidirectional_retention.py
===========================

Bidirectional Multi-Scale Retention block, adapted from:

  - Sun et al., "Retentive Network: A Successor to Transformer for Large
    Language Models" (RetNet), arXiv:2307.08621
  - Wang et al., "Convolutional Retentive Network for EEG Decoding"
    (RetEEG), ICASSP 2025

Design choices (matching RetEEG's non-causal variant, not RetNet's
original causal/recurrent variant):

  1. Bidirectional decay matrix:
         D_B[h, n, m] = gamma_h ** |n - m|
     instead of RetNet's causal (n >= m) triangular decay. Every token
     attends to both past and future tokens in the sequence, with the
     attention weight decaying exponentially with temporal distance.

  2. Softmax is kept (unlike RetNet's linear "(QK^T . D) V"):
         head_h = ( softmax(Q_h K_h^T)  .  D_B_h ) V_h
     Softmax is applied to the raw attention scores first; the decay
     matrix is then applied as an element-wise multiplicative prior.
     This is only valid/useful because we do NOT need the RNN-form
     dual representation that RetNet relies on for O(1) autoregressive
     decoding -- this block is meant for offline/batch classification
     of a fixed-length window (e.g. a window of CAN bus frames), not
     streaming next-token generation. Dropping softmax is what breaks
     the recurrent form; keeping it here costs nothing because we never
     need that recurrent form in the first place.

  3. Multi-scale decay: each head has its own FIXED gamma_h so that
     different heads specialize in different temporal ranges (short
     bursts vs. long-range dependencies) -- useful for CAN IDS, where
     e.g. DoS-style attacks have very short-range temporal signatures
     while spoofing/replay attacks can have longer-range signatures.

  4. Swish-gated output projection + GroupNorm, following RetNet's
     multi-scale retention (MSR) layer design.

  5. Rotary position embedding (RoPE) is used to inject *relative*
     positional information into Q/K, similar in spirit to RetNet's
     xPos rotation (Q_n e^{i n theta}, K_m e^{i m theta}), implemented
     here in the standard real-valued "rotate-half" form.

Input / output shape convention (as requested):
    x: (batch_size, seq_len, token_dim)
    e.g. (128, 32, 128)  ->  output: (128, 32, 128)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotary position embedding (relative position prior for Q/K)
# --------------------------------------------------------------------------- #
class RotaryPositionEmbedding(nn.Module):
    """
    Standard RoPE, used to inject relative positional information into
    Q and K before the retention product, in the spirit of RetNet's
    xPos rotation Q_n e^{i n theta}, K_m e^{i m theta}.
    """

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
    # x:        (batch, heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Bidirectional multi-scale retention (softmax kept)
# --------------------------------------------------------------------------- #
class BidirectionalMultiScaleRetention(nn.Module):
    """
    Bidirectional Multi-Scale Retention layer.

        head_h = ( softmax(Q_h K_h^T / sqrt(d_h)) . D_B_h ) V_h
        D_B_h[n, m] = gamma_h ** |n - m|,   gamma_h = 1 - 2 ** (-5 - h)

        MSR(X) = swish(X W_G)  .  Concat(head_1, ..., head_H) W_O
                 (with GroupNorm applied to the concatenated heads)

    Args:
        dim:        token embedding dimension (token_dim)
        num_heads:  number of retention heads
        dropout:    dropout applied to attention weights
        use_rotary: whether to apply rotary position embedding to Q/K
    """

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
        self.use_rotary = use_rotary

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # swish gate, as in RetNet's MSR: swish(X W_G) . Y
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # one group per head -> each head's output channels are
        # normalized separately, following RetNet/RetEEG (SubLN-style).
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim)
        self.dropout = nn.Dropout(dropout)

        if use_rotary:
            self.rotary = RotaryPositionEmbedding(self.head_dim)

        # Fixed (non-learnable) multi-scale decay rates: gamma_h = 1 - 2^(-5-h)
        gammas = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())
        self.register_buffer("gammas", gammas)  # shape: (num_heads,)

    def _bidirectional_decay(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build D_B for every head: D_B[h, n, m] = gamma_h ** |n - m|.
        Returns shape (num_heads, seq_len, seq_len).
        """
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype)     # (L, L)
        gammas = self.gammas.to(device=device, dtype=dtype)      # (H,)
        log_gamma = torch.log(gammas)                            # (H,)
        # gamma_h ** dist == exp(dist * log(gamma_h))
        decay = torch.exp(log_gamma[:, None, None] * dist[None, :, :])  # (H, L, L)
        return decay

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x:         (batch, seq_len, dim)
            attn_mask: optional (batch, seq_len) key-padding mask,
                       1 = valid token, 0 = padding.
        Returns:
            (batch, seq_len, dim)
        """
        b, l, d = x.shape
        h, hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(b, l, h, hd).transpose(1, 2)  # (b, h, l, hd)
        k = self.k_proj(x).view(b, l, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, l, h, hd).transpose(1, 2)

        if self.use_rotary:
            cos, sin = self.rotary(l, x.device, x.dtype)
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)  # (b,h,l,l)

        if attn_mask is not None:
            key_mask = attn_mask[:, None, None, :].to(dtype=torch.bool)  # (b,1,1,l)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (b, h, l, l) -- nonlinearity kept here

        decay = self._bidirectional_decay(l, x.device, x.dtype)  # (h, l, l)
        attn = attn * decay.unsqueeze(0)  # broadcast over batch -> (b, h, l, l)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)              # (b, h, l, hd)
        out = out.transpose(1, 2).reshape(b, l, d)  # (b, l, d)

        # GroupNorm expects (batch, channels, ...) layout.
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        gate = F.silu(self.g_proj(x))  # swish(X W_G)
        out = gate * out
        out = self.out_proj(out)
        return out


# --------------------------------------------------------------------------- #
# Feed-forward network
# --------------------------------------------------------------------------- #
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        # RetNet doubles the FFN intermediate dim (2d) to keep parameter
        # count comparable to a standard Transformer block after widening V.
        hidden_dim = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# One retention block (pre-LN MSR + pre-LN FFN, residual connections)
# --------------------------------------------------------------------------- #
class BidirectionalRetentionBlock(nn.Module):
    """
    Y  = BidirectionalMSR(LN(X)) + X
    X' = FFN(LN(Y)) + Y
    """

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
        self.retention = BidirectionalMultiScaleRetention(
            dim=dim, num_heads=num_heads, dropout=dropout, use_rotary=use_rotary
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = self.retention(self.norm1(x), attn_mask=attn_mask) + x
        out = self.ffn(self.norm2(y)) + y
        return out


# --------------------------------------------------------------------------- #
# Stack of blocks -- drop-in temporal encoder, e.g. after a CNN front-end
# in a CAN-bus intrusion detection pipeline.
# --------------------------------------------------------------------------- #
class BidirectionalRetentionEncoder(nn.Module):
    """
    Stack of BidirectionalRetentionBlock layers.

    Intended usage in an in-vehicle IDS pipeline:

        raw CAN window -> Conv front-end -> (batch, seq_len, token_dim)
                        -> BidirectionalRetentionEncoder (this module)
                        -> pooling -> FC classifier -> attack class logits
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_rotary: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BidirectionalRetentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                    use_rotary=use_rotary,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x



