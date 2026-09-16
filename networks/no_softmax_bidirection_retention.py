"""
no_softmax_bidirection_retention.py
======================================

Bidirectional Multi-Scale Retention WITHOUT softmax -- following
RetNet's ORIGINAL linear formulation, adapted to bidirectional decay
instead of RetNet's causal decay:

    Sun et al., "Retentive Network: A Successor to Transformer for
    Large Language Models" (RetNet), arXiv:2307.08621

RetNet's original retention (causal):
    Retention(X) = (Q K^T ⊙ D) V,   D[n,m] = gamma^(n-m) if n>=m else 0

This file:
    Retention(X) = (Q K^T ⊙ D) V,   D[n,m] = gamma^|n-m|   (BIDIRECTIONAL,
                                                              no causal mask
                                                              needed at all --
                                                              see note below)

Key differences from the softmax-based version used elsewhere in this
project (see prior discussion for the full derivation):

  1. NO softmax anywhere. QK^T is used directly (raw dot product),
     multiplied element-wise by the decay matrix D, with no
     normalization step in between.

  2. Consequently, attention weights here:
       - CAN be negative (softmax guarantees weights > 0; a raw dot
         product QK^T has no such guarantee)
       - are NOT bounded in any fixed range (softmax bounds weights to
         (0,1) summing to 1; here, |QK^T| can grow arbitrarily large,
         especially as head_dim grows)
       - do NOT have a probabilistic ("how much attention, as a
         percentage") interpretation
     This is why RetNet's own architecture ALWAYS follows retention
     with GroupNorm -- it is not optional here, it is the mechanism
     that keeps the otherwise-unbounded output at a stable, trainable
     scale. Removing GroupNorm from this file would very likely cause
     training instability.

  3. No causal masking is needed for the bidirectional decay matrix
     itself, unlike the causal (uni-directional) variant discussed
     earlier: that variant needed a causal mask applied to the RAW
     SCORES *before* softmax specifically to fix softmax's
     normalization (the denominator must only sum over valid
     positions). Since there is no softmax here at all, that whole
     problem does not exist -- D[n,m] = gamma^|n-m| already directly
     multiplies the correct (bidirectional, symmetric) decay into
     QK^T, with nothing left to "leak" through a normalization step
     that isn't there.

  4. This means the ONLY way this file's formula could be pushed
     towards a causal variant would be to zero out D[n,m] for n<m
     directly (exactly RetNet's original causal decay) -- there is no
     separate "mask the scores" step required, precisely because
     there is no softmax to protect. This file only implements the
     bidirectional case, per what was requested.

Padding: because there is no softmax (hence no exp(-inf)=0 trick),
padding is instead handled by directly zeroing out the decay matrix at
padded KEY positions (a batch-dependent mask multiplied into the
otherwise batch-independent, per-head positional decay).

Other design choices (kept from the softmax-based version):
  - Multi-scale decay: each head has its own FIXED gamma_h
        gamma_h = 1 - 2^(-5-h)
  - Swish-gated output projection + GroupNorm (RetNet's MSR layer design)
  - Rotary position embedding (RoPE) for relative positional information
  - FFN: GELU-MLP or SwiGLU (selectable via ffn_type)

Input / output shape convention:
    x: (batch_size, seq_len, token_dim)
    e.g. (128, 32, 128)  ->  output: (128, 32, 128)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotary position embedding (unchanged from the softmax-based version)
# --------------------------------------------------------------------------- #
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for rotary embedding"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Bidirectional multi-scale retention -- NO softmax (RetNet's original,
# linear formulation), just adapted to bidirectional decay.
# --------------------------------------------------------------------------- #
class NoSoftmaxBidirectionalRetention(nn.Module):
    """
    Multi-Scale Retention, WITHOUT softmax, supporting two decay
    directions selectable via `causal`:

        head_h = ( (Q_h K_h^T) ⊙ D_h ) V_h

        Bidirectional (causal=False): D_h[n,m] = gamma_h ** |n-m|
        Causal        (causal=True):  D_h[n,m] = gamma_h ** (n-m)  if n >= m
                                                 = 0                  if n <  m
                                       (RetNet's ORIGINAL decay)

        MSR(X) = swish(X W_G) . GroupNorm(Concat(head_1,...,head_H)) W_O

    NOTE on causal masking: unlike the softmax-based version (which
    needs an EXTRA causal mask applied to the raw scores, before
    softmax, so the softmax denominator only sums over valid
    positions -- see the earlier discussion of why "just zeroing decay
    after softmax" silently breaks normalization), no such extra step
    is needed here. Since there is no softmax at all, there is no
    normalization step that future positions could "leak" into --
    simply setting D[n,m]=0 for n<m already, on its own, guarantees
    those positions contribute EXACTLY zero to the output. This is one
    of the few places where removing softmax actually makes the causal
    case simpler, not just different.

    Args:
        dim:        token embedding dimension (token_dim)
        num_heads:  number of retention heads
        dropout:    dropout applied to the raw (Q K^T . D) weights
        use_rotary: whether to apply rotary position embedding to Q/K
        causal:     False (default) = bidirectional decay, every token
                    attends to past AND future tokens. True = original
                    RetNet-style causal decay, each token attends only
                    to itself and past tokens.
        use_scale:  whether to divide QK^T by sqrt(head_dim) before
                    multiplying by the decay matrix. NOTE: RetNet's
                    original paper formula does NOT include this
                    scaling term (it relies on GroupNorm afterward for
                    stability instead) -- this is added here, ON by
                    default, purely as an extra numerical-stability
                    measure. Set to False to match the literal paper
                    formula exactly.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
        causal: bool = False,
        use_scale: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rotary = use_rotary
        self.causal = causal
        self.use_scale = use_scale

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # swish gate, as in RetNet's MSR: swish(X W_G) . Y
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # GroupNorm is NOT optional here (see module docstring): with
        # no softmax to bound the attention weights, this is the only
        # mechanism keeping the retention output at a stable scale.
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim)
        self.dropout = nn.Dropout(dropout)

        if use_rotary:
            self.rotary = RotaryPositionEmbedding(self.head_dim)

        # Fixed (non-learnable) multi-scale decay rates: gamma_h = 1 - 2^(-5-h)
        gammas = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())
        self.register_buffer("gammas", gammas)  # shape: (num_heads,)

    def _build_decay(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build D for every head, in either bidirectional or causal mode:
            bidirectional: D[h,n,m] = gamma_h ** |n-m|
            causal:        D[h,n,m] = gamma_h ** (n-m)  if n >= m
                                    = 0                   if n <  m
        Returns (num_heads, seq_len, seq_len).
        """
        idx = torch.arange(seq_len, device=device)
        gammas = self.gammas.to(device=device, dtype=dtype)  # (H,)
        log_gamma = torch.log(gammas)  # (H,)

        if not self.causal:
            dist = (idx[:, None] - idx[None, :]).abs().to(dtype)  # (L, L)
            decay = torch.exp(log_gamma[:, None, None] * dist[None, :, :])  # (H, L, L)
            return decay
        else:
            n_minus_m = (idx[:, None] - idx[None, :]).to(dtype)  # (L, L), signed
            causal_mask = (n_minus_m >= 0).to(dtype)  # 1 where n>=m, 0 where n<m
            decay = torch.exp(log_gamma[:, None, None] * n_minus_m[None, :, :])  # (H, L, L)
            decay = decay * causal_mask[None, :, :]  # zero out n<m (future) -- exact, no softmax to leak through
            return decay

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x:         (batch, seq_len, dim)
            attn_mask: optional (batch, seq_len) key-padding mask,
                       1 = valid token, 0 = padding. Since there is no
                       softmax here, padding is handled by directly
                       zeroing the decay weight for padded KEY
                       positions (no -inf/exp trick needed or possible).
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

        # RAW dot product -- NO softmax. Can be negative, unbounded.
        scores = torch.matmul(q, k.transpose(-1, -2))  # (b, h, l, l)
        if self.use_scale:
            scores = scores / math.sqrt(hd)

        decay = self._build_decay(l, x.device, x.dtype)  # (h, l, l)
        decay = decay.unsqueeze(0)  # (1, h, l, l) -- broadcast over batch

        if attn_mask is not None:
            # zero out decay at padded KEY positions directly (no exp
            # trick available/needed without softmax)
            key_mask = attn_mask[:, None, None, :].to(dtype=x.dtype)  # (b,1,1,l)
            decay = decay * key_mask  # -> (b, h, l, l) after broadcasting

        attn = scores * decay  # (b, h, l, l) -- linear retention weights, NOT a probability
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (b, h, l, hd)
        out = out.transpose(1, 2).reshape(b, l, d)  # (b, l, d)

        # GroupNorm: NOT optional here, stabilizes the otherwise-unbounded scale.
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        gate = F.silu(self.g_proj(x))  # swish(X W_G)
        out = gate * out
        out = self.out_proj(out)
        return out


# --------------------------------------------------------------------------- #
# Feed-forward network (GELU-MLP, unchanged)
# --------------------------------------------------------------------------- #
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
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
# SwiGLU feed-forward (unchanged; Shazeer 2020, "GLU Variants Improve
# Transformer")
# --------------------------------------------------------------------------- #
class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 2 * 2 / 3)  # parameter parity with a 2x GELU-MLP
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))


# --------------------------------------------------------------------------- #
# One retention block (pre-LN retention + pre-LN FFN, residual connections)
# --------------------------------------------------------------------------- #
class NoSoftmaxBidirectionalRetentionBlock(nn.Module):
    """
    Y  = NoSoftmaxBidirectionalRetention(LN(X)) + X
    X' = FFN(LN(Y)) + Y
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_rotary: bool = True,
        causal: bool = False,
        use_scale: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        assert ffn_type in ("gelu", "swiglu"), "ffn_type must be 'gelu' or 'swiglu'"
        self.norm1 = nn.LayerNorm(dim)
        self.retention = NoSoftmaxBidirectionalRetention(
            dim=dim, num_heads=num_heads, dropout=dropout,
            use_rotary=use_rotary, causal=causal, use_scale=use_scale,
        )
        self.norm2 = nn.LayerNorm(dim)
        if ffn_type == "gelu":
            self.ffn = FeedForward(dim, hidden_dim=ffn_hidden_dim, dropout=dropout)
        else:
            self.ffn = SwiGLUFeedForward(dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = self.retention(self.norm1(x), attn_mask=attn_mask) + x
        out = self.ffn(self.norm2(y)) + y
        return out


# --------------------------------------------------------------------------- #
# Stack of blocks
# --------------------------------------------------------------------------- #
class NoSoftmaxBidirectionalRetentionEncoder(nn.Module):
    """
    Stack of NoSoftmaxBidirectionalRetentionBlock layers.

        raw CAN window -> Conv front-end -> (batch, seq_len, token_dim)
                        -> NoSoftmaxBidirectionalRetentionEncoder (this module)
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
        causal: bool = False,
        use_scale: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NoSoftmaxBidirectionalRetentionBlock(
                    dim=dim, num_heads=num_heads, ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout, use_rotary=use_rotary, causal=causal,
                    use_scale=use_scale, ffn_type=ffn_type,
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


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, seq_len, token_dim = 128, 32, 128
    num_heads = 8
    num_layers = 4

    x = torch.randn(batch_size, seq_len, token_dim)

    encoder = NoSoftmaxBidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers, dropout=0.1,
    )
    out = encoder(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == x.shape

    # padding mask
    mask = torch.ones(batch_size, seq_len)
    mask[:, 28:] = 0
    out_masked = encoder(x, attn_mask=mask)
    print(f"Masked output shape: {tuple(out_masked.shape)}")
    assert out_masked.shape == x.shape

    # gradient flow
    loss = out.sum()
    loss.backward()
    q_grad = encoder.layers[0].retention.q_proj.weight.grad
    grad_ok = q_grad is not None and torch.isfinite(q_grad).all() and q_grad.abs().sum() > 0
    print(f"q_proj gradient finite & nonzero: {grad_ok.item()}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {n_params:,}")

    # --- key sanity/demonstration checks: contrast with the softmax version ---
    print("\n=== Demonstrating the no-softmax properties (vs. softmax-based version) ===")
    retention = encoder.layers[0].retention
    with torch.no_grad():
        q = retention.q_proj(x).view(batch_size, seq_len, num_heads, retention.head_dim).transpose(1, 2)
        k = retention.k_proj(x).view(batch_size, seq_len, num_heads, retention.head_dim).transpose(1, 2)
        if retention.use_rotary:
            cos, sin = retention.rotary(seq_len, x.device, x.dtype)
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)
        raw_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(retention.head_dim)
        decay = retention._build_decay(seq_len, x.device, x.dtype).unsqueeze(0)
        attn = raw_scores * decay

    frac_negative = (attn < 0).float().mean().item()
    print(f"Fraction of attention weights that are NEGATIVE: {frac_negative:.2%} "
          f"(softmax-based version: always exactly 0%)")
    assert frac_negative > 0.0, "expected some negative weights without softmax"

    row_sums = attn.sum(dim=-1)
    print(f"Row sums (should sum to 1 for softmax-based version): "
          f"min={row_sums.min().item():.4f}, max={row_sums.max().item():.4f} "
          f"(here: unconstrained, NOT expected to be 1)")

    print(f"\nGroupNorm is applied unconditionally in forward() -- required for stability "
          f"since these unbounded, possibly-negative weights are not self-normalizing "
          f"the way softmax weights are.")

    # --- SwiGLU FFN variant ---
    encoder_swiglu = NoSoftmaxBidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=1, ffn_type="swiglu",
    )
    out_swiglu = encoder_swiglu(x)
    print(f"\n[SwiGLU FFN] Output shape: {tuple(out_swiglu.shape)}")
    assert out_swiglu.shape == x.shape

    # --- use_scale=False, matching the literal RetNet paper formula exactly ---
    encoder_no_scale = NoSoftmaxBidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=1, use_scale=False,
    )
    out_no_scale = encoder_no_scale(x)
    print(f"[use_scale=False, literal paper formula] Output shape: {tuple(out_no_scale.shape)}")
    assert out_no_scale.shape == x.shape

    # -------------------------------------------------------------- #
    # Causal (uni-directional, original RetNet) vs Bidirectional
    # -------------------------------------------------------------- #
    print("\n=== Causal (uni-directional) vs Bidirectional decay ===")
    encoder_causal = NoSoftmaxBidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers,
        dropout=0.1, causal=True,
    )
    out_causal = encoder_causal(x)
    print(f"[Causal] Output shape: {tuple(out_causal.shape)}")
    assert out_causal.shape == x.shape

    loss = out_causal.sum()
    loss.backward()
    q_grad_causal = encoder_causal.layers[0].retention.q_proj.weight.grad
    grad_ok = q_grad_causal is not None and torch.isfinite(q_grad_causal).all() \
        and q_grad_causal.abs().sum() > 0
    print(f"[Causal] q_proj gradient finite & nonzero: {grad_ok.item()}")

    # Unlike the softmax-based version (where we had to inspect the raw
    # attention weight matrix to avoid a misleading test at random
    # initialization -- see prior discussion), here we can check the
    # DECAY MATRIX directly: D[n,m]=0 for n<m is guaranteed by
    # construction, completely independent of Q/K, so multiplying it
    # into ANY scores (random or trained) gives EXACTLY zero at those
    # positions -- no ambiguity possible.
    causal_decay = encoder_causal.layers[0].retention._build_decay(seq_len, x.device, x.dtype)
    bidir_decay = encoder.layers[0].retention._build_decay(seq_len, x.device, x.dtype)

    print(f"\n[Causal decay]        D[5, 20] (query=5, FUTURE key=20): "
          f"{causal_decay[0, 5, 20].item():.2e}  (must be exactly 0)")
    print(f"[Bidirectional decay] D[5, 20] (query=5, key=20, allowed nonzero): "
          f"{bidir_decay[0, 5, 20].item():.4f}")
    assert causal_decay[:, 5, 20].abs().max().item() == 0.0, (
        "causal decay must be EXACTLY zero for future positions (n<m)"
    )
    assert bidir_decay[0, 5, 20].item() > 0.0
    print("PASSED: causal decay is exactly 0 for future positions (by construction, "
          "no dependence on Q/K at all); bidirectional decay is nonzero, as expected.")

    n_params_causal = sum(p.numel() for p in encoder_causal.parameters())
    print(f"\n[Causal] Total parameters: {n_params_causal:,}")
