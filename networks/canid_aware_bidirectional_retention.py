"""
bidirectional_retention.py
===========================

Bidirectional Multi-Scale Retention block, adapted from:

  - Sun et al., "Retentive Network: A Successor to Transformer for Large
    Language Models" (RetNet), arXiv:2307.08621
  - Wang et al., "Convolutional Retentive Network for EEG Decoding"
    (RetEEG), ICASSP 2025

Design choices:

  1. Bidirectional decay matrix instead of RetNet's causal (n >= m)
     triangular decay. Every token attends to both past and future
     tokens in the sequence.

  2. Softmax is kept (unlike RetNet's linear "(QK^T . D) V"): this is
     only valid/useful because we do NOT need the RNN-form dual
     representation that RetNet relies on for O(1) autoregressive
     decoding -- this block is meant for offline/batch classification
     of a fixed-length window (e.g. a window of CAN bus frames), not
     streaming next-token generation.

  3. Multi-scale decay: each head has its own FIXED gamma_h so that
     different heads specialize in different temporal ranges (short
     bursts vs. long-range dependencies).

  4. NEW -- Same-ID / diff-ID decay prior (FIXED, not learnable):
     on a shared CAN bus, two frames with the SAME CAN ID form a real,
     meaningful time series (e.g. periodic signal drift, spoofing),
     even if far apart in the window because other IDs were
     interleaved in between. Two frames with DIFFERENT CAN IDs that
     happen to sit near each other are adjacent purely by coincidence
     of bus arbitration. Plain distance-based decay treats both cases
     identically; this prior lets same-ID pairs decay more slowly than
     diff-ID pairs, at the SAME positional distance:

         D_h[n,m] = gamma_same_h ^ |n-m|   if id[n] == id[m]
                  = gamma_diff_h ^ |n-m|   otherwise

     gamma_diff_h follows RetNet's original schedule (1 - 2^(-5-h));
     gamma_same_h is derived from it by shifting the exponent by a
     fixed constant `id_decay_delta > 0`:

         gamma_diff_h = 1 - 2^(-5-h)
         gamma_same_h = 1 - 2^(-5-h-delta)

     Since -5-h-delta < -5-h, this guarantees gamma_same_h > gamma_diff_h
     structurally, for any delta > 0 -- no separate constraint needed.
     Both are registered as buffers (non-learnable), matching RetNet's
     original choice of fixing the decay schedule rather than learning
     it.

  5. Swish-gated output projection + GroupNorm, following RetNet's
     multi-scale retention (MSR) layer design.

  6. Rotary position embedding (RoPE) is used to inject *relative*
     positional information into Q/K.

Input / output shape convention:
    x:       (batch_size, seq_len, token_dim)
    can_ids: (batch_size, seq_len)  int64, optional -- required only
             when use_id_aware_decay=True.
    output:  (batch_size, seq_len, token_dim)
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

        head_h = ( softmax(Q_h K_h^T / sqrt(d_h)) . D_h ) V_h

        MSR(X) = swish(X W_G)  .  Concat(head_1, ..., head_H) W_O
                 (with GroupNorm applied to the concatenated heads)

    Two decay modes are supported:

    (a) Position-only decay (default, use_id_aware_decay=False):
            D_h[n, m] = gamma_h ** |n - m|,  gamma_h = 1 - 2^(-5-h)  (FIXED)

    (b) Same-ID / diff-ID decay (use_id_aware_decay=True, requires
        `can_ids` at forward time), BOTH FIXED (non-learnable):
            D_h[n, m] = gamma_same_h ** |n-m|   if can_id[n] == can_id[m]
                      = gamma_diff_h ** |n-m|   otherwise
            gamma_diff_h = 1 - 2^(-5-h)
            gamma_same_h = 1 - 2^(-5-h-id_decay_delta)

    Args:
        dim:              token embedding dimension (token_dim)
        num_heads:        number of retention heads
        dropout:          dropout applied to attention weights
        use_rotary:       whether to apply rotary position embedding to Q/K
        use_id_aware_decay: whether to use the fixed same-ID / diff-ID
                          decay scheme described above
        id_decay_delta:   exponent shift controlling how much slower
                          same-ID pairs decay relative to diff-ID pairs
                          (gamma_same = 1 - 2^(-5-h-delta)). Larger
                          delta => same-ID pairs remembered much longer.
                          Typical range: 1 (mild) to 5 (strong).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_id_aware_decay: bool = False,
        id_decay_delta: float = 3.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rotary = use_rotary
        self.use_id_aware_decay = use_id_aware_decay

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

        h_idx = torch.arange(num_heads, dtype=torch.float32)
        # gamma_diff_h = 1 - 2^(-5-h)  -- RetNet's original fixed schedule
        gamma_diff = 1.0 - torch.pow(2.0, -5.0 - h_idx)

        if not use_id_aware_decay:
            # Position-only decay: single fixed gamma per head.
            self.register_buffer("gammas", gamma_diff)  # (num_heads,)
        else:
            # Same-ID / diff-ID decay: gamma_diff as above, gamma_same
            # derived by shifting the exponent by `id_decay_delta`, so
            # gamma_same_h > gamma_diff_h holds structurally for any
            # delta > 0 -- no extra constraint needed. Both FIXED (buffers).
            gamma_same = 1.0 - torch.pow(2.0, -5.0 - h_idx - id_decay_delta)
            self.register_buffer("gamma_diff", gamma_diff)  # (num_heads,)
            self.register_buffer("gamma_same", gamma_same)  # (num_heads,)

    def _position_only_decay(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """D_h[n, m] = gamma_h ** |n - m|.  Shape: (1, num_heads, L, L)."""
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype)      # (L, L)
        gammas = self.gammas.to(device=device, dtype=dtype)       # (H,)
        log_gamma = torch.log(gammas)                             # (H,)
        decay = torch.exp(log_gamma[:, None, None] * dist[None, :, :])  # (H, L, L)
        return decay.unsqueeze(0)  # (1, H, L, L) -- broadcasts over batch

    def _id_aware_decay(
        self,
        seq_len: int,
        can_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        D_h[n, m] = gamma_same_h ** |n-m|  if can_id[n] == can_id[m]
                  = gamma_diff_h ** |n-m|  otherwise
        Shape: (batch, num_heads, L, L) -- depends on the batch because
        CAN ID layout differs per sample.
        """
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype)  # (L, L)

        gamma_diff = self.gamma_diff.to(device=device, dtype=dtype)  # (H,)
        gamma_same = self.gamma_same.to(device=device, dtype=dtype)  # (H,)

        log_diff = torch.log(gamma_diff)  # (H,)
        log_same = torch.log(gamma_same)  # (H,)

        decay_diff = torch.exp(log_diff[:, None, None] * dist[None, :, :])  # (H, L, L)
        decay_same = torch.exp(log_same[:, None, None] * dist[None, :, :])  # (H, L, L)

        # same_id_mask[b, n, m] = 1 if can_ids[b, n] == can_ids[b, m] else 0
        same_id_mask = (can_ids[:, :, None] == can_ids[:, None, :]).to(dtype)  # (B, L, L)

        decay = (
            same_id_mask[:, None, :, :] * decay_same[None, :, :, :]
            + (1.0 - same_id_mask[:, None, :, :]) * decay_diff[None, :, :, :]
        )  # (B, H, L, L)
        return decay

    def get_decay_rates(self) -> dict:
        """Utility for inspection/logging: returns the fixed gamma(s) per
        head as plain floats."""
        if not self.use_id_aware_decay:
            return {"gamma": self.gammas.tolist()}
        return {
            "gamma_diff": self.gamma_diff.tolist(),
            "gamma_same": self.gamma_same.tolist(),
        }

    def forward(
        self,
        x: torch.Tensor,
        can_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (batch, seq_len, dim)
            can_ids:   (batch, seq_len) int tensor of CAN IDs, required
                       if use_id_aware_decay=True, ignored otherwise.
            attn_mask: optional (batch, seq_len) key-padding mask,
                       1 = valid token, 0 = padding.
        Returns:
            (batch, seq_len, dim)
        """
        b, l, d = x.shape
        h, hd = self.num_heads, self.head_dim

        if self.use_id_aware_decay and can_ids is None:
            raise ValueError(
                "use_id_aware_decay=True requires `can_ids` (batch, seq_len) "
                "to be passed to forward()."
            )
        if self.use_id_aware_decay and can_ids is not None:
            if can_ids.shape[0] != b or can_ids.shape[1] != l:
                raise ValueError(
                    f"Shape mismatch between `x` and `can_ids`: x is "
                    f"(batch={b}, seq_len={l}, dim={d}) but can_ids is "
                    f"{tuple(can_ids.shape)} (expected (batch={b}, seq_len={l}))."
                )

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

        if self.use_id_aware_decay:
            decay = self._id_aware_decay(l, can_ids, x.device, x.dtype)  # (b, h, l, l)
        else:
            decay = self._position_only_decay(l, x.device, x.dtype)      # (1, h, l, l)

        attn = attn * decay  # broadcasts (1,h,l,l) or matches (b,h,l,l)
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
    """Original GELU-MLP feed-forward: Linear -> GELU -> Dropout -> Linear."""

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
# SwiGLU feed-forward (gated FFN, as used in LLaMA/PaLM/Mistral)
# --------------------------------------------------------------------------- #
class SwiGLUFeedForward(nn.Module):
    """
    Gated feed-forward: two parallel projections instead of one.
        h = SiLU(x @ W_gate) * (x @ W_up); out = h @ W_down
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            base_hidden_dim = dim * 2
            hidden_dim = int(base_hidden_dim * 2 / 3)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))


# --------------------------------------------------------------------------- #
# SGLU feed-forward (CellFM-style Simple Gated Linear Unit, no activation)
# --------------------------------------------------------------------------- #
class SGLUFeedForward(nn.Module):
    """
    Simple Gated Linear Unit (CellFM-style): out = (x@W_u . x@W_v) @ W_o,
    with NO activation function -- the multiplicative gating alone is
    treated as sufficient nonlinearity, trading a bit of representational
    power for faster computation.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            base_hidden_dim = dim * 2
            hidden_dim = int(base_hidden_dim * 2 / 3)
        self.w_u = nn.Linear(dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(dim, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.w_u(x) * self.w_v(x)
        return self.w_o(self.dropout(gated))


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
        use_id_aware_decay: bool = False,
        id_decay_delta: float = 3.0,
        ffn_type: str = "gelu",
    ):
        super().__init__()
        assert ffn_type in ("gelu", "swiglu", "sglu"), (
            "ffn_type must be 'gelu', 'swiglu', or 'sglu'"
        )
        self.norm1 = nn.LayerNorm(dim)
        self.retention = BidirectionalMultiScaleRetention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            use_id_aware_decay=use_id_aware_decay,
            id_decay_delta=id_decay_delta,
        )
        self.norm2 = nn.LayerNorm(dim)
        if ffn_type == "gelu":
            self.ffn = FeedForward(dim, hidden_dim=ffn_hidden_dim, dropout=dropout)
        elif ffn_type == "swiglu":
            self.ffn = SwiGLUFeedForward(dim, hidden_dim=ffn_hidden_dim, dropout=dropout)
        else:  # "sglu"
            self.ffn = SGLUFeedForward(dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        can_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.retention(self.norm1(x), can_ids=can_ids, attn_mask=attn_mask) + x
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

        raw CAN window -> local encoder -> (batch, seq_len, token_dim)
                        -> BidirectionalRetentionEncoder (this module)
                        -> pooling -> classifier -> attack class logits

    If use_id_aware_decay=True, the CAN ID of each frame in the window
    must be passed as `can_ids` (batch, seq_len) at forward time.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_id_aware_decay: bool = False,
        id_decay_delta: float = 3.0,
        ffn_type: str = "swiglu",
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
                    use_id_aware_decay=use_id_aware_decay,
                    id_decay_delta=id_decay_delta,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        can_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, can_ids=can_ids, attn_mask=attn_mask)
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

    # -------------------------------------------------------------- #
    # 1) Baseline: position-only decay (no ID awareness)
    # -------------------------------------------------------------- #
    encoder = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers, dropout=0.1,
    )
    out = encoder(x)
    print("[Position-only decay]")
    print(f"  Output shape: {tuple(out.shape)}")
    assert out.shape == x.shape

    mask = torch.ones(batch_size, seq_len)
    mask[:, 20:] = 0
    out_masked = encoder(x, attn_mask=mask)
    print(f"  Masked output shape: {tuple(out_masked.shape)}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {n_params:,}")

    # -------------------------------------------------------------- #
    # 2) NEW: Fixed same-ID / diff-ID decay
    # -------------------------------------------------------------- #
    num_can_ids = 20
    can_ids = torch.randint(0, num_can_ids, (batch_size, seq_len))

    id_aware_encoder = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers, dropout=0.1,
        use_id_aware_decay=True, id_decay_delta=3.0,
    )
    out_id_aware = id_aware_encoder(x, can_ids=can_ids)
    print("\n[Fixed same-ID / diff-ID decay]")
    print(f"  can_ids shape: {tuple(can_ids.shape)}")
    print(f"  Output shape:  {tuple(out_id_aware.shape)}")
    assert out_id_aware.shape == x.shape

    rates = id_aware_encoder.layers[0].retention.get_decay_rates()
    print(f"  gamma_diff (per head): {[f'{v:.6f}' for v in rates['gamma_diff']]}")
    print(f"  gamma_same (per head): {[f'{v:.6f}' for v in rates['gamma_same']]}")
    assert all(s > d for s, d in zip(rates["gamma_same"], rates["gamma_diff"])), (
        "gamma_same must always be > gamma_diff"
    )

    # confirm these are FIXED buffers, not learnable parameters
    retention0 = id_aware_encoder.layers[0].retention
    is_buffer = "gamma_same" in dict(retention0.named_buffers())
    is_param = "gamma_same" in dict(retention0.named_parameters())
    print(f"  gamma_same is a fixed buffer (not learnable): {is_buffer and not is_param}")

    # padding mask still works together with id-aware decay
    out_id_masked = id_aware_encoder(x, can_ids=can_ids, attn_mask=mask)
    print(f"  [+ padding mask] Output shape: {tuple(out_id_masked.shape)}")

    n_params_id = sum(p.numel() for p in id_aware_encoder.parameters())
    print(f"  Total parameters: {n_params_id:,} (same as baseline: {n_params_id == n_params})")

    # -------------------------------------------------------------- #
    # 3) Try a couple of id_decay_delta values
    # -------------------------------------------------------------- #
    print("\n[Effect of id_decay_delta on gamma_same, head 0]")
    for delta in [1.0, 3.0, 5.0]:
        enc = BidirectionalRetentionEncoder(
            dim=token_dim, num_heads=num_heads, num_layers=1,
            use_id_aware_decay=True, id_decay_delta=delta,
        )
        r = enc.layers[0].retention.get_decay_rates()
        print(f"  delta={delta}: gamma_diff[0]={r['gamma_diff'][0]:.6f}, "
              f"gamma_same[0]={r['gamma_same'][0]:.6f}")

    # -------------------------------------------------------------- #
    # 4) FFN variants still work unchanged (gelu / swiglu / sglu)
    # -------------------------------------------------------------- #
    for ffn_type in ("gelu", "swiglu", "sglu"):
        enc = BidirectionalRetentionEncoder(
            dim=token_dim, num_heads=num_heads, num_layers=num_layers,
            dropout=0.1, ffn_type=ffn_type,
            use_id_aware_decay=True, id_decay_delta=3.0,
        )
        out_ffn = enc(x, can_ids=can_ids)
        assert out_ffn.shape == x.shape
        n_p = sum(p.numel() for p in enc.parameters())
        print(f"\n[ffn_type={ffn_type} + id-aware decay] Output shape: {tuple(out_ffn.shape)}, "
              f"parameters: {n_p:,}")
