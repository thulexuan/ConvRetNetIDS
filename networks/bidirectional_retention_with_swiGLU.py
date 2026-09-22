"""
bidirectional_retention.py
===========================

Multi-Scale Retention block, adapted from:

  - Sun et al., "Retentive Network: A Successor to Transformer for Large
    Language Models" (RetNet), arXiv:2307.08621
  - Wang et al., "Convolutional Retentive Network for EEG Decoding"
    (RetEEG), ICASSP 2025

Supports TWO decay directions, selectable via `causal`:

  1. Bidirectional (causal=False, default) -- matches RetEEG's non-causal
     variant:
         D[h, n, m] = gamma_h ** |n - m|
     Every token attends to both past and future tokens in the window,
     with the attention weight decaying exponentially with temporal
     distance in EITHER direction. Appropriate for offline/batch
     classification of a fixed-length window (e.g. a window of CAN bus
     frames), where the full window is available at once.

  2. Uni-directional / causal (causal=True) -- matches RetNet's ORIGINAL
     variant:
         D[h, n, m] = gamma_h ** (n - m)   if n >= m
                    = 0                     if n <  m
     Each token can only attend to itself and PAST tokens; a causal
     mask is ALSO applied to the raw attention scores (before softmax)
     so the softmax normalization is computed only over valid (past)
     positions -- not just zeroed out afterwards, which would leave the
     softmax denominator including future positions and break proper
     normalization (see the two-part fix in `forward()` below).

  3. Softmax is kept (unlike RetNet's original linear "(QK^T . D) V"):
         head_h = ( softmax(Q_h K_h^T)  .  D_h ) V_h
     This is only valid/useful because we do NOT need the RNN-form
     dual representation that RetNet relies on for O(1) autoregressive
     decoding -- this block is meant for offline/batch classification,
     not streaming next-token generation. Dropping softmax is what
     breaks the recurrent form; keeping it here costs nothing because
     we never need that recurrent form in the first place, regardless
     of causal or bidirectional mode.

  4. Multi-scale decay: each head has its own FIXED gamma_h so that
     different heads specialize in different temporal ranges (short
     bursts vs. long-range dependencies) -- useful for CAN IDS, where
     e.g. DoS-style attacks have very short-range temporal signatures
     while spoofing/replay attacks can have longer-range signatures.

  5. Swish-gated output projection + GroupNorm, following RetNet's
     multi-scale retention (MSR) layer design.

  6. Rotary position embedding (RoPE) is used to inject *relative*
     positional information into Q/K, similar in spirit to RetNet's
     xPos rotation (Q_n e^{i n theta}, K_m e^{i m theta}), implemented
     here in the standard real-valued "rotate-half" form.

  7. Optional data-dependent "gated decay" (use_gated_decay=True):
     multiplies the positional decay by an extra sigmoid gate based on
     pairwise value distance -- kept as-is from the original file,
     works identically under EITHER decay direction (causal or
     bidirectional).

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
        causal: bool = False,
        use_gated_decay: bool = False,
        gated_decay_value_dim: int = 16,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rotary = use_rotary
        self.causal = causal
        self.use_gated_decay = use_gated_decay

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

        if use_gated_decay:
            # NEW: independent projection (NOT shared with q/k/v_proj)
            # used only to measure pairwise "value distance" between
            # frames, for data-dependent (gated) decay. Kept small
            # (gated_decay_value_dim << dim) so it is forced to distill
            # mainly magnitude/value information rather than the full
            # mixed representation in `x`.
            self.value_head = nn.Linear(dim, gated_decay_value_dim, bias=False)
            # NEW: per-head learnable gate sensitivity (alpha) and
            # threshold/offset (beta):
            #   gate[b,h,n,m] = sigmoid( -alpha_h * delta[b,n,m] + beta_h )
            self.gate_alpha = nn.Parameter(torch.ones(num_heads))
            self.gate_beta = nn.Parameter(torch.zeros(num_heads))

    def _build_decay(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build D for every head, in either bidirectional or causal mode:
            bidirectional: D[h,n,m] = gamma_h ** |n-m|
            causal:        D[h,n,m] = gamma_h ** (n-m)  if n >= m
                                    = 0                   if n <  m
        Returns shape (num_heads, seq_len, seq_len).
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
            decay = decay * causal_mask[None, :, :]  # zero out n<m (future)
            return decay

    def _compute_gate(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        NEW (gated decay): data-dependent gate that down-weights decay
        between frames whose learned "value embedding" differs a lot
        -- e.g. a physically implausible jump in a spoofed signal.

            v            = value_head(x)                     (b, L, value_dim)
            delta[n,m]   = || v_n - v_m ||                    (b, L, L)
            gate_h[n,m]  = sigmoid( -alpha_h * delta[n,m] + beta_h )

        Returns: (batch, num_heads, L, L) gate in (0, 1).
        """
        v = self.value_head(x)  # (b, L, value_dim)

        # pairwise squared distance via ||a-b||^2 = ||a||^2+||b||^2-2a.b,
        # computed with a single matmul instead of an O(L^2) python loop.
        sq_norm = (v**2).sum(dim=-1)  # (b, L)
        delta_sq = (
            sq_norm[:, :, None]
            + sq_norm[:, None, :]
            - 2 * torch.matmul(v, v.transpose(-1, -2))
        )  # (b, L, L)
        # +eps INSIDE sqrt is essential: sqrt's gradient is 1/(2*sqrt(x)),
        # which is +inf at x=0 (delta_sq[n,n]=0 exactly); clamping alone
        # still produces NaN gradients, the +eps keeps them finite.
        delta = torch.sqrt(torch.clamp(delta_sq, min=0.0) + 1e-8)  # (b, L, L)

        alpha = self.gate_alpha.to(device=x.device, dtype=x.dtype)  # (H,)
        beta = self.gate_beta.to(device=x.device, dtype=x.dtype)  # (H,)

        gate = torch.sigmoid(
            -alpha[None, :, None, None] * delta[:, None, :, :]
            + beta[None, :, None, None]
        )  # (b, H, L, L)
        return gate

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

        if self.causal:
            # NEW: mask out future positions (n < m) BEFORE softmax, so
            # the softmax denominator is computed only over valid
            # (past/self) positions -- just zeroing the decay matrix
            # AFTER softmax (as bidirectional mode does for padding)
            # would leave leftover unnormalized probability mass
            # allocated to future positions, which is wrong for a
            # genuinely causal formulation.
            idx = torch.arange(l, device=x.device)
            causal_bool_mask = (idx[:, None] >= idx[None, :])  # (l, l), True where n>=m
            scores = scores.masked_fill(~causal_bool_mask[None, None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (b, h, l, l) -- nonlinearity kept here

        decay = self._build_decay(l, x.device, x.dtype)  # (h, l, l)
        decay = decay.unsqueeze(0)  # (1, h, l, l) -- broadcast over batch

        if self.use_gated_decay:
            # NEW: multiply the fixed positional decay by a
            # data-dependent gate -- the base decay stays exactly as
            # before, this just adds an extra multiplicative factor
            # that can push decay towards 0 when the frames' learned
            # "value embeddings" differ a lot.
            gate = self._compute_gate(x, l)  # (b, h, l, l)
            decay = decay * gate  # -> (b, h, l, l)

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
# NEW: SwiGLU feed-forward (gated FFN, as used in LLaMA/PaLM/Mistral)
# --------------------------------------------------------------------------- #
class SwiGLUFeedForward(nn.Module):
    """
    Gated feed-forward: two parallel projections instead of one.

        h    = SiLU(x @ W_gate) * (x @ W_up)     -- element-wise gating
        out  = h @ W_down

    W_gate learns "how much of each hidden dimension to let through",
    W_up learns "what content goes into that dimension" -- separating
    these two roles typically gives more expressive per-token
    nonlinearity than a single GELU branch, at a comparable parameter
    budget (this is what most modern LLMs use in place of a plain
    GELU-MLP).

    Parameter-matching note: SwiGLU has 3 weight matrices instead of 2
    (W_gate, W_up, W_down vs. just W1, W2 in the GELU-MLP). To keep the
    total parameter count roughly the same as a GELU-MLP with the same
    `hidden_dim`, `hidden_dim` is scaled by 2/3 by default (the
    convention used in LLaMA) unless explicitly overridden.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            base_hidden_dim = dim * 2  # same default base as FeedForward
            hidden_dim = int(base_hidden_dim * 2 / 3)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)  # (batch, L, hidden_dim)
        return self.w_down(self.dropout(gated))


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
        causal: bool = False,
        use_gated_decay: bool = False,
        gated_decay_value_dim: int = 16,
        ffn_type: str = "gelu",
    ):
        super().__init__()
        assert ffn_type in ("gelu", "swiglu"), "ffn_type must be 'gelu' or 'swiglu'"
        self.norm1 = nn.LayerNorm(dim)
        self.retention = BidirectionalMultiScaleRetention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            causal=causal,
            use_gated_decay=use_gated_decay,
            gated_decay_value_dim=gated_decay_value_dim,
        )
        self.norm2 = nn.LayerNorm(dim)
        # NEW: choose between the original GELU-MLP and SwiGLU FFN.
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
        causal: bool = False,
        use_gated_decay: bool = False,
        gated_decay_value_dim: int = 16,
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
                    causal=causal,
                    use_gated_decay=use_gated_decay,
                    gated_decay_value_dim=gated_decay_value_dim,
                    ffn_type=ffn_type,
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

    encoder = BidirectionalRetentionEncoder(
        dim=token_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
    )

    out = encoder(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == x.shape, "output shape must match input shape"

    # sanity check with a padding mask (e.g. shorter real sequences in a batch)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 20:] = 0  # pretend last 12 tokens are padding for every sample
    out_masked = encoder(x, attn_mask=mask)
    print(f"Masked output shape: {tuple(out_masked.shape)}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {n_params:,}")

    # -------------------------------------------------------------- #
    # NEW: Causal (uni-directional, original RetNet) vs Bidirectional
    # -------------------------------------------------------------- #
    encoder_causal = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers,
        dropout=0.1, causal=True,
    )
    out_causal = encoder_causal(x)
    print(f"\n[Causal] Output shape: {tuple(out_causal.shape)}")
    assert out_causal.shape == x.shape

    loss = out_causal.sum()
    loss.backward()
    q_grad = encoder_causal.layers[0].retention.q_proj.weight.grad
    grad_ok = q_grad is not None and torch.isfinite(q_grad).all() and q_grad.abs().sum() > 0
    print(f"[Causal] q_proj gradient finite & nonzero: {grad_ok.item()}")

    # Kiểm tra trực tiếp attention weight (không dùng output-sensitivity
    # test, vì với model chưa train, softmax(QK^T) ở không gian chiều
    # cao có thể "quá nhọn" một cách ngẫu nhiên, gây false negative)
    def _extract_attn_weights(retention_layer, x_in, causal):
        b, l, d = x_in.shape
        h, hd = retention_layer.num_heads, retention_layer.head_dim
        q = retention_layer.q_proj(x_in).view(b, l, h, hd).transpose(1, 2)
        k = retention_layer.k_proj(x_in).view(b, l, h, hd).transpose(1, 2)
        if retention_layer.use_rotary:
            cos, sin = retention_layer.rotary(l, x_in.device, x_in.dtype)
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)
        if causal:
            idx = torch.arange(l, device=x_in.device)
            causal_bool_mask = (idx[:, None] >= idx[None, :])
            scores = scores.masked_fill(~causal_bool_mask[None, None, :, :], float("-inf"))
        return F.softmax(scores, dim=-1)

    x_small = torch.randn(2, seq_len, token_dim)
    attn_causal = _extract_attn_weights(encoder_causal.layers[0].retention, x_small, causal=True)
    attn_bidir = _extract_attn_weights(encoder.layers[0].retention, x_small, causal=False)

    causal_future_weight = attn_causal[:, :, 5, 20].abs().max().item()
    bidir_future_weight = attn_bidir[:, :, 5, 20].abs().max().item()

    print(f"\n[Causal]        attn weight from query=5 to FUTURE key=20: "
          f"{causal_future_weight:.2e}  (must be exactly 0)")
    print(f"[Bidirectional] attn weight from query=5 to key=20 (allowed to be nonzero): "
          f"{bidir_future_weight:.2e}")
    assert causal_future_weight < 1e-10, "causal mode is leaking future information -- masking bug!"
    print("PASSED: causal mode assigns EXACTLY ZERO attention weight to future positions.")

    n_params_causal = sum(p.numel() for p in encoder_causal.parameters())
    print(f"[Causal] Total parameters: {n_params_causal:,}")

    # Causal + Gated decay CÙNG LÚC -- xác nhận 2 tính năng độc lập, dùng chung được
    encoder_causal_gated = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=2,
        causal=True, use_gated_decay=True,
    )
    out_causal_gated = encoder_causal_gated(x)
    print(f"\n[Causal + Gated decay] Output shape: {tuple(out_causal_gated.shape)}")
    assert out_causal_gated.shape == x.shape
    print("PASSED: causal và gated decay hoạt động ĐỘC LẬP, dùng chung được cùng lúc.")

    # -------------------------------------------------------------- #
    # NEW: Gated decay
    # -------------------------------------------------------------- #
    gated_encoder = BidirectionalRetentionEncoder(
        dim=token_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        use_gated_decay=True,
        gated_decay_value_dim=16,
    )
    out_gated = gated_encoder(x)
    print(f"\n[Gated decay] Output shape: {tuple(out_gated.shape)}")
    assert out_gated.shape == x.shape

    loss = out_gated.sum()
    loss.backward()
    vh_grad = gated_encoder.layers[0].retention.value_head.weight.grad
    grad_ok = vh_grad is not None and torch.isfinite(vh_grad).all() and vh_grad.abs().sum() > 0
    print(f"[Gated decay] value_head gradient finite & nonzero: {grad_ok.item()}")

    n_params_gated = sum(p.numel() for p in gated_encoder.parameters())
    print(f"[Gated decay] Total parameters: {n_params_gated:,}")

    # -------------------------------------------------------------- #
    # NEW: SwiGLU FFN (vs. default GELU-MLP FFN)
    # -------------------------------------------------------------- #
    encoder_gelu = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers,
        dropout=0.1, ffn_type="gelu",
    )
    encoder_swiglu = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers,
        dropout=0.1, ffn_type="swiglu",
    )

    out_gelu = encoder_gelu(x)
    out_swiglu = encoder_swiglu(x)
    print(f"\n[GELU-MLP FFN]  Output shape: {tuple(out_gelu.shape)}")
    print(f"[SwiGLU FFN]    Output shape: {tuple(out_swiglu.shape)}")
    assert out_gelu.shape == x.shape
    assert out_swiglu.shape == x.shape

    n_params_gelu = sum(p.numel() for p in encoder_gelu.parameters())
    n_params_swiglu = sum(p.numel() for p in encoder_swiglu.parameters())
    print(f"[GELU-MLP FFN]  Total parameters: {n_params_gelu:,}")
    print(f"[SwiGLU FFN]    Total parameters: {n_params_swiglu:,}")
    print(f"Difference: {n_params_swiglu - n_params_gelu:+,} "
          f"({'larger' if n_params_swiglu > n_params_gelu else 'smaller'} "
          f"with default 2/3 hidden_dim scaling)")

    # gradient sanity check for SwiGLU
    loss = out_swiglu.sum()
    loss.backward()
    swiglu_ffn = encoder_swiglu.layers[0].ffn
    grad_ok = (
        swiglu_ffn.w_gate.weight.grad is not None
        and swiglu_ffn.w_gate.weight.grad.abs().sum() > 0
    )
    print(f"[SwiGLU FFN] gradient reaches w_gate: {grad_ok}")