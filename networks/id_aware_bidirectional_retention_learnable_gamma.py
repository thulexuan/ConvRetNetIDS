"""
bidirectional_retention_full.py
==================================

Bidirectional Multi-Scale Retention block with:
  (a) LEARNABLE gamma (sigmoid reparameterization, initialized from
      RetNet's original fixed schedule), and
  (b) ID-aware decay: same-CAN-ID pairs decay more slowly than
      different-ID pairs, at the same positional distance.

Both features can be toggled independently:
    learnable_gamma:     True  -> gamma(s) are nn.Parameter, updated
                                  every optimizer step
                          False -> gamma(s) are fixed buffers (RetNet's
                                  original choice), never updated
    use_id_aware_decay:  True  -> two decay rates, gamma_same / gamma_diff,
                                  selected per (n,m) pair via can_ids
                          False -> a single decay rate per head,
                                  gamma_h ** |n-m| (position-only)

Design recap
------------
1. Bidirectional decay: every token attends to both past and future
   tokens (unlike RetNet's original causal decay).

2. Softmax is kept (unlike RetNet's linear form): valid here because
   this block runs on a fixed-length window in one shot (offline/batch
   classification), never needing RetNet's O(1) autoregressive
   recurrent dual form -- dropping softmax is what enables that dual
   form, and we simply don't need it.

3. gamma reparameterization: gamma = sigmoid(w), w unconstrained. This
   guarantees gamma in (0,1) regardless of what gradient descent does
   to w. w is initialized so gamma starts exactly at RetNet's original
   fixed schedule (gamma_h = 1 - 2^(-5-h)), then is free to drift from
   there based on the data. gamma changes after every optimizer.step()
   (i.e. every batch, not just every epoch) -- there is no per-epoch
   "checkpoint" that pauses updates between epochs; training is a
   continuous stream of steps and epoch boundaries have no special
   effect on when gamma updates.

4. ID-aware decay:
       D_h[n,m] = gamma_same_h ** |n-m|   if can_id[n] == can_id[m]
                = gamma_diff_h ** |n-m|   otherwise
   gamma_diff_h follows RetNet's schedule; gamma_same_h is derived so
   that gamma_same_h >= gamma_diff_h holds STRUCTURALLY (by construction
   of the reparameterization), never violated during training:
       gamma_diff_h = sigmoid(w_diff_h)
       gamma_same_h = gamma_diff_h + (1 - gamma_diff_h) * sigmoid(w_extra_h)
   Both w_diff_h and w_extra_h are learnable when learnable_gamma=True;
   fixed (buffers) otherwise, in which case gamma_same_h is instead
   derived by an exponent shift (gamma_same_h = 1 - 2^(-5-h-delta)),
   matching the earlier fixed-decay design.

5. Swish-gated output projection + GroupNorm, RoPE for Q/K -- unchanged
   from earlier versions.

Input / output:
    x:       (batch, seq_len, token_dim)
    can_ids: (batch, seq_len) int64, required only if
             use_id_aware_decay=True
    output:  (batch, seq_len, token_dim)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotary position embedding
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


def _logit(p: torch.Tensor) -> torch.Tensor:
    """Inverse sigmoid: logit(p) = log(p/(1-p)), so sigmoid(logit(p)) == p."""
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


# --------------------------------------------------------------------------- #
# Bidirectional multi-scale retention (softmax kept)
# --------------------------------------------------------------------------- #
class BidirectionalMultiScaleRetention(nn.Module):
    """
    Bidirectional Multi-Scale Retention layer, with learnable gamma and
    optional ID-aware decay.

        head_h = ( softmax(Q_h K_h^T / sqrt(d_h)) . D_h ) V_h

    Args:
        dim:              token embedding dimension
        num_heads:        number of retention heads
        dropout:          dropout applied to attention weights
        use_rotary:       whether to apply RoPE to Q/K
        learnable_gamma:  if True, gamma(s) are learnable (sigmoid
                          reparameterization), initialized from RetNet's
                          fixed schedule. If False, gamma(s) are fixed
                          buffers, never updated.
        use_id_aware_decay: if True, use same-ID / diff-ID decay (requires
                          `can_ids` at forward time). If False, use plain
                          position-only decay.
        same_id_init_extra: initial value (0,1) for the sigmoid term
                          controlling how much slower gamma_same starts
                          relative to gamma_diff (only used when both
                          learnable_gamma and use_id_aware_decay are True).
        id_decay_delta:   exponent shift for gamma_same when
                          learnable_gamma=False and use_id_aware_decay=True
                          (gamma_same_h = 1 - 2^(-5-h-delta)).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
        learnable_gamma: bool = True,
        use_id_aware_decay: bool = False,
        same_id_init_extra: float = 0.9,
        id_decay_delta: float = 3.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rotary = use_rotary
        self.learnable_gamma = learnable_gamma
        self.use_id_aware_decay = use_id_aware_decay

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)  # swish gate
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim)
        self.dropout = nn.Dropout(dropout)

        if use_rotary:
            self.rotary = RotaryPositionEmbedding(self.head_dim)

        # RetNet's original fixed schedule -- used as the FIXED value
        # (learnable_gamma=False) or as the INITIALIZATION point for
        # learnable parameters (learnable_gamma=True).
        h_idx = torch.arange(num_heads, dtype=torch.float32)
        gamma_diff_init = 1.0 - torch.pow(2.0, -5.0 - h_idx)  # (num_heads,)

        if not use_id_aware_decay:
            # ---- Position-only decay: a single gamma per head ----
            if learnable_gamma:
                raw_init = _logit(gamma_diff_init)
                self.raw_gamma = nn.Parameter(raw_init)  # learnable
            else:
                self.register_buffer("gammas", gamma_diff_init)  # fixed
        else:
            # ---- Same-ID / diff-ID decay: two gammas per head ----
            if learnable_gamma:
                # gamma_diff_h = sigmoid(w_diff_h), initialized to RetNet's schedule
                raw_diff_init = _logit(gamma_diff_init)
                self.raw_gamma_diff = nn.Parameter(raw_diff_init)  # learnable
                # gamma_same_h = gamma_diff_h + (1-gamma_diff_h)*sigmoid(w_extra_h)
                # => gamma_same_h >= gamma_diff_h holds STRUCTURALLY, always,
                # regardless of what training does to either parameter.
                raw_extra_init = torch.full(
                    (num_heads,), _logit(torch.tensor(same_id_init_extra)).item()
                )
                self.raw_same_extra = nn.Parameter(raw_extra_init)  # learnable
            else:
                # Fixed: gamma_diff from RetNet's schedule, gamma_same by
                # shifting the exponent (always > gamma_diff for delta>0).
                gamma_same_init = 1.0 - torch.pow(2.0, -5.0 - h_idx - id_decay_delta)
                self.register_buffer("gamma_diff", gamma_diff_init)  # fixed
                self.register_buffer("gamma_same", gamma_same_init)  # fixed

    # ----------------------------------------------------------------- #
    # gamma accessors (always return the CURRENT effective value)
    # ----------------------------------------------------------------- #
    def get_gamma(self) -> torch.Tensor:
        """Position-only mode: returns gamma_h, shape (num_heads,)."""
        assert not self.use_id_aware_decay
        if self.learnable_gamma:
            return torch.sigmoid(self.raw_gamma)
        return self.gammas

    def get_gamma_same_diff(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ID-aware mode: returns (gamma_same_h, gamma_diff_h)."""
        assert self.use_id_aware_decay
        if self.learnable_gamma:
            gamma_diff = torch.sigmoid(self.raw_gamma_diff)
            same_extra = torch.sigmoid(self.raw_same_extra)
            gamma_same = gamma_diff + (1.0 - gamma_diff) * same_extra
            return gamma_same, gamma_diff
        return self.gamma_same, self.gamma_diff

    def get_decay_rates(self) -> dict:
        """Utility for inspection/logging."""
        with torch.no_grad():
            if not self.use_id_aware_decay:
                return {"gamma": self.get_gamma().tolist()}
            gamma_same, gamma_diff = self.get_gamma_same_diff()
            return {"gamma_same": gamma_same.tolist(), "gamma_diff": gamma_diff.tolist()}

    # ----------------------------------------------------------------- #
    # Decay matrix construction
    # ----------------------------------------------------------------- #
    def _position_only_decay(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """D_h[n,m] = gamma_h ** |n-m|.  Returns (1, num_heads, L, L)."""
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype)
        gammas = self.get_gamma().to(device=device, dtype=dtype)
        log_gamma = torch.log(gammas)
        decay = torch.exp(log_gamma[:, None, None] * dist[None, :, :])
        return decay.unsqueeze(0)

    def _id_aware_decay(
        self, seq_len: int, can_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        D_h[n,m] = gamma_same_h**|n-m|  if can_id[n]==can_id[m]
                 = gamma_diff_h**|n-m|  otherwise
        Returns (batch, num_heads, L, L).
        """
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype)  # (L, L)

        gamma_same, gamma_diff = self.get_gamma_same_diff()
        gamma_same = gamma_same.to(device=device, dtype=dtype)
        gamma_diff = gamma_diff.to(device=device, dtype=dtype)

        log_same = torch.log(gamma_same)
        log_diff = torch.log(gamma_diff)

        decay_same = torch.exp(log_same[:, None, None] * dist[None, :, :])  # (H,L,L)
        decay_diff = torch.exp(log_diff[:, None, None] * dist[None, :, :])  # (H,L,L)

        same_id_mask = (can_ids[:, :, None] == can_ids[:, None, :]).to(dtype)  # (B,L,L)

        decay = (
            same_id_mask[:, None, :, :] * decay_same[None, :, :, :]
            + (1.0 - same_id_mask[:, None, :, :]) * decay_diff[None, :, :, :]
        )  # (B, H, L, L)
        return decay

    def forward(
        self,
        x: torch.Tensor,
        can_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (batch, seq_len, dim)
            can_ids:   (batch, seq_len) int, required iff use_id_aware_decay=True
            attn_mask: optional (batch, seq_len), 1=valid, 0=padding
        """
        b, l, d = x.shape
        h, hd = self.num_heads, self.head_dim

        if self.use_id_aware_decay:
            if can_ids is None:
                raise ValueError(
                    "use_id_aware_decay=True requires `can_ids` (batch, seq_len)."
                )
            if can_ids.shape[0] != b or can_ids.shape[1] != l:
                raise ValueError(
                    f"Shape mismatch: x is (batch={b}, seq_len={l}, dim={d}) but "
                    f"can_ids is {tuple(can_ids.shape)} (expected (batch={b}, seq_len={l}))."
                )

        q = self.q_proj(x).view(b, l, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, l, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, l, h, hd).transpose(1, 2)

        if self.use_rotary:
            cos, sin = self.rotary(l, x.device, x.dtype)
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)

        if attn_mask is not None:
            key_mask = attn_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        if self.use_id_aware_decay:
            decay = self._id_aware_decay(l, can_ids, x.device, x.dtype)  # (b,h,l,l)
        else:
            decay = self._position_only_decay(l, x.device, x.dtype)  # (1,h,l,l)

        attn = attn * decay
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, l, d)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        gate = F.silu(self.g_proj(x))
        out = gate * out
        out = self.out_proj(out)
        return out


# --------------------------------------------------------------------------- #
# Feed-forward network (GELU-MLP)
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
# One retention block
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
        learnable_gamma: bool = True,
        use_id_aware_decay: bool = False,
        same_id_init_extra: float = 0.9,
        id_decay_delta: float = 3.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.retention = BidirectionalMultiScaleRetention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            learnable_gamma=learnable_gamma,
            use_id_aware_decay=use_id_aware_decay,
            same_id_init_extra=same_id_init_extra,
            id_decay_delta=id_decay_delta,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

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
# Stack of blocks
# --------------------------------------------------------------------------- #
class BidirectionalRetentionEncoder(nn.Module):
    """
    Stack of BidirectionalRetentionBlock layers.

        raw CAN window -> local encoder -> (batch, seq_len, token_dim)
                        -> BidirectionalRetentionEncoder (this module)
                        -> pooling -> classifier -> attack class logits

    If use_id_aware_decay=True, the CAN ID of each frame must be passed
    as `can_ids` (batch, seq_len) at forward time.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_rotary: bool = True,
        learnable_gamma: bool = True,
        use_id_aware_decay: bool = False,
        same_id_init_extra: float = 0.9,
        id_decay_delta: float = 3.0,
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
                    learnable_gamma=learnable_gamma,
                    use_id_aware_decay=use_id_aware_decay,
                    same_id_init_extra=same_id_init_extra,
                    id_decay_delta=id_decay_delta,
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

    def get_all_decay_rates(self) -> list:
        return [layer.retention.get_decay_rates() for layer in self.layers]


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, seq_len, token_dim = 128, 32, 128
    num_heads = 8
    num_layers = 4
    num_can_ids = 20
    can_ids = torch.randint(0, num_can_ids, (batch_size, seq_len))
    x = torch.randn(batch_size, seq_len, token_dim)

    # ================================================================ #
    # 1) learnable_gamma=True, use_id_aware_decay=True  (main request)
    # ================================================================ #
    encoder = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers, dropout=0.1,
        learnable_gamma=True, use_id_aware_decay=True, same_id_init_extra=0.9,
    )
    out = encoder(x, can_ids=can_ids)
    print("[learnable_gamma=True, use_id_aware_decay=True]")
    print(f"  Output shape: {tuple(out.shape)}")
    assert out.shape == x.shape

    rates_before = encoder.get_all_decay_rates()[0]
    print(f"  gamma_diff (layer0, at init): {[f'{v:.6f}' for v in rates_before['gamma_diff']]}")
    print(f"  gamma_same (layer0, at init): {[f'{v:.6f}' for v in rates_before['gamma_same']]}")

    expected_diff = (1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())).tolist()
    assert all(abs(a - b) < 1e-4 for a, b in zip(rates_before["gamma_diff"], expected_diff)), (
        "gamma_diff should initialize to RetNet's fixed schedule"
    )
    assert all(s > d for s, d in zip(rates_before["gamma_same"], rates_before["gamma_diff"])), (
        "gamma_same must be > gamma_diff at init"
    )

    r0 = encoder.layers[0].retention
    assert isinstance(r0.raw_gamma_diff, nn.Parameter)
    assert isinstance(r0.raw_same_extra, nn.Parameter)
    print("  raw_gamma_diff and raw_same_extra are learnable nn.Parameter: True")

    # confirm gradient flows and gammas actually move after an optimizer step
    loss = out.sum()
    loss.backward()
    grad_ok = (
        r0.raw_gamma_diff.grad is not None and r0.raw_gamma_diff.grad.abs().sum() > 0
        and r0.raw_same_extra.grad is not None and r0.raw_same_extra.grad.abs().sum() > 0
    )
    print(f"  Gradients finite & nonzero for both raw_gamma_diff and raw_same_extra: {grad_ok}")

    opt = torch.optim.Adam(encoder.parameters(), lr=0.1)
    opt.step()
    rates_after = encoder.get_all_decay_rates()[0]
    changed = any(
        abs(a - b) > 1e-6 for a, b in zip(rates_before["gamma_diff"], rates_after["gamma_diff"])
    )
    print(f"  gamma_diff changed after 1 optimizer step: {changed}")
    assert all(0.0 < g < 1.0 for g in rates_after["gamma_diff"] + rates_after["gamma_same"]), (
        "all gammas must remain within (0,1)"
    )
    assert all(s > d for s, d in zip(rates_after["gamma_same"], rates_after["gamma_diff"])), (
        "gamma_same must STILL be > gamma_diff after the optimizer step " \
        "(constraint must hold structurally, not just at init)"
    )
    print("  gamma_same > gamma_diff still holds after training step: True")
    print("  All gammas remain within (0,1) after training step: True")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {n_params:,}")

    # ================================================================ #
    # 2) learnable_gamma=False, use_id_aware_decay=True (fixed, for contrast)
    # ================================================================ #
    encoder_fixed_id = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers,
        learnable_gamma=False, use_id_aware_decay=True, id_decay_delta=3.0,
    )
    out_fixed_id = encoder_fixed_id(x, can_ids=can_ids)
    print("\n[learnable_gamma=False, use_id_aware_decay=True]")
    print(f"  Output shape: {tuple(out_fixed_id.shape)}")
    is_buffer = "gamma_same" in dict(encoder_fixed_id.layers[0].retention.named_buffers())
    is_param = "gamma_same" in dict(encoder_fixed_id.layers[0].retention.named_parameters())
    print(f"  gamma_same is a fixed buffer: {is_buffer and not is_param}")
    n_params_fixed = sum(p.numel() for p in encoder_fixed_id.parameters())
    print(f"  Total parameters: {n_params_fixed:,} "
          f"(fewer than learnable version by {n_params - n_params_fixed}: "
          f"{n_params - n_params_fixed == num_layers * num_heads * 2})")

    # ================================================================ #
    # 3) learnable_gamma=True, use_id_aware_decay=False (position-only, learnable)
    # ================================================================ #
    encoder_pos = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers,
        learnable_gamma=True, use_id_aware_decay=False,
    )
    out_pos = encoder_pos(x)  # no can_ids needed
    print("\n[learnable_gamma=True, use_id_aware_decay=False]")
    print(f"  Output shape: {tuple(out_pos.shape)}")
    assert out_pos.shape == x.shape
    print(f"  gamma (layer0): {[f'{v:.6f}' for v in encoder_pos.get_all_decay_rates()[0]['gamma']]}")

    # ================================================================ #
    # 4) padding mask works together with id-aware + learnable gamma
    # ================================================================ #
    mask = torch.ones(batch_size, seq_len)
    mask[:, 20:] = 0
    out_masked = encoder(x, can_ids=can_ids, attn_mask=mask)
    print(f"\n[+ padding mask, id-aware + learnable] Output shape: {tuple(out_masked.shape)}")

    # ================================================================ #
    # 5) missing can_ids raises a clear error when required
    # ================================================================ #
    try:
        encoder(x)  # forgot can_ids
        print("\n[ERROR] Expected ValueError was not raised!")
    except ValueError as e:
        print(f"\n[Missing can_ids check] Correctly raised: {e}")
