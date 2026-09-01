"""
bidirectional_retention_learnable_gamma.py
=============================================

Bidirectional Multi-Scale Retention block -- base version (no ID-aware
decay), with LEARNABLE gamma per head instead of RetNet's original
fixed schedule.

Design:

  1. Bidirectional decay matrix:
         D_h[n, m] = gamma_h ** |n - m|
     Every token attends to both past and future tokens.

  2. Softmax is kept (unlike RetNet's linear "(QK^T . D) V"): valid
     here because this block is used for offline/batch classification
     of a fixed-length window, not streaming autoregressive decoding,
     so we never need the O(1) recurrent dual form that motivates
     RetNet's choice to drop softmax.

  3. gamma_h is LEARNABLE (not fixed), reparameterized through a
     sigmoid so it always stays in (0, 1) regardless of what the raw
     underlying parameter becomes during training:

         gamma_h = sigmoid(w_h),   w_h in R (unconstrained, learnable)

     w_h is initialized so that sigmoid(w_h) exactly matches RetNet's
     original fixed schedule at step 0 (gamma_h = 1 - 2^(-5-h)) --
     i.e. training starts from the same multi-scale prior RetNet uses,
     but gradient descent is then free to adjust each head's decay
     rate based on the data, instead of keeping it frozen forever.

     Why sigmoid: it is the simplest monotonic map R -> (0,1). Note
     it saturates (near-zero gradient) exactly in the region gamma is
     close to 1, which is where RetNet's gammas actually live
     (0.97-0.9998) -- an alternative, gamma = exp(-exp(w)) (the S4/
     Mamba-style parameterization, see StableSSM, arXiv:2311.14495),
     spreads resolution more evenly across that region and may train
     better in practice; sigmoid is used here for simplicity and is
     provided as the default, swappable via `gamma_parameterization`.

  4. Swish-gated output projection + GroupNorm, following RetNet's
     multi-scale retention (MSR) layer design.

  5. Rotary position embedding (RoPE) is used to inject *relative*
     positional information into Q/K.

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
# Rotary position embedding (relative position prior for Q/K)
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
    """Inverse sigmoid: logit(p) = log(p / (1-p)), so sigmoid(logit(p)) == p."""
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


# --------------------------------------------------------------------------- #
# Bidirectional multi-scale retention (softmax kept, gamma learnable)
# --------------------------------------------------------------------------- #
class BidirectionalMultiScaleRetention(nn.Module):
    """
    Bidirectional Multi-Scale Retention layer.

        head_h = ( softmax(Q_h K_h^T / sqrt(d_h)) . D_h ) V_h
        D_h[n, m] = gamma_h ** |n - m|

        MSR(X) = swish(X W_G)  .  Concat(head_1, ..., head_H) W_O

    Args:
        dim:        token embedding dimension (token_dim)
        num_heads:  number of retention heads
        dropout:    dropout applied to attention weights
        use_rotary: whether to apply rotary position embedding to Q/K
        learnable_gamma: if True (default), gamma_h is a learnable
            parameter (reparameterized via sigmoid to stay in (0,1)),
            initialized from RetNet's original fixed schedule
            (gamma_h = 1 - 2^(-5-h)). If False, gamma_h is fixed at
            that schedule and never updated (RetNet's original choice).
        gamma_parameterization: "sigmoid" (default) or "expexp"
            (gamma = exp(-exp(w)), the S4/Mamba-style parameterization
            -- see module docstring for the trade-off between the two).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
        learnable_gamma: bool = True,
        gamma_parameterization: str = "sigmoid",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert gamma_parameterization in ("sigmoid", "expexp"), (
            "gamma_parameterization must be 'sigmoid' or 'expexp'"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rotary = use_rotary
        self.learnable_gamma = learnable_gamma
        self.gamma_parameterization = gamma_parameterization

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # swish gate, as in RetNet's MSR: swish(X W_G) . Y
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim)
        self.dropout = nn.Dropout(dropout)

        if use_rotary:
            self.rotary = RotaryPositionEmbedding(self.head_dim)

        # RetNet's original fixed schedule, used either directly (if
        # learnable_gamma=False) or as the INITIALIZATION point for the
        # learnable raw parameter (if learnable_gamma=True).
        h_idx = torch.arange(num_heads, dtype=torch.float32)
        init_gamma = 1.0 - torch.pow(2.0, -5.0 - h_idx)  # (num_heads,)

        if not learnable_gamma:
            self.register_buffer("gammas", init_gamma)
        else:
            if gamma_parameterization == "sigmoid":
                raw_init = _logit(init_gamma)  # sigmoid(raw_init) == init_gamma
            else:  # "expexp": gamma = exp(-exp(w))  =>  w = log(-log(gamma))
                raw_init = torch.log(-torch.log(init_gamma))
            self.raw_gamma = nn.Parameter(raw_init)  # (num_heads,), learnable

    def get_gamma(self) -> torch.Tensor:
        """Returns the current effective gamma_h, shape (num_heads,)."""
        if not self.learnable_gamma:
            return self.gammas
        if self.gamma_parameterization == "sigmoid":
            return torch.sigmoid(self.raw_gamma)
        else:  # "expexp"
            return torch.exp(-torch.exp(self.raw_gamma))

    def _bidirectional_decay(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """D_h[n, m] = gamma_h ** |n - m|.  Returns (1, num_heads, L, L)."""
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype)  # (L, L)
        gammas = self.get_gamma().to(device=device, dtype=dtype)  # (H,)
        log_gamma = torch.log(gammas)  # (H,)
        decay = torch.exp(log_gamma[:, None, None] * dist[None, :, :])  # (H, L, L)
        return decay.unsqueeze(0)  # (1, H, L, L) -- broadcasts over batch

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x:         (batch, seq_len, dim)
            attn_mask: optional (batch, seq_len), 1=valid, 0=padding
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
            key_mask = attn_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (b, h, l, l)

        decay = self._bidirectional_decay(l, x.device, x.dtype)  # (1, h, l, l)
        attn = attn * decay
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (b, h, l, hd)
        out = out.transpose(1, 2).reshape(b, l, d)

        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        gate = F.silu(self.g_proj(x))  # swish(X W_G)
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
        learnable_gamma: bool = True,
        gamma_parameterization: str = "sigmoid",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.retention = BidirectionalMultiScaleRetention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            learnable_gamma=learnable_gamma,
            gamma_parameterization=gamma_parameterization,
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
# Stack of blocks
# --------------------------------------------------------------------------- #
class BidirectionalRetentionEncoder(nn.Module):
    """
    Stack of BidirectionalRetentionBlock layers.

        raw CAN window -> local encoder -> (batch, seq_len, token_dim)
                        -> BidirectionalRetentionEncoder (this module)
                        -> pooling -> classifier -> attack class logits
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
        gamma_parameterization: str = "sigmoid",
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
                    gamma_parameterization=gamma_parameterization,
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

    def get_all_gammas(self) -> list:
        """Utility: returns the current gamma_h (per head) for every layer."""
        return [layer.retention.get_gamma().tolist() for layer in self.layers]


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
    # 1) Learnable gamma (sigmoid parameterization), default
    # -------------------------------------------------------------- #
    encoder = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers,
        dropout=0.1, learnable_gamma=True, gamma_parameterization="sigmoid",
    )
    out = encoder(x)
    print("[Learnable gamma, sigmoid parameterization]")
    print(f"  Output shape: {tuple(out.shape)}")
    assert out.shape == x.shape

    gammas_before = encoder.get_all_gammas()
    print(f"  Layer 0 gamma at init: {[f'{g:.6f}' for g in gammas_before[0]]}")

    # confirm init matches RetNet's original fixed schedule exactly
    expected = (1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())).tolist()
    print(f"  RetNet fixed schedule: {[f'{g:.6f}' for g in expected]}")
    assert all(abs(a - b) < 1e-4 for a, b in zip(gammas_before[0], expected)), (
        "learnable gamma should be initialized to match RetNet's fixed schedule"
    )

    # confirm raw_gamma is trainable and receives gradient
    loss = out.sum()
    loss.backward()
    raw_grad = encoder.layers[0].retention.raw_gamma.grad
    grad_ok = raw_grad is not None and torch.isfinite(raw_grad).all() and raw_grad.abs().sum() > 0
    print(f"  raw_gamma is a learnable Parameter: "
          f"{isinstance(encoder.layers[0].retention.raw_gamma, nn.Parameter)}")
    print(f"  raw_gamma gradient finite & nonzero: {grad_ok.item()}")

    # simulate one optimizer step and confirm gamma actually changes
    opt = torch.optim.Adam(encoder.parameters(), lr=0.1)
    opt.step()
    gammas_after = encoder.get_all_gammas()
    changed = any(
        abs(a - b) > 1e-6 for a, b in zip(gammas_before[0], gammas_after[0])
    )
    print(f"  gamma changed after 1 optimizer step: {changed}")
    print(f"  Layer 0 gamma after step: {[f'{g:.6f}' for g in gammas_after[0]]}")

    # gamma must always stay within (0,1) regardless of training
    assert all(0.0 < g < 1.0 for g in gammas_after[0]), "gamma escaped (0,1)!"
    print("  gamma remains strictly within (0,1): True")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {n_params:,} (includes {num_layers * num_heads} learnable gammas)")

    # -------------------------------------------------------------- #
    # 2) Fixed gamma (learnable_gamma=False), for comparison
    # -------------------------------------------------------------- #
    encoder_fixed = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=num_layers,
        dropout=0.1, learnable_gamma=False,
    )
    out_fixed = encoder_fixed(x)
    print("\n[Fixed gamma, for comparison]")
    print(f"  Output shape: {tuple(out_fixed.shape)}")
    is_buffer = "gammas" in dict(encoder_fixed.layers[0].retention.named_buffers())
    is_param = "gammas" in dict(encoder_fixed.layers[0].retention.named_parameters())
    print(f"  gammas is a fixed buffer (not learnable): {is_buffer and not is_param}")

    n_params_fixed = sum(p.numel() for p in encoder_fixed.parameters())
    print(f"  Total parameters: {n_params_fixed:,} "
          f"(should be {num_layers * num_heads} fewer than learnable version: "
          f"{n_params - n_params_fixed == num_layers * num_heads})")

    # -------------------------------------------------------------- #
    # 3) Alternative parameterization: gamma = exp(-exp(w))
    # -------------------------------------------------------------- #
    encoder_expexp = BidirectionalRetentionEncoder(
        dim=token_dim, num_heads=num_heads, num_layers=1,
        learnable_gamma=True, gamma_parameterization="expexp",
    )
    out_expexp = encoder_expexp(x)
    print("\n[Learnable gamma, exp(-exp(w)) parameterization]")
    print(f"  Output shape: {tuple(out_expexp.shape)}")
    g_expexp = encoder_expexp.get_all_gammas()[0]
    print(f"  Layer 0 gamma at init: {[f'{g:.6f}' for g in g_expexp]}")
    assert all(abs(a - b) < 1e-4 for a, b in zip(g_expexp, expected)), (
        "expexp parameterization should also initialize to match RetNet's schedule"
    )
    print("  Also correctly initialized to RetNet's fixed schedule: True")

    # sanity: padding mask still works
    mask = torch.ones(batch_size, seq_len)
    mask[:, 20:] = 0
    out_masked = encoder(x, attn_mask=mask)
    print(f"\n[+ padding mask] Output shape: {tuple(out_masked.shape)}")
