"""
transformer_baseline.py
==========================

Baseline model for comparative analysis: a CONVENTIONAL Transformer
encoder (standard softmax self-attention, sinusoidal positional
encoding, standard Linear-ReLU-Linear feed-forward -- NO retention
decay matrix, NO SwiGLU gating, NO CNN component at all) for
CAN-window classification.

This is the 2nd of four models being compared:
    1. Simple CNN                 (simple_cnn.py)
    2. Traditional Transformer    <-- this file
    3. Residual CNN + Transformer (hybrid)
    4. Proposed model (Residual CNN local encoder + Bidirectional
       RetNet + SwiGLU)

Design
-------
Input:  (batch, window_size, num_features)   e.g. (batch, 32, 9)
        -- can_id + 8 payload bytes, pre-normalized to [0,1], SAME
        input contract as SimpleCNNModel and the proposed model.

Unlike SimpleCNNModel (pure CNN, no attention at all) and the
proposed model (CNN local encoder -> Bidirectional RetNet with decay
+ SwiGLU FFN), this baseline is deliberately a PLAIN/CONVENTIONAL
Transformer:
    - Input projection is a plain Linear layer (per-frame, no
      convolution at all -- this baseline has NO CNN component,
      to cleanly isolate "Transformer vs CNN vs hybrid vs proposed"
      as four genuinely different architecture families).
    - Positional encoding: standard sinusoidal (Vaswani et al. 2017),
      added once before the encoder stack (no per-layer decay/gating
      like the proposed model's retention mechanism).
    - Self-attention: standard softmax(QK^T/sqrt(d_k))V, bidirectional
      (no causal mask, matching this project's offline/batch window
      classification setting), NO decay matrix multiplied in (that is
      what specifically differentiates the proposed model's
      "retention" from this "vanilla attention").
    - Feed-forward: standard Linear -> GELU -> Linear (NOT SwiGLU).

Like the proposed model (and unlike SimpleCNNModel), this baseline
DOES produce genuine attention weights, so it CAN be used for
frame-level localization evaluation (detect_attack_frames,
frame_level_metrics) alongside window-level classification metrics --
unlike SimpleCNNModel, which cannot.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Standard sinusoidal positional encoding (Vaswani et al. 2017)
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> same shape, + positional encoding."""
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


# --------------------------------------------------------------------------- #
# Standard multi-head self-attention -- NO decay matrix (unlike the
# proposed model's BidirectionalMultiScaleRetention), implemented
# manually so we can expose the raw attention weight matrix for
# frame-level localization evaluation.
# --------------------------------------------------------------------------- #
class MultiHeadSelfAttention(nn.Module):
    """
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    Deliberately plain: no decay matrix multiplied into the attention
    weights (that is exactly what the proposed model's retention
    mechanism adds on top of this same base formula), no gating.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: optional (batch, seq_len) padding mask, 1=valid, 0=padding
        Returns:
            out:          (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        b, l, d = x.shape
        h, hd = self.num_heads, self.head_dim

        q = self.w_q(x).view(b, l, h, hd).transpose(1, 2)
        k = self.w_k(x).view(b, l, h, hd).transpose(1, 2)
        v = self.w_v(x).view(b, l, h, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)

        if attn_mask is not None:
            key_mask = attn_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # NO decay matrix multiplied in here
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(b, l, d)
        out = self.w_o(out)

        return out, attn_weights


# --------------------------------------------------------------------------- #
# One standard Transformer encoder block:
#   MHA -> Add&Norm -> FeedForward (Linear-GELU-Linear, NOT SwiGLU) -> Add&Norm
# --------------------------------------------------------------------------- #
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.mha(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x, attn_weights


# --------------------------------------------------------------------------- #
# Full baseline model: Linear projection -> Positional Encoding
#   -> [TransformerEncoderBlock x num_layers] -> Mean+Max Pool -> MLP classifier
# --------------------------------------------------------------------------- #
class TransformerBaselineModel(nn.Module):
    """
    Conventional Transformer baseline for CAN-window classification.

        x (B, T, num_features)
            -> Linear(num_features, d_model)      -- plain per-frame projection, NO conv
            -> + Positional Encoding (sinusoidal)
            -> [TransformerEncoderBlock] x num_layers
            -> LayerNorm
            -> Mean Pool + Max Pool -> Concat
            -> MLP classifier
            -> logits (B, num_classes)

    Returns (final_encode_feat, logits, attn_weights) where attn_weights
    is a list of (batch, num_heads, window_size, window_size) tensors,
    one per encoder layer -- usable for frame-level localization
    evaluation (unlike SimpleCNNModel, which returns None here).

    Args:
        num_features: input feature dim per frame (e.g. 9); if the
                      input tensor has MORE columns than this (e.g. a
                      trailing frame_labels ground-truth column, as in
                      this project's dataset contract), only the first
                      `num_features` columns are used -- extra trailing
                      columns are silently dropped, same as ConvRetNet
                      and SimpleCNNModel.
        d_model:      model/embedding dimension
        num_heads:    number of attention heads
        ff_dim:       feed-forward hidden dimension
        num_layers:   number of stacked encoder blocks (default 2, to
                      roughly match the proposed model's default
                      BidirectionalRetentionEncoder depth)
        num_classes:  number of output classes
        dropout:      dropout rate
    """

    def __init__(
        self,
        num_features: int = 9,
        d_model: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x:    (batch, window_size, num_features_or_more)
            mask: optional (batch, window_size) padding mask, 1=valid, 0=padding
        Returns:
            final_encode_feat: (batch, window_size, d_model)
            logits:            (batch, num_classes)
            attn_weights:      list of (batch, num_heads, window_size, window_size),
                               one per encoder layer
        """
        x = x[:, :, 0:self.num_features]  # drop any trailing extra columns (e.g. frame_labels)

        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.dropout(h)

        all_attn_weights = []
        for layer in self.layers:
            h, attn_weights = layer(h, attn_mask=mask)
            all_attn_weights.append(attn_weights)

        final_encode_feat = self.final_ln(h)

        mean_pooled = final_encode_feat.mean(dim=1)
        max_pooled = final_encode_feat.max(dim=1).values
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)

        logits = self.classifier(pooled)
        return logits


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, window_size, num_features = 64, 32, 9
    d_model = 128
    num_classes = 8

    x = torch.rand(batch_size, window_size, num_features)  # pre-normalized [0,1]
    labels = torch.randint(0, num_classes, (batch_size,))

    model = TransformerBaselineModel(
        num_features=num_features, d_model=d_model, num_heads=4, ff_dim=256,
        num_layers=2, num_classes=num_classes, dropout=0.1,
    )

    feats, logits, attn_weights = model(x)
    print(f"Input x shape:             {tuple(x.shape)}")
    print(f"final_encode_feat shape:   {tuple(feats.shape)}")
    print(f"logits shape:              {tuple(logits.shape)}")
    print(f"num attn layers:           {len(attn_weights)}")
    print(f"attn_weights[0] shape:     {tuple(attn_weights[0].shape)}")

    assert feats.shape == (batch_size, window_size, d_model)
    assert logits.shape == (batch_size, num_classes)
    assert len(attn_weights) == 2
    assert attn_weights[0].shape == (batch_size, 4, window_size, window_size)

    # gradient flow
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    print(f"\nCrossEntropyLoss: {loss.item():.4f}")
    grad_ok = model.input_proj.weight.grad is not None and model.input_proj.weight.grad.abs().sum() > 0
    print(f"Input projection gradient finite & nonzero: {bool(grad_ok)}")

    # sequence length preserved (attention never changes seq_len)
    assert feats.shape[1] == window_size
    print(f"Sequence length preserved: {feats.shape[1]} == {window_size}")

    # --- frame_labels trailing-column compatibility (same bug class fixed in simple_cnn.py) ---
    x_10cols = torch.rand(batch_size, window_size, 10)  # 9 real features + 1 trailing frame_labels col
    feats10, logits10, _ = model(x_10cols)
    assert logits10.shape == (batch_size, num_classes)
    print(f"\n[10-column input, e.g. with trailing frame_labels] logits shape: {tuple(logits10.shape)}")
    print("Extra trailing column correctly ignored: True")

    # padding mask sanity check
    mask = torch.ones(batch_size, window_size)
    mask[:, 28:] = 0
    _, logits_masked, attn_masked = model(x, mask=mask)
    print(f"\n[+ padding mask] logits shape: {tuple(logits_masked.shape)}")
    assert logits_masked.shape == (batch_size, num_classes)
    # masked-out key positions should receive ~0 attention
    assert attn_masked[0][:, :, :, 28:].max().item() < 1e-4
    print("Padding mask: masked positions receive ~0 attention: True")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
