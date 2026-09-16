"""
simple_cnn.py
===============

Baseline model for comparative analysis: a PLAIN 1D-CNN backbone
(no residual/skip connections, no attention, no recurrence) for
CAN-window classification.

This is one of four models being compared in the ablation-style
comparative analysis:
    1. Simple CNN               <-- this file
    2. Traditional Transformer  (vanilla self-attention, no decay/SwiGLU)
    3. Residual CNN + Transformer (hybrid)
    4. Proposed model (Residual CNN local encoder + Bidirectional
       RetNet + SwiGLU)

Design
-------
Input:  (batch, window_size, num_features)   e.g. (batch, 32, 9)
        -- can_id + 8 payload bytes, pre-normalized to [0,1], SAME
        input contract as the proposed model's backbone.

Unlike `PayloadLocalEncoderPerFrame` (kernel_size=1, strictly
per-frame, no mixing across the window dimension -- used inside the
proposed model), this baseline's Conv1D layers use kernel_size > 1
and mix information ACROSS neighboring frames in the window. This is
intentional: a "simple CNN" baseline is expected to do its temporal
modeling purely through convolution over the sequence dimension (the
same idea as HyDL-IDS's Convo Units, Lo et al. 2022), with NO residual
connections (that would be the separate "Residual CNN + Transformer"
baseline) and NO attention/recurrence of any kind.

Padding is set to keep the sequence length constant across layers
(`padding = (kernel_size - 1) // 2`, stride=1), so the output shape
matches (batch, window_size, d_model) -- the same shape convention
used by the proposed model's BidirectionalRetentionEncoder, making
the final Mean+Max Pool + MLP classifier tail identical across all
four compared models for a fair, apples-to-apples comparison.

Limitation (please read before using in the same evaluation loop as
the proposed model): this backbone has no attention mechanism, so it
produces NO `attn_weights` -- `forward()` returns `None` in that slot.
Consequently, `detect_attack_frames` / frame-level localization
evaluation (AUROC/AUPR/Hit@k etc.) is NOT applicable to this baseline.
Only WINDOW-LEVEL classification metrics (accuracy, F1, precision,
recall, confusion matrix) should be compared for this model in your
comparative analysis -- the frame-level localization comparison is
only meaningful for models that produce genuine attention weights
(the proposed model, the Transformer baseline, and the Residual
CNN + Transformer baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Plain 1D-CNN backbone: Conv1D -> BatchNorm -> ReLU -> Dropout, stacked,
# NO residual/skip connections (kept deliberately simple, per its name).
# --------------------------------------------------------------------------- #
class SimpleCNNBackbone(nn.Module):
    """
    A stack of plain Conv1D blocks over the window (time) dimension.

    Each block: Conv1D(kernel_size, stride=1, padding=same) -> BatchNorm1d
                -> ReLU -> Dropout

    kernel_size > 1 lets each layer mix information from NEIGHBORING
    frames in the window (unlike the proposed model's kernel_size=1
    per-frame local encoder) -- this is the plain-CNN way of capturing
    sequential/temporal patterns, with no attention or gating of any
    kind and NO residual connections.

    Args:
        num_features: input feature dimension per frame (e.g. 9 = can_id + 8 bytes)
        d_model:      output channel dimension (kept constant across layers
                      after the first, for a clean, uniform stack)
        num_layers:   number of stacked Conv1D blocks
        kernel_size:  convolution kernel size (must be odd, so that
                      padding=(kernel_size-1)//2 exactly preserves the
                      sequence length with stride=1)
        dropout:      dropout rate applied after each block
    """

    def __init__(
        self,
        num_features: int = 9,
        d_model: int = 128,
        num_layers: int = 4,  # matches proposed model: 2 residual blocks x 2 conv1D each = 4
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd to keep 'same' padding exact"
        padding = (kernel_size - 1) // 2

        blocks = []
        in_channels = num_features
        for _ in range(num_layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, d_model, kernel_size=kernel_size,
                              stride=1, padding=padding),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = d_model  # every subsequent layer: d_model -> d_model

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, num_features)
        Returns:
            (batch, window_size, d_model) -- sequence length preserved
        """
        h = x.transpose(1, 2)  # (batch, num_features, window_size) for Conv1d
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)  # back to (batch, window_size, d_model)
        return h


# --------------------------------------------------------------------------- #
# Full baseline model: backbone -> Mean+Max Pool -> MLP classifier
# (SAME classifier tail as the proposed model, for a fair comparison
# isolating the effect of the backbone architecture alone)
# --------------------------------------------------------------------------- #
class SimpleCNNModel(nn.Module):
    """
    Plain 1D-CNN baseline for CAN-window classification.

        x (B, T, num_features)
            -> SimpleCNNBackbone (plain stacked Conv1D, no residual, no attention)
            -> LayerNorm
            -> Mean Pool + Max Pool -> Concat
            -> MLP classifier
            -> logits (B, num_classes)

    Returns a 3-tuple (final_encode_feat, logits, attn_weights) to stay
    drop-in compatible with the training/evaluation loop used for the
    proposed model -- `attn_weights` is always None here (see module
    docstring: this baseline has no attention, so frame-level
    localization evaluation does not apply to it).
    """

    def __init__(
        self,
        num_features: int = 9,
        d_model: int = 128,
        num_layers: int = 4,  # matches proposed model: 2 residual blocks x 2 conv1D each = 4
        kernel_size: int = 3,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = SimpleCNNBackbone(
            num_features=num_features, d_model=d_model, num_layers=num_layers,
            kernel_size=kernel_size, dropout=dropout,
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
            x:    (batch, window_size, num_features)
            mask: accepted for interface compatibility with the proposed
                  model's forward() signature, but UNUSED here -- a plain
                  Conv1D stack has no notion of attention masking; padded
                  frames simply get convolved like any other frame. If you
                  need masking support, zero out padded frames in `x`
                  before calling this model.
        Returns:
            final_encode_feat: (batch, window_size, d_model)
            logits:            (batch, num_classes)
            attn_weights:       None (no attention in this baseline)
        """
        x = x[:, :, 0:9]
        h = self.backbone(x)
        final_encode_feat = self.final_ln(h)

        mean_pooled = final_encode_feat.mean(dim=1)
        max_pooled = final_encode_feat.max(dim=1).values
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)

        logits = self.classifier(pooled)
        return logits

if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, window_size, num_features = 64, 32, 9
    d_model = 128
    num_classes = 8

    x = torch.rand(batch_size, window_size, num_features)  # pre-normalized [0,1]
    labels = torch.randint(0, num_classes, (batch_size,))

    model = SimpleCNNModel(
        num_features=num_features, d_model=d_model, num_layers=4,
        kernel_size=3, num_classes=num_classes, dropout=0.1,
    )

    feats, logits, attn_weights = model(x)
    print(f"Input x shape:             {tuple(x.shape)}")
    print(f"final_encode_feat shape:   {tuple(feats.shape)}")
    print(f"logits shape:              {tuple(logits.shape)}")
    print(f"attn_weights:              {attn_weights}")

    assert feats.shape == (batch_size, window_size, d_model)
    assert logits.shape == (batch_size, num_classes)
    assert attn_weights is None

    # gradient flow
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    print(f"\nCrossEntropyLoss: {loss.item():.4f}")
    grad_ok = model.backbone.blocks[0][0].weight.grad is not None and \
        model.backbone.blocks[0][0].weight.grad.abs().sum() > 0
    print(f"First Conv1D layer gradient finite & nonzero: {bool(grad_ok)}")

    # sequence length preserved across layers (sanity check for the 'same' padding)
    with torch.no_grad():
        h_only_backbone = model.backbone(x)
    assert h_only_backbone.shape[1] == window_size, (
        f"sequence length changed: expected {window_size}, got {h_only_backbone.shape[1]}"
    )
    print(f"Sequence length preserved through backbone: {h_only_backbone.shape[1]} == {window_size}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    # kernel_size sweep sanity (must all preserve window_size)
    print("\n[kernel_size sweep -- sequence length must stay 32]")
    for k in (1, 3, 5, 7):
        m = SimpleCNNBackbone(num_features=num_features, d_model=64, num_layers=2, kernel_size=k)
        out = m(x)
        print(f"  kernel_size={k}: output seq_len={out.shape[1]}")
        assert out.shape[1] == window_size
