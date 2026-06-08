"""
Separated loss for one-view KCL/TSC.

Use with `KCLOneViewModel` from `builder_oneview_model.py`.

Example:
    output = model(images, labels)
    loss_output = criterion(output["logits"], output["mask_class"], output["mask_target"])
    loss = loss_output["loss"]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class KCLTSCCriterion(nn.Module):
    """
    Separated criterion for one-view KCL/TSC.

    Args:
        target_weight:
            Weight for target loss. Same role as `tw` in the original builder.py.
        original_style_weighting:
            If True, class and target terms are divided by total positive count per anchor,
            close to the original implementation.
            If False, class positives and target positives are averaged separately.
        eps:
            Numerical stability.
    """

    def __init__(
        self,
        target_weight: float = 1.0,
        original_style_weighting: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.target_weight = float(target_weight)
        self.original_style_weighting = bool(original_style_weighting)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        mask_class: torch.Tensor,
        mask_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits:
                [B, K] or [B, K + C*tr].
            mask_class:
                Same shape as logits. 1 means same-class queue positive.
            mask_target:
                Same shape as logits. 1 means assigned target positive.

        Returns:
            dict with:
                loss
                loss_class
                loss_target
                num_class_pos
                num_target_pos
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be [B, N], got {logits.shape}.")
        if mask_class.shape != logits.shape:
            raise ValueError(f"mask_class shape {mask_class.shape} must match logits {logits.shape}.")
        if mask_target.shape != logits.shape:
            raise ValueError(f"mask_target shape {mask_target.shape} must match logits {logits.shape}.")

        mask_class = mask_class.float()
        mask_target = mask_target.float()
        mask_total = mask_class + mask_target

        log_prob = F.log_softmax(logits, dim=1)

        class_count = mask_class.sum(dim=1)
        target_count = mask_target.sum(dim=1)
        total_count = mask_total.sum(dim=1)

        zero = logits.sum() * 0.0

        if self.original_style_weighting:
            # Original style: divide each term by total positives of that anchor.
            valid = total_count > 0
            denom = total_count.clamp_min(1.0)

            loss_class_anchor = -((mask_class * log_prob).sum(dim=1) / denom)
            loss_target_anchor = -((mask_target * log_prob).sum(dim=1) / denom)

            if valid.any():
                loss_class = loss_class_anchor[valid].mean()
                loss_target = loss_target_anchor[valid].mean()
            else:
                loss_class = zero
                loss_target = zero
        else:
            # Cleaner separated averaging.
            valid_class = class_count > 0
            valid_target = target_count > 0

            if valid_class.any():
                loss_class = -((mask_class * log_prob).sum(dim=1)[valid_class] / class_count[valid_class]).mean()
            else:
                loss_class = zero

            if valid_target.any():
                loss_target = -((mask_target * log_prob).sum(dim=1)[valid_target] / target_count[valid_target]).mean()
            else:
                loss_target = zero

        loss_target = loss_target * self.target_weight
        loss = loss_class + loss_target

        return {
            "loss": loss,
            "loss_class": loss_class.detach(),
            "loss_target": loss_target.detach(),
            "num_class_pos": class_count.detach(),
            "num_target_pos": target_count.detach(),
        }
