"""
One-view PaCo loss.

Drop-in replacement for:
    PaCo/Full-ImageNet/losses.py

Main change from original PaCoLoss:
    original expects features = [q, k, queue], so B = (N - K) // 2
    one-view expects features = [q, queue], so B = sup_logits.shape[0]

The loss still combines:
    1) sample-to-class-center contrast through sup_logits
    2) sample-to-sample contrast through features and labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaCoLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        supt=1.0,
        temperature=1.0,
        base_temperature=None,
        K=128,
        num_classes=1000,
    ):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def forward(self, features, labels=None, sup_logits=None):
        """
        Args:
            features:   [B + K, D], concatenation of [q, queue]
            labels:     [B + K], labels for [q, queue]
            sup_logits: [B, C], classifier/class-center logits for current batch q

        Returns:
            scalar PaCo loss
        """
        if labels is None:
            raise ValueError("labels must not be None")
        if sup_logits is None:
            raise ValueError("sup_logits must not be None")

        device = features.device
        batch_size = sup_logits.shape[0]  # one-view: current anchors are features[:B]

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features and labels must have same first dimension, got "
                f"{features.shape[0]} and {labels.shape[0]}"
            )
        if features.shape[0] < batch_size:
            raise ValueError("features must contain at least the current batch anchors")
        if sup_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"sup_logits second dimension must equal num_classes={self.num_classes}, "
                f"got {sup_logits.shape[1]}"
            )

        labels = labels.contiguous().view(-1, 1)

        # Positive sample mask: same label between anchors and all features.
        sample_pos_mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # Sample-to-sample logits: anchors q against [q, queue].
        anchor_dot_contrast = torch.matmul(features[:batch_size], features.T) / self.temperature

        # Add sample-to-class-center logits at the front.
        # sup_logits is similarity/logit between each anchor and each class center.
        anchor_dot_contrast = torch.cat((sup_logits / self.supt, anchor_dot_contrast), dim=1)

        # Numerical stability.
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Remove self-contrast for sample-to-sample part.
        logits_mask = torch.ones_like(sample_pos_mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        sample_pos_mask = sample_pos_mask * logits_mask

        # Correct class center is always a positive.
        center_pos_mask = F.one_hot(
            labels[:batch_size].view(-1),
            num_classes=self.num_classes,
        ).float().to(device)

        # Final positive mask:
        # [class centers | sample candidates]
        pos_mask = torch.cat(
            (center_pos_mask * self.beta, sample_pos_mask * self.alpha),
            dim=1,
        )

        # Denominator mask:
        # all class centers are denominator candidates;
        # sample candidates are weighted by gamma and exclude self.
        denom_mask = torch.cat(
            (
                torch.ones(batch_size, self.num_classes, device=device),
                self.gamma * logits_mask,
            ),
            dim=1,
        )

        exp_logits = torch.exp(logits) * denom_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-12)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# Optional explicit alias if you prefer clearer naming.
OneViewPaCoLoss = PaCoLoss
