"""
Negative-Dominant Contrastive Learning loss for embedding-space training.

This file defines a loss function inspired by the NDCL paper, but adapted for a
single imbalanced dataset and for embedding features instead of prediction-space
probabilities.

Main use case:
    embeddings = projection_head(backbone(x))  # [B, D]
    logits = classifier(backbone(x))           # [B, C]

    ce_loss = F.cross_entropy(logits, y, weight=class_weights)
    nd_loss, nd_stats = ndcl_loss(embeddings, y)
    loss = ce_loss + alpha * nd_loss

Why this loss is useful for class imbalance:
    Standard supervised contrastive learning is positive-dominant: it mainly
    pulls same-class samples together. For minority classes, positives are few.
    This loss is negative-dominant: it mainly uses abundant different-class
    samples to push confusing negatives away from each anchor.

Important practical note:
    Contrastive learning is mini-batch dependent. If a minority class does not
    appear in a mini-batch, no anchor from that class is updated by this loss.
    A class-aware/balanced batch sampler is strongly recommended.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeDominantEmbeddingLoss(nn.Module):
    """
    Negative-Dominant Contrastive Loss for embedding space.

    Given embedding vectors z_i and labels y_i, the loss uses cosine
    dissimilarity:

        d_ij = 1 - cos(z_i, z_j)

    and uses negative pairs in the numerator:

        L_i = -log [ sum_{n in N(i)} q_in exp(d_in / tau)
                    / sum_{a != i} exp(d_ia / tau) ]

    where:
        N(i) = samples with label different from y_i
        q_in = hard-negative weight, larger for negatives closer to anchor i

    The loss also supports class-balanced anchor weighting, which makes minority
    anchors contribute more to the final loss.

    Args:
        class_counts:
            Optional tensor/list with dataset-level sample count per class.
            Example: [100000, 3000, 1500, 500, 300]. If provided, anchor weights
            are computed from these counts. If None, inverse sqrt in-batch class
            frequency is used.

        temperature:
            Temperature for the dissimilarity logits exp(d_ij / temperature).
            Lower values make the contrastive objective sharper.

        hard_temperature:
            Temperature for hard-negative weighting computed from cosine
            similarity. Lower values focus more strongly on the closest
            negatives.

        beta:
            Hyperparameter for effective-number class weighting.
            Typical values: 0.99, 0.999, 0.9999.

        use_effective_number:
            If True, use effective-number class weighting:
                w_k = 1 / ((1 - beta^n_k) / (1 - beta))
            If False, use simple inverse-frequency weighting:
                w_k = 1 / n_k

        require_positive:
            If True, an anchor is used only if at least one same-class positive
            and one negative exist in the mini-batch. This is recommended because
            the loss can become weak/degenerate when a class has only one sample
            in the batch.

        eps:
            Small value for numerical stability.
    """

    def __init__(
        self,
        class_counts: Optional[torch.Tensor] = None,
        temperature: float = 0.2,
        hard_temperature: float = 0.1,
        beta: float = 0.9999,
        use_effective_number: bool = True,
        require_positive: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if hard_temperature <= 0:
            raise ValueError("hard_temperature must be positive.")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta should be in (0, 1), e.g., 0.9999.")

        self.temperature = float(temperature)
        self.hard_temperature = float(hard_temperature)
        self.beta = float(beta)
        self.use_effective_number = bool(use_effective_number)
        self.require_positive = bool(require_positive)
        self.eps = float(eps)

        if class_counts is not None:
            counts = torch.as_tensor(class_counts, dtype=torch.float32)
            if counts.ndim != 1:
                raise ValueError("class_counts must be a 1-D tensor/list.")
            if torch.any(counts <= 0):
                raise ValueError("All class_counts must be positive.")

            if self.use_effective_number:
                effective_num = (1.0 - torch.pow(self.beta, counts)) / (1.0 - self.beta)
                class_weight = 1.0 / effective_num.clamp_min(self.eps)
            else:
                class_weight = 1.0 / counts.clamp_min(1.0)

            # Normalize to keep the mean loss scale stable.
            class_weight = class_weight / class_weight.mean().clamp_min(self.eps)
            self.register_buffer("class_weight", class_weight)
        else:
            self.register_buffer("class_weight", None)

    def _anchor_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Return one scalar weight for each anchor in the mini-batch."""
        labels = labels.long()
        device = labels.device

        if self.class_weight is not None:
            if labels.max().item() >= self.class_weight.numel():
                raise ValueError(
                    "labels contain a class index larger than class_counts length."
                )
            return self.class_weight.to(device)[labels]

        # Fallback: in-batch class balancing. Inverse sqrt is less aggressive
        # than inverse frequency and is usually more stable.
        num_classes = int(labels.max().item()) + 1
        counts = torch.bincount(labels, minlength=num_classes).float().to(device)
        weights = 1.0 / torch.sqrt(counts[labels].clamp_min(1.0))
        weights = weights / weights.mean().clamp_min(self.eps)
        return weights

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute negative-dominant embedding contrastive loss.

        Args:
            embeddings:
                Tensor of shape [B, D]. Usually the output of a projection head
                or backbone embedding layer.
            labels:
                Tensor of shape [B]. Class labels.

        Returns:
            loss:
                Scalar tensor.
            stats:
                Dictionary with useful diagnostic values.
        """
        # print("NDCL forward: embeddings shape:", embeddings.shape, "labels shape:", labels.shape)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [B, D].")
        if labels.ndim != 1:
            raise ValueError("labels must have shape [B].")
        if embeddings.size(0) != labels.size(0):
            raise ValueError("embeddings and labels must have the same batch size.")

        device = embeddings.device
        labels = labels.to(device=device, dtype=torch.long)
        batch_size = embeddings.size(0)

        if batch_size <= 1:
            zero = embeddings.sum() * 0.0
            return zero, {
                "ndcl_loss": 0.0,
                "valid_anchors": 0.0,
                "mean_pos_per_anchor": 0.0,
                "mean_neg_per_anchor": 0.0,
            }

        # Normalize for cosine similarity.
        z = F.normalize(embeddings, dim=1)

        # Cosine similarity and dissimilarity matrices: [B, B]
        sim = torch.matmul(z, z.T)
        dis = 1.0 - sim

        eye = torch.eye(batch_size, dtype=torch.bool, device=device)
        same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1))

        pos_mask = same_class & (~eye)
        neg_mask = (~same_class) & (~eye)
        all_mask = ~eye

        pos_count = pos_mask.sum(dim=1)
        neg_count = neg_mask.sum(dim=1)

        if self.require_positive:
            valid = (pos_count > 0) & (neg_count > 0)
        else:
            valid = neg_count > 0

        if valid.sum() == 0:
            zero = embeddings.sum() * 0.0
            return zero, {
                "ndcl_loss": 0.0,
                "valid_anchors": 0.0,
                "mean_pos_per_anchor": float(pos_count.float().mean().detach().cpu()),
                "mean_neg_per_anchor": float(neg_count.float().mean().detach().cpu()),
            }

        # Dissimilarity logits. Larger values mean farther samples.
        dis_logits = dis / self.temperature

        # Denominator: all non-self pairs.
        log_den = torch.logsumexp(
            dis_logits.masked_fill(~all_mask, -1e9),
            dim=1,
        )

        # Hard-negative weight q_in.
        # Close negatives have high cosine similarity, so they receive larger q.
        # Detach q so it behaves like a mining weight, not a second objective.
        hard_logits = sim / self.hard_temperature
        hard_logits = hard_logits.masked_fill(~neg_mask, -1e9)
        hard_weight = F.softmax(hard_logits, dim=1).detach()
        hard_weight = hard_weight * neg_mask.float()
        hard_weight = hard_weight / hard_weight.sum(dim=1, keepdim=True).clamp_min(self.eps)

        # Numerator: hard-negative weighted exp(d_in / tau).
        log_q = torch.log(hard_weight.clamp_min(self.eps))
        log_num = torch.logsumexp(
            (dis_logits + log_q).masked_fill(~neg_mask, -1e9),
            dim=1,
        )

        loss_per_anchor = -(log_num - log_den)

        anchor_weight = self._anchor_weights(labels)
        loss = (loss_per_anchor[valid] * anchor_weight[valid]).sum()
        loss = loss / anchor_weight[valid].sum().clamp_min(self.eps)

        stats = {
            "ndcl_loss": float(loss.detach().cpu()),
            "valid_anchors": float(valid.sum().detach().cpu()),
            "mean_pos_per_anchor": float(pos_count.float().mean().detach().cpu()),
            "mean_neg_per_anchor": float(neg_count.float().mean().detach().cpu()),
        }
        return loss, stats


class BalancedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with class-balanced weights.

    This is optional, but commonly useful with the negative-dominant embedding
    loss. The NDCL term handles boundary separation, while balanced CE handles
    class-prior bias in the classifier.
    """

    def __init__(
        self,
        class_counts: torch.Tensor,
        beta: float = 0.9999,
        use_effective_number: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1:
            raise ValueError("class_counts must be a 1-D tensor/list.")
        if torch.any(counts <= 0):
            raise ValueError("All class_counts must be positive.")

        if use_effective_number:
            effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
            weights = 1.0 / effective_num.clamp_min(eps)
        else:
            weights = 1.0 / counts.clamp_min(1.0)

        weights = weights / weights.mean().clamp_min(eps)
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels.long(), weight=self.weights.to(logits.device))


def negative_dominant_embedding_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_counts: Optional[torch.Tensor] = None,
    temperature: float = 0.2,
    hard_temperature: float = 0.1,
    beta: float = 0.9999,
    use_effective_number: bool = True,
    require_positive: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Functional wrapper for quick use.

    For repeated training iterations, prefer constructing
    NegativeDominantEmbeddingLoss once and reusing it, instead of calling this
    wrapper every iteration.
    """
    criterion = NegativeDominantEmbeddingLoss(
        class_counts=class_counts,
        temperature=temperature,
        hard_temperature=hard_temperature,
        beta=beta,
        use_effective_number=use_effective_number,
        require_positive=require_positive,
    ).to(embeddings.device)
    return criterion(embeddings, labels)


if __name__ == "__main__":
    # Small sanity check.
    torch.manual_seed(0)

    B, D, C = 16, 128, 4
    embeddings = torch.randn(B, D)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    class_counts = torch.tensor([10000, 1000, 200, 50])

    ndcl = NegativeDominantEmbeddingLoss(class_counts=class_counts)
    loss, stats = ndcl(embeddings, labels)

    print("loss:", float(loss))
    print("stats:", stats)
