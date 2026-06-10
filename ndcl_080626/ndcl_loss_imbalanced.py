"""
Single-domain NDCL loss adapted for highly imbalanced datasets.

Main changes compared with the paper-oriented NDCL implementation:
1) No domain-alignment term: this is for one-domain long-tailed data.
2) Class-balanced CE + within-class hard-example reweighting.
3) Tail-centric hard-negative augmentation: tail classes receive more synthetic
   hard negatives; the majority class receives zero or very few.
4) Negative-dominant contrastive loss is computed in prediction space, so it can
   work with a normal classifier output `logits` without requiring a special
   `get_logits_from_embeddings()` method.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleDomainImbalancedNDCL(nn.Module):
    """Negative-dominant loss for one-domain imbalanced classification.

    Args:
        num_classes: number of classes.
        class_counts: training-set class counts, shape [num_classes]. If given,
            it is used for class-balanced CE, majority-class detection, and
            inverse-frequency hard-negative augmentation.
        alpha: weight of the negative-dominant contrastive term.
        beta_mix: Beta(rho, rho) parameter for hard-negative mixup.
        max_aug_per_class: maximum generated hard negatives for the rarest class.
        min_aug_per_class: base hard negatives for non-majority classes.
        majority_aug_ratio: multiply augmentation count of the majority class by
            this value. Keep 0.0 for a dataset dominated by one normal/majority
            class.
        inv_aug_power: controls how strongly augmentation count grows for tail
            classes. 0.5 is usually safer than 1.0.
        low_pos_frac: fraction of low-confidence positives used for mining.
        hard_neg_frac: fraction of high-confidence negatives used for mining.
        cb_beta: effective-number beta for class-balanced CE.
        use_class_balanced_ce: use class-balanced CE weights.
        use_within_class_hard_ce: paper-style within-class loss reweighting.
        warmup_ce_epochs: during the first N epochs, only CE is optimized; this
            avoids unstable hard-negative mining from noisy early predictions.
        eps: numerical stability constant.
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        alpha: float = 0.1,
        beta_mix: float = 0.5,
        max_aug_per_class: int = 32,
        min_aug_per_class: int = 4,
        majority_aug_ratio: float = 0.0,
        inv_aug_power: float = 0.5,
        low_pos_frac: float = 0.25,
        hard_neg_frac: float = 0.25,
        cb_beta: float = 0.9999,
        use_class_balanced_ce: bool = True,
        use_within_class_hard_ce: bool = True,
        warmup_ce_epochs: int = 5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta_mix = float(beta_mix)
        self.max_aug_per_class = int(max_aug_per_class)
        self.min_aug_per_class = int(min_aug_per_class)
        self.majority_aug_ratio = float(majority_aug_ratio)
        self.inv_aug_power = float(inv_aug_power)
        self.low_pos_frac = float(low_pos_frac)
        self.hard_neg_frac = float(hard_neg_frac)
        self.cb_beta = float(cb_beta)
        self.use_class_balanced_ce = bool(use_class_balanced_ce)
        self.use_within_class_hard_ce = bool(use_within_class_hard_ce)
        self.warmup_ce_epochs = int(warmup_ce_epochs)
        self.eps = float(eps)

        if class_counts is None:
            counts = torch.ones(self.num_classes, dtype=torch.float32)
        else:
            counts = torch.as_tensor(class_counts, dtype=torch.float32).clone()
            if counts.numel() != self.num_classes:
                raise ValueError(
                    f"class_counts must have {self.num_classes} elements, "
                    f"but got {counts.numel()}"
                )
            counts = torch.clamp(counts, min=1.0)

        self.register_buffer("class_counts", counts)
        self.register_buffer("class_weights", self._effective_number_weights(counts))
        self.majority_class = int(torch.argmax(counts).item())
        self.minority_class = int(torch.argmin(counts).item())

    def _effective_number_weights(self, counts: torch.Tensor) -> torch.Tensor:
        beta = self.cb_beta
        if beta <= 0.0 or beta >= 1.0:
            weights = 1.0 / counts
        else:
            effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=counts.dtype), counts)
            weights = (1.0 - beta) / torch.clamp(effective_num, min=self.eps)
        weights = weights / torch.clamp(weights.mean(), min=self.eps)
        return weights.float()

    def _classwise_hard_ce(self, per_sample_ce: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Paper-style CE: average over active classes, then weight hard samples inside each class."""
        active_classes = torch.unique(labels)
        loss = per_sample_ce.new_tensor(0.0)
        for k in active_classes:
            li = per_sample_ce[labels == k]
            if li.numel() == 0:
                continue
            if self.use_within_class_hard_ce:
                # Higher CE within the same class receives larger weight.
                weight = F.softmax(li.detach(), dim=0)
                loss = loss + torch.sum(li * weight)
            else:
                loss = loss + li.mean()
        return loss / torch.clamp(active_classes.numel(), min=1)

    def _num_aug_for_class(self, k: int) -> int:
        """Tail-centric augmentation count, based on global class counts."""
        counts = self.class_counts.detach().cpu().float()
        max_count = float(counts.max().item())
        min_count = float(counts.min().item())
        count_k = float(counts[k].item())

        if k == self.majority_class:
            return int(round(self.min_aug_per_class * self.majority_aug_ratio))

        if max_count <= min_count:
            return self.min_aug_per_class

        rarity = (max_count / max(count_k, 1.0)) ** self.inv_aug_power
        max_rarity = (max_count / max(min_count, 1.0)) ** self.inv_aug_power
        score = (rarity - 1.0) / max(max_rarity - 1.0, self.eps)
        n_aug = self.min_aug_per_class + score * (self.max_aug_per_class - self.min_aug_per_class)
        return int(np.clip(round(n_aug), self.min_aug_per_class, self.max_aug_per_class))

    def _sample_mix_lambda(self) -> float:
        lam = float(np.random.beta(self.beta_mix, self.beta_mix))
        # The synthetic sample is used as a negative of class k, so keep it closer
        # to the hard negative than to the positive.
        lam = min(lam, 1.0 - lam)
        return float(np.clip(lam, 0.05, 0.50))

    @torch.no_grad()
    def _build_hard_negative_probs(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[int, int]]:
        """Create class-wise synthetic hard negatives in prediction space.

        For class k:
        - positives are true samples y=k with low p_k;
        - negatives are samples y!=k with high p_k;
        - p_hat = lam * p_pos + (1-lam) * p_neg.

        Returned aug_y stores the class k for which p_hat is considered a hard
        negative, not the semantic hard label of p_hat.
        """
        device = probs.device
        active_classes = torch.unique(labels)
        if active_classes.numel() <= 1:
            return None, None, {}

        aug_probs = []
        aug_y = []
        aug_stats: Dict[int, int] = {}

        for k_tensor in active_classes:
            k = int(k_tensor.item())
            n_aug = self._num_aug_for_class(k)
            if n_aug <= 0:
                aug_stats[k] = 0
                continue

            pos_idx = torch.where(labels == k)[0]
            neg_idx = torch.where(labels != k)[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                aug_stats[k] = 0
                continue

            # Low-confidence positives: bottom p_k among true class-k samples.
            pos_scores = probs[pos_idx, k]
            pos_order = torch.argsort(pos_scores, descending=False)
            n_low_pos = max(1, int(np.ceil(pos_idx.numel() * self.low_pos_frac)))
            pos_pool = pos_idx[pos_order[:n_low_pos]]

            # High-confidence negatives: top p_k among non-k samples.
            neg_scores = probs[neg_idx, k]
            neg_order = torch.argsort(neg_scores, descending=True)
            n_hard_neg = max(1, int(np.ceil(neg_idx.numel() * self.hard_neg_frac)))
            # Also ensure enough negatives for the requested augmentations.
            n_hard_neg = min(neg_idx.numel(), max(n_hard_neg, int(np.ceil(n_aug / max(1, pos_pool.numel())))))
            neg_pool = neg_idx[neg_order[:n_hard_neg]]

            generated = []
            for _ in range(n_aug):
                p_id = pos_pool[torch.randint(0, pos_pool.numel(), (1,), device=device)]
                n_id = neg_pool[torch.randint(0, neg_pool.numel(), (1,), device=device)]
                lam = self._sample_mix_lambda()
                p_hat = lam * probs[p_id] + (1.0 - lam) * probs[n_id]
                p_hat = p_hat.squeeze(0)
                # p_hat is already a probability vector because it is a convex
                # combination, but renormalize for numerical safety.
                p_hat = p_hat / torch.clamp(p_hat.sum(), min=self.eps)
                generated.append(p_hat)

            if generated:
                generated = torch.stack(generated, dim=0)
                aug_probs.append(generated)
                aug_y.append(torch.full((generated.size(0),), k, dtype=torch.long, device=device))
                aug_stats[k] = int(generated.size(0))
            else:
                aug_stats[k] = 0

        if not aug_probs:
            return None, None, aug_stats

        return torch.cat(aug_probs, dim=0), torch.cat(aug_y, dim=0), aug_stats

    def _negative_dominant_contrastive(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        aug_probs: Optional[torch.Tensor],
        aug_y: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """InfoNCE-like negative-dominant contrastive loss in prediction space.

        For anchor i of class k:
        numerator   = mean dissimilarity(anchor, negatives of class k)
        denominator = sum  dissimilarity(anchor, all non-self candidates)

        The loss is averaged class-wise to avoid majority-class domination.
        """
        device = probs.device
        probs_norm = F.normalize(probs, p=2, dim=1)
        aug_norm = F.normalize(aug_probs, p=2, dim=1) if aug_probs is not None else None

        active_classes = torch.unique(labels)
        class_losses = []

        for k_tensor in active_classes:
            k = int(k_tensor.item())
            anchor_indices = torch.where(labels == k)[0]
            if anchor_indices.numel() == 0:
                continue

            anchor_losses = []
            for i in anchor_indices:
                anchor = probs_norm[i : i + 1]

                real_pos_mask = labels == k
                real_pos_mask[i] = False
                real_neg_mask = labels != k

                candidates = []
                negatives = []

                if real_pos_mask.any():
                    candidates.append(probs_norm[real_pos_mask])
                if real_neg_mask.any():
                    real_negs = probs_norm[real_neg_mask]
                    candidates.append(real_negs)
                    negatives.append(real_negs)
                if aug_norm is not None and aug_y is not None:
                    aug_mask = aug_y == k
                    if aug_mask.any():
                        aug_negs = aug_norm[aug_mask]
                        candidates.append(aug_negs)
                        negatives.append(aug_negs)

                if len(candidates) == 0 or len(negatives) == 0:
                    continue

                cand = torch.cat(candidates, dim=0)
                neg = torch.cat(negatives, dim=0)

                dis_cand = 1.0 - F.cosine_similarity(anchor, cand, dim=1)
                dis_neg = 1.0 - F.cosine_similarity(anchor, neg, dim=1)

                numerator = dis_neg.mean()
                denominator = dis_cand.sum()
                ratio = numerator / torch.clamp(denominator, min=self.eps)
                anchor_losses.append(-torch.log(torch.clamp(ratio, min=self.eps)))

            if anchor_losses:
                class_losses.append(torch.stack(anchor_losses).mean())

        if not class_losses:
            return probs.new_tensor(0.0)
        return torch.stack(class_losses).mean()

    def forward(
        self,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        return_dict: bool = False,
        *args,
        **kwargs,
    ):
        """Compute loss.

        Preferred call:
            criterion(logits=logits, labels=labels, epoch=epoch, return_dict=True)

        Backward-compatible call with the old signature is also accepted:
            criterion(embeddings, labels, logits, opt=opt)
        """
        # Backward compatibility with old code:
        #     criterion(images_or_features, labels, logits, opt=opt)
        # Because of Python positional binding, this arrives here as
        #     logits=images_or_features, labels=labels, embeddings=logits.
        if (
            embeddings is not None
            and torch.is_tensor(embeddings)
            and embeddings.dim() == 2
            and logits is not None
            and torch.is_tensor(logits)
            and logits.dim() != 2
        ):
            logits = embeddings
        elif len(args) >= 3:
            embeddings, labels, logits = args[:3]

        if labels is None or logits is None:
            raise ValueError("Both logits and labels are required.")
        if not torch.is_tensor(logits) or logits.dim() != 2:
            raise ValueError(
                "logits must be a [batch, num_classes] tensor. "
                "Use criterion(logits=logits, labels=labels, ...) in the training loop."
            )

        labels = labels.long()
        device = logits.device
        class_weights = self.class_weights.to(device) if self.use_class_balanced_ce else None

        per_sample_ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
        loss_ce = self._classwise_hard_ce(per_sample_ce, labels)

        use_con = epoch is None or epoch > self.warmup_ce_epochs
        if use_con and self.alpha > 0.0:
            probs = F.softmax(logits, dim=1)
            # Mining is based on current probabilities but detached, preventing
            # unstable second-order-like gradients through synthetic negatives.
            aug_probs, aug_y, aug_stats = self._build_hard_negative_probs(probs.detach(), labels)
            loss_con = self._negative_dominant_contrastive(probs, labels, aug_probs, aug_y)
        else:
            loss_con = logits.new_tensor(0.0)
            aug_stats = {}

        loss = loss_ce + self.alpha * loss_con

        if return_dict:
            return {
                "loss": loss,
                "loss_ce": loss_ce.detach(),
                "loss_con": loss_con.detach(),
                "alpha": torch.tensor(self.alpha, device=device),
                "majority_class": torch.tensor(self.majority_class, device=device),
                "minority_class": torch.tensor(self.minority_class, device=device),
                "aug_total": torch.tensor(sum(aug_stats.values()) if aug_stats else 0, device=device),
            }
        return loss


# Alias for easier drop-in replacement if your training script still imports NDCL.
NDCL = SingleDomainImbalancedNDCL
