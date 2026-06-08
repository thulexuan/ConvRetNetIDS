import math
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


class SingleDomainNDCLPaper(nn.Module):
    """
    Single-domain NDCL implementation close to the paper.

    Original NDCL paper:
        L = Lce + alpha * Lcon + beta * Lconst

    Your setting:
        only one domain
        => remove Lconst

    Final:
        L = Lce + alpha * Lcon

    Paper-style parts implemented:
        1. Class-wise reweighted CE
        2. Negative-dominant contrastive loss in prediction space
        3. Hard negative augmentation by mixup:
            x_hat = lambda * x_positive_low_conf
                  + (1 - lambda) * x_negative_high_conf
    """

    def __init__(
        self,
        num_classes: int = 10,
        alpha: float = 0.1,
        rho: float = 0.5,
        low_positive_ratio: float = 0.25,
        base_num_aug: int = 4,
        max_aug_per_class: int = 8,
        class_counts: Optional[torch.Tensor] = None,
        use_dataset_inverse_aug: bool = True,
        eps: float = 1e-8,
        logits_index: int = -1,
    ):
        """
        Args:
            num_classes:
                Number of classes.

            alpha:
                Weight for negative-dominant contrastive loss.

            rho:
                Beta distribution parameter in lambda ~ Beta(rho, rho).

            low_positive_ratio:
                Ratio for selecting low-confidence positives.
                Paper appendix gives an example using bottom quarter.

            base_num_aug:
                Base number of augmented hard negatives per class.

            max_aug_per_class:
                Maximum augmented hard negatives per class.
                This prevents rare classes from creating too many augmented samples.

            class_counts:
                Tensor/list with global training count per class.
                If provided, number of augmented negatives is inversely proportional
                to class frequency, close to paper Appendix C.

            use_dataset_inverse_aug:
                If True and class_counts is provided:
                    n_hat_k proportional to max_count / count_k.
                If False:
                    every class uses base_num_aug.

            eps:
                Numerical stability.

            logits_index:
                If model returns tuple/list, this index is used as logits.
                Example:
                    model(images) -> logits
                    model(images) -> (features, logits)
        """
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.rho = rho
        self.low_positive_ratio = low_positive_ratio
        self.base_num_aug = base_num_aug
        self.max_aug_per_class = max_aug_per_class
        self.use_dataset_inverse_aug = use_dataset_inverse_aug
        self.eps = eps
        self.logits_index = logits_index

        if class_counts is not None:
            class_counts = torch.as_tensor(class_counts, dtype=torch.float32)
            if class_counts.numel() != num_classes:
                raise ValueError("class_counts must have length == num_classes")
            self.register_buffer("class_counts", class_counts)
        else:
            self.class_counts = None

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        This forward directly receives model, because paper-style hard negative
        augmentation requires:
            1. forward original images
            2. select hard negatives based on prediction confidence
            3. create x_hat in input space
            4. forward x_hat again
            5. compute Lcon
        """

        labels = labels.long().view(-1)

        # 1. Forward original batch
        original_output = model(images)
        logits = self._extract_logits(original_output)

        # 2. Class-wise reweighted CE
        loss_ce = self.class_wise_reweighted_ce(logits, labels)

        # 3. Build paper-style hard negative mixup samples
        with torch.no_grad():
            probs_for_selection = F.softmax(logits, dim=1)

        aug_images, aug_negative_for_class = self.build_hard_negative_mixup(
            images=images,
            labels=labels,
            probs=probs_for_selection,
        )

        # 4. Forward augmented hard negatives
        if aug_images is not None and aug_images.size(0) > 0:
            aug_output = model(aug_images)
            aug_logits = self._extract_logits(aug_output)
        else:
            aug_logits = None
            aug_negative_for_class = None

        # 5. Negative-dominant contrastive loss
        loss_con = self.negative_dominant_contrastive_loss(
            logits=logits,
            labels=labels,
            aug_logits=aug_logits,
            aug_negative_for_class=aug_negative_for_class,
        )

        loss_total = loss_ce + self.alpha * loss_con

        loss_dict = {
            "loss_total": loss_total.detach(),
            "loss_ce_reweighted": loss_ce.detach(),
            "loss_nd_contrastive": loss_con.detach(),
            "num_augmented_negatives": torch.tensor(
                0 if aug_images is None else aug_images.size(0),
                device=images.device,
                dtype=torch.float32,
            ),
        }

        return loss_total, loss_dict

    def _extract_logits(self, output: TensorOrTuple) -> torch.Tensor:
        """
        Supports:
            model(images) -> logits
            model(images) -> (features, logits)
            model(images) -> [features, logits]
        """
        if isinstance(output, (tuple, list)):
            return output[self.logits_index]
        return output

    def class_wise_reweighted_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mini-batch version of paper Eq. (4):

            Lce = 1/K sum_k sum_i omega_i^k * CE(p_i^k, k)

            omega_i^k = exp(CE_i) / sum_j exp(CE_j)

        In mini-batch training, we average only over classes present in batch.
        """

        ce_each = F.cross_entropy(logits, labels, reduction="none")

        present_classes = labels.unique()
        total = logits.new_tensor(0.0)

        for k in present_classes:
            cls_mask = labels == k
            ce_k = ce_each[cls_mask]

            # Same idea as paper: high CE samples receive higher weight.
            weights = F.softmax(ce_k, dim=0)
            total = total + torch.sum(weights * ce_k)

        return total / max(len(present_classes), 1)

    def build_hard_negative_mixup(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Paper-style hard negative augmentation.

        For each class k:
            1. Rank all batch samples by p_k.
            2. X_k^low:
                label == k, low p_k
                uncertain positives.
            3. X_k-^high:
                label != k, high p_k
                hard negatives.
            4. Mix:
                x_hat = lambda * x_k_positive_low
                      + (1 - lambda) * x_k_negative_high

        Returns:
            aug_images:
                [M, C, H, W] or [M, ...]
            aug_negative_for_class:
                [M]
                aug_negative_for_class[m] = k means this augmented sample
                is used as a refined negative for class k.
        """

        device = images.device
        aug_list = []
        aug_neg_class_list = []

        for k in range(self.num_classes):
            pos_idx = torch.where(labels == k)[0]
            neg_idx = torch.where(labels != k)[0]

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            n_aug_k = self._num_aug_for_class(k)

            # Select low-confidence positives: bottom p_k among label == k
            pk_pos = probs[pos_idx, k]
            num_low_pos = max(1, math.ceil(pos_idx.numel() * self.low_positive_ratio))
            num_low_pos = min(num_low_pos, pos_idx.numel())

            low_order = torch.argsort(pk_pos, descending=False)
            low_pos_idx = pos_idx[low_order[:num_low_pos]]

            # Select high-confidence negatives: top p_k among label != k
            # Following paper Appendix idea:
            # if target n_aug_k and we have m low positives,
            # choose about ceil(n_aug_k / m) hard negatives,
            # then mix all pairs.
            num_high_neg = max(1, math.ceil(n_aug_k / max(num_low_pos, 1)))
            num_high_neg = min(num_high_neg, neg_idx.numel())

            pk_neg = probs[neg_idx, k]
            high_order = torch.argsort(pk_neg, descending=True)
            high_neg_idx = neg_idx[high_order[:num_high_neg]]

            # Pair all low positives with selected high negatives
            pos_grid = low_pos_idx.repeat_interleave(num_high_neg)
            neg_grid = high_neg_idx.repeat(num_low_pos)

            if pos_grid.numel() == 0:
                continue

            # If too many pairs, randomly sample n_aug_k
            if pos_grid.numel() > n_aug_k:
                perm = torch.randperm(pos_grid.numel(), device=device)[:n_aug_k]
                pos_grid = pos_grid[perm]
                neg_grid = neg_grid[perm]

            num_pairs = pos_grid.numel()

            beta_dist = torch.distributions.Beta(self.rho, self.rho)
            lam = beta_dist.sample((num_pairs,)).to(device)

            # Reshape lambda to broadcast over image dimensions
            lam_shape = [num_pairs] + [1] * (images.dim() - 1)
            lam = lam.view(*lam_shape)

            x_pos = images[pos_grid]
            x_neg = images[neg_grid]

            x_hat = lam * x_pos + (1.0 - lam) * x_neg

            aug_list.append(x_hat)
            aug_neg_class_list.append(
                torch.full(
                    (num_pairs,),
                    fill_value=k,
                    device=device,
                    dtype=torch.long,
                )
            )

        if len(aug_list) == 0:
            return None, None

        aug_images = torch.cat(aug_list, dim=0)
        aug_negative_for_class = torch.cat(aug_neg_class_list, dim=0)

        return aug_images, aug_negative_for_class

    def _num_aug_for_class(self, k: int) -> int:
        """
        Paper Appendix C says the augmented negative number n_hat_k is
        inversely proportional to the number of observed samples of class k.

        If global class_counts is provided:
            n_aug_k = base_num_aug * max_count / count_k

        Then cap by max_aug_per_class for stability.
        """

        if (
            self.class_counts is not None
            and self.use_dataset_inverse_aug
        ):
            counts = self.class_counts
            max_count = torch.max(counts).item()
            count_k = max(counts[k].item(), 1.0)

            n_aug_k = int(round(self.base_num_aug * max_count / count_k))
            n_aug_k = max(1, n_aug_k)
            n_aug_k = min(n_aug_k, self.max_aug_per_class)
            return n_aug_k

        return self.base_num_aug

    def negative_dominant_contrastive_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aug_logits: Optional[torch.Tensor] = None,
        aug_negative_for_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single-domain paper-style Lcon.

        For anchor i with class k:
            positives:
                original batch samples with label == k, except itself

            negatives:
                original batch samples with label != k
                + augmented hard negatives generated for class k

        Lcon_i =
            -log(
                mean_{n in N(i)} [1 - s(p_i, p_n)]
                /
                sum_{a in A(i)} [1 - s(p_i, p_a)]
            )

        s is cosine similarity in prediction space.
        """

        device = logits.device
        batch_size = logits.size(0)

        if batch_size <= 1:
            return logits.new_tensor(0.0)

        # Prediction vectors p = f_theta(x)
        # Use softmax probabilities, then normalize for cosine similarity.
        p = F.softmax(logits, dim=1)
        p = F.normalize(p, dim=1)

        sim_orig = torch.matmul(p, p.T)
        sim_orig = sim_orig.clamp(min=0.0, max=1.0)
        dissim_orig = 1.0 - sim_orig

        if aug_logits is not None and aug_negative_for_class is not None:
            p_aug = F.softmax(aug_logits, dim=1)
            p_aug = F.normalize(p_aug, dim=1)

            sim_aug = torch.matmul(p, p_aug.T)
            sim_aug = sim_aug.clamp(min=0.0, max=1.0)
            dissim_aug = 1.0 - sim_aug
        else:
            dissim_aug = None

        eye = torch.eye(batch_size, dtype=torch.bool, device=device)

        loss_items = []

        for i in range(batch_size):
            k = labels[i]

            original_negative_mask = labels != k
            original_all_mask = ~eye[i]

            # Numerator: original negatives
            neg_dissim_sum = dissim_orig[i, original_negative_mask].sum()
            neg_count = original_negative_mask.sum()

            # Denominator: all original samples except itself
            denom_sum = dissim_orig[i, original_all_mask].sum()

            # Add class-specific augmented hard negatives
            if dissim_aug is not None:
                aug_mask = aug_negative_for_class == k

                if aug_mask.any():
                    aug_dissim = dissim_aug[i, aug_mask]

                    neg_dissim_sum = neg_dissim_sum + aug_dissim.sum()
                    neg_count = neg_count + aug_mask.sum()

                    # Augmented negatives are also included in denominator
                    denom_sum = denom_sum + aug_dissim.sum()

            if neg_count.item() == 0:
                continue

            numerator = neg_dissim_sum / neg_count.float().clamp(min=1.0)
            ratio = numerator / (denom_sum + self.eps)

            loss_i = -torch.log(ratio + self.eps)
            loss_items.append(loss_i)

        if len(loss_items) == 0:
            return logits.new_tensor(0.0)

        return torch.stack(loss_items).mean()