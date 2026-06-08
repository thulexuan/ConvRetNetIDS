"""
One-view Targeted Supervised Contrastive Learning (TSC-1View).

This file implements a practical one-view variant of:
"Targeted Supervised Contrastive Learning for Long-Tailed Recognition" (CVPR 2022).

Original TSC uses:
    anchor feature v_i
    augmented feature v_i_tilde
    k same-class positives
    class target c_i*

This implementation removes the augmented view and keeps:
    anchor feature z_i
    k same-class positives from the current batch
    assigned class target c_i*

So the loss becomes:
    L = L_same_class_contrastive + lambda_target * L_target_contrastive

Inputs to the loss:
    features: [B, D] normalized or unnormalized embeddings from backbone
    labels:   [B] class labels

The module internally:
    1. Generates C uniform class targets on a unit hypersphere.
    2. Maintains class centers using moving average.
    3. Assigns each class center to one target using Hungarian matching.
    4. Computes one-view TSC loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def _linear_sum_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for scipy.optimize.linear_sum_assignment.

    If scipy is unavailable, uses a greedy fallback. The greedy fallback is not exactly optimal,
    but keeps the code runnable. For serious experiments, install scipy.
    """
    cost_cpu = cost.detach().float().cpu()

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        row_ind, col_ind = linear_sum_assignment(cost_cpu.numpy())
        return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)
    except Exception:
        # Greedy fallback: repeatedly choose the smallest remaining pair.
        C = cost_cpu.size(0)
        used_rows = set()
        used_cols = set()
        rows = []
        cols = []

        flat = torch.argsort(cost_cpu.reshape(-1), descending=False)
        for idx in flat.tolist():
            r = idx // C
            c = idx % C
            if r not in used_rows and c not in used_cols:
                rows.append(r)
                cols.append(c)
                used_rows.add(r)
                used_cols.add(c)
                if len(rows) == C:
                    break

        return torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)


def generate_uniform_targets(
    num_classes: int = 10,
    emb_dim: int = 128,
    tau: float = 0.07,
    steps: int = 10000,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device | str] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Generate C target vectors on a D-dimensional unit hypersphere.

    It minimizes:
        L_u = mean_i log sum_j exp(t_i^T t_j / tau)

    Args:
        num_classes: C.
        emb_dim: D.
        tau: temperature for target uniformity optimization.
        steps: number of optimization steps.
        lr: target generation learning rate.
        seed: random seed.
        device: CPU/GPU.
        verbose: print progress.

    Returns:
        targets: [C, D], L2-normalized.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    targets = torch.randn(num_classes, emb_dim, generator=g, device=device)
    targets = _safe_normalize(targets, dim=1)
    targets = nn.Parameter(targets)

    opt = torch.optim.Adam([targets], lr=lr)

    for step in range(steps):
        with torch.no_grad():
            targets.data = _safe_normalize(targets.data, dim=1)

        sim = targets @ targets.t() / tau
        loss = torch.logsumexp(sim, dim=1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose and (step % max(1, steps // 10) == 0 or step == steps - 1):
            print(f"[target generation] step={step:04d}/{steps}, Lu={loss.item():.6f}")

    with torch.no_grad():
        targets.data = _safe_normalize(targets.data, dim=1)

    return targets.detach()


@dataclass
class TSCOneViewOutput:
    loss: torch.Tensor
    same_class_loss: torch.Tensor
    target_loss: torch.Tensor
    num_positive_pairs: int
    assignment: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        out = {
            "loss": self.loss,
            "same_class_loss": self.same_class_loss,
            "target_loss": self.target_loss,
            "num_positive_pairs": torch.tensor(self.num_positive_pairs),
            "assignment": self.assignment,
        }
       
        return out


class TSCOneViewLoss(nn.Module):
    """
    One-view variant of Targeted Supervised Contrastive Learning.

    Important:
        - This is NOT the exact original TSC loss because original TSC uses an augmented view.
        - This version is designed for pipelines where each input has only one view.
        - It preserves the core TSC idea: uniform class targets + online class-target matching.
    """

    def __init__(
        self,
        num_classes: int = 10,
        emb_dim: int = 128,
        tau: float = 0.07,
        k: int = 6,
        lambda_target: float = 1.0,
        center_momentum: float = 0.9,
        targets: Optional[torch.Tensor] = None,
        target_generation_steps: int = 10000,
        target_generation_lr: float = 1e-3,
        target_generation_seed: int = 0,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError("num_classes must be > 1.")
        if emb_dim <= 1:
            raise ValueError("emb_dim must be > 1.")
        if k < 0:
            raise ValueError("k must be >= 0.")

        self.num_classes = int(num_classes)
        self.emb_dim = int(emb_dim)
        self.tau = float(tau)
        self.k = int(k)
        self.lambda_target = float(lambda_target)
        self.center_momentum = float(center_momentum)

        if targets is None:
            targets = generate_uniform_targets(
                num_classes=num_classes,
                emb_dim=emb_dim,
                tau=tau,
                steps=target_generation_steps,
                lr=target_generation_lr,
                seed=target_generation_seed,
                device="cpu",
                verbose=False,
            )
        else:
            if targets.shape != (num_classes, emb_dim):
                raise ValueError(f"targets must have shape [{num_classes}, {emb_dim}], got {targets.shape}.")
            targets = targets.detach().float().cpu()

        targets = _safe_normalize(targets, dim=1)

        self.register_buffer("targets", targets.clone())
        # Initialize centers with targets. This avoids zero centers before all classes appear.
        self.register_buffer("class_centers", targets.clone())
        self.register_buffer("center_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("assignment", torch.arange(num_classes, dtype=torch.long))

    @torch.no_grad()
    def update_class_centers(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update moving-average class centers from current batch features.
        """
        features = _safe_normalize(features.detach(), dim=1)
        labels = labels.detach().long()

        for cls in labels.unique().tolist():
            cls = int(cls)
            mask = labels == cls
            if mask.sum() == 0:
                continue

            center_new = _safe_normalize(features[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)

            if not bool(self.center_initialized[cls].item()):
                self.class_centers[cls] = center_new
                self.center_initialized[cls] = True
            else:
                updated = self.center_momentum * self.class_centers[cls] + (1.0 - self.center_momentum) * center_new
                self.class_centers[cls] = _safe_normalize(updated, dim=0)

    @torch.no_grad()
    def update_assignment(self) -> torch.Tensor:
        """
        Assign each class center to one target using Hungarian matching.

        assignment[class_id] = target_id
        """
        centers = _safe_normalize(self.class_centers, dim=1)
        targets = _safe_normalize(self.targets, dim=1)

        # Euclidean distance on hypersphere.
        cost = torch.cdist(centers, targets, p=2)

        rows, cols = _linear_sum_assignment(cost)
        new_assignment = torch.empty(self.num_classes, dtype=torch.long)
        new_assignment[rows] = cols

        self.assignment.copy_(new_assignment.to(self.assignment.device))
        return self.assignment

    def _sample_positive_indices(self, pos_idx: torch.Tensor, k: int) -> torch.Tensor:
        """
        Randomly sample up to k positives. If k == 0, use all positives.
        """
        if pos_idx.numel() == 0:
            return pos_idx

        if k <= 0 or pos_idx.numel() <= k:
            return pos_idx

        perm = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:k]
        return pos_idx[perm]

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        update_centers: bool = True,
        use_target_term: bool = True,
        use_same_class_term: bool = True,
    ) -> TSCOneViewOutput:
        """
        Compute one-view TSC loss.

        Args:
            features: [B, D] embeddings from backbone.
            labels: [B] integer class labels.
            update_centers: update class centers and class-target assignment.
            use_target_term: enable target contrastive term.
            use_same_class_term: enable same-class contrastive term.

        Returns:
            TSCOneViewOutput.
        """
        if features.dim() != 2:
            raise ValueError(f"features must be [B, D], got {features.shape}.")
        if features.size(1) != self.emb_dim:
            raise ValueError(f"Expected feature dim {self.emb_dim}, got {features.size(1)}.")

        labels = labels.long()
        if labels.dim() != 1 or labels.size(0) != features.size(0):
            raise ValueError("labels must be [B] and match features batch size.")
        if labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise ValueError(f"labels must be in [0, {self.num_classes - 1}].")

        z = _safe_normalize(features, dim=1)
        B = z.size(0)
        device = z.device

        # Move buffers to the same device automatically.
        targets = _safe_normalize(self.targets.to(device), dim=1)

        if update_centers:
            self.update_class_centers(z, labels)
            self.update_assignment()

        assignment = self.assignment.to(device)
        assigned_target_idx = assignment[labels]  # [B]

        # Logits with samples and targets.
        sample_logits = z @ z.t() / self.tau  # [B, B]
        eye = torch.eye(B, device=device, dtype=torch.bool)
        sample_logits = sample_logits.masked_fill(eye, float("-inf"))  # exclude anchor itself

        target_logits = z @ targets.t() / self.tau  # [B, C]

        # Denominator: other batch samples + all class targets.
        denom_logits = torch.cat([sample_logits, target_logits], dim=1)  # [B, B + C]
        log_denom = torch.logsumexp(denom_logits, dim=1)  # [B]

        # Same-class contrastive term.
        same_class_loss = z.new_tensor(0.0)
        num_positive_pairs = 0

        if use_same_class_term:
            log_probs = sample_logits - log_denom[:, None]  # [B, B]
            anchor_losses = []

            for i in range(B):
                pos_mask = labels == labels[i]
                pos_mask[i] = False
                pos_idx = torch.where(pos_mask)[0]
                pos_idx = self._sample_positive_indices(pos_idx, self.k)

                if pos_idx.numel() > 0:
                    # Average positives for this anchor, like KCL/TSC-style positive averaging.
                    anchor_losses.append(-log_probs[i, pos_idx].mean())
                    num_positive_pairs += int(pos_idx.numel())

            if len(anchor_losses) > 0:
                same_class_loss = torch.stack(anchor_losses).mean()

        # Target contrastive term.
        target_loss = z.new_tensor(0.0)
        if use_target_term and self.lambda_target != 0:
            target_pos_logits = target_logits[torch.arange(B, device=device), assigned_target_idx]
            target_loss = -(target_pos_logits - log_denom).mean()

        total_loss = same_class_loss + self.lambda_target * target_loss

        return TSCOneViewOutput(
            loss=total_loss,
            same_class_loss=same_class_loss.detach(),
            target_loss=target_loss.detach(),
            num_positive_pairs=num_positive_pairs,
            assignment=assignment.detach().cpu(),
        )
