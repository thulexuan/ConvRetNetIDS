"""
One-view KCL/TSC model with loss separated.

This file is adapted from the original `builder.py` style, but:
    1. Uses one input view: forward(im, im_labels)
    2. Removes encoder_k / im_k / momentum key encoder
    3. Does NOT compute loss inside the model
    4. Returns logits and positive masks; the loss is computed in `losses_oneview.py`

Original KCL.forward(im_q, im_k, im_labels) computed:
    - q from im_q
    - k from im_k
    - logits
    - positive masks
    - loss, loss_class, loss_target

This version computes:
    - q from im
    - logits against queue and optional targets
    - mask_class and mask_target
and returns them in a dictionary.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def _hungarian(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve linear assignment.

    Uses scipy if available. Otherwise falls back to a greedy approximation.
    """
    cost_cpu = cost.detach().float().cpu()

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_cpu.numpy())
        return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)

    # Greedy fallback.
    n_rows, n_cols = cost_cpu.shape
    used_rows, used_cols = set(), set()
    rows, cols = [], []

    flat = torch.argsort(cost_cpu.reshape(-1), descending=False)
    for idx in flat.tolist():
        r = idx // n_cols
        c = idx % n_cols
        if r not in used_rows and c not in used_cols:
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)
            if len(rows) == min(n_rows, n_cols):
                break

    return torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)


def generate_uniform_targets(
    n_cls: int,
    dim: int,
    tau: float = 0.07,
    steps: int = 2000,
    lr: float = 0.05,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Generate class targets uniformly distributed on a unit hypersphere.

    This replaces dependency on:
        optimal_{n_cls}_{dim}_analytical.npy

    Objective:
        L_u = mean_i log sum_j exp(t_i^T t_j / tau)

    Returns:
        targets: [n_cls, dim], L2-normalized.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    targets = torch.randn(n_cls, dim, generator=gen, device=device)
    targets = _l2_normalize(targets, dim=1)
    targets = nn.Parameter(targets)

    optimizer = torch.optim.Adam([targets], lr=lr)

    for _ in range(steps):
        with torch.no_grad():
            targets.data = _l2_normalize(targets.data, dim=1)

        logits = targets @ targets.t() / tau
        loss = torch.logsumexp(logits, dim=1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        targets.data = _l2_normalize(targets.data, dim=1)

    return targets.detach().cpu()


class KCLOneViewModel(nn.Module):
    """
    One-view KCL/TSC model. Loss is NOT computed here.

    Use with:
        from losses_oneview import KCLTSCCriterion

        output = model(images, labels)
        loss_output = criterion(output["logits"], output["mask_class"], output["mask_target"])
        loss = loss_output.loss

    Args:
        base_encoder:
            Backbone constructor. It should support base_encoder(num_classes=dim), following
            the original builder.py convention.
        dim:
            Contrastive projection dimension.
        K:
            Queue size.
        T:
            Temperature used when computing logits.
        mlp:
            If True and encoder has .fc, replace fc with MLP head.
        num_positive:
            If > 0, sample at most this many same-class positives from the queue per anchor.
            If 0, use all same-class positives.
        targeted:
            If True, append class targets to denominator and return target positive mask.
        tr:
            Target repeat. Usually 1.
        n_cls:
            Number of classes. Original file hard-coded 1000; here it is configurable.
        target_path:
            Optional .npy target file [n_cls, dim]. If missing, targets are generated.
        center_momentum:
            Moving-average momentum for class centers.
    """

    def __init__(
        self,
        base_encoder,
        dim: int = 128,
        K: int = 65536,
        T: float = 0.07,
        mlp: bool = False,
        num_positive: int = 0,
        targeted: bool = False,
        tr: int = 1,
        n_cls: int = 1000,
        target_path: Optional[str] = None,
        target_generation_steps: int = 2000,
        target_generation_lr: float = 0.05,
        target_generation_seed: int = 0,
        center_momentum: float = 0.9,
        enqueue: bool = True,
    ) -> None:
        super().__init__()

        if n_cls <= 1:
            raise ValueError("n_cls must be > 1.")
        if dim <= 1:
            raise ValueError("dim must be > 1.")
        if K <= 0:
            raise ValueError("K must be > 0.")
        if tr <= 0:
            raise ValueError("tr must be > 0.")

        self.K = int(K)
        self.T = float(T)
        self.num_positive = int(num_positive)
        self.n_cls = int(n_cls)
        self.targeted = bool(targeted)
        self.tr = int(tr)
        self.center_momentum = float(center_momentum)
        self.enqueue = bool(enqueue)

        # Encoder.
        self.encoder_q = base_encoder(num_classes=dim)

        if mlp:
            if not hasattr(self.encoder_q, "fc"):
                raise AttributeError("mlp=True requires encoder_q to have `.fc`.")
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(inplace=True),
                self.encoder_q.fc,
            )

        # Queue and queue labels.
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = _l2_normalize(self.queue, dim=0)
        self.register_buffer("queue_labels", -torch.ones(1, self.K).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Class centroids for online target assignment.
        self.register_buffer("class_centroid", torch.randn(self.n_cls, dim))
        self.class_centroid = _l2_normalize(self.class_centroid, dim=1)

        # Targets.
        target = self._load_or_generate_targets(
            n_cls=self.n_cls,
            dim=dim,
            target_path=target_path,
            steps=target_generation_steps,
            lr=target_generation_lr,
            seed=target_generation_seed,
            tau=self.T,
        )  # [C, D]

        target = torch.as_tensor(target).float()
        target = _l2_normalize(target, dim=1)

        # Repeat targets if tr > 1.
        target_repeated = torch.cat(
            [target[i:i+1, :].repeat(self.tr, 1) for i in range(self.n_cls)],
            dim=0,
        )  # [C*tr, D]

        target_labels = torch.cat(
            [torch.tensor([i]).repeat(self.tr) for i in range(self.n_cls)],
            dim=0,
        ).long().unsqueeze(-1)  # [C*tr, 1]

        self.register_buffer("optimal_target", target_repeated)
        self.register_buffer("optimal_target_unique", target.transpose(0, 1).contiguous())  # [D, C]
        self.register_buffer("target_labels", target_labels)
        self.register_buffer("target_assignment", torch.arange(self.n_cls, dtype=torch.long))

    def _load_or_generate_targets(
        self,
        n_cls: int,
        dim: int,
        target_path: Optional[str],
        steps: int,
        lr: float,
        seed: int,
        tau: float,
    ) -> torch.Tensor:
        if target_path is None:
            target_path = f"optimal_{n_cls}_{dim}_analytical.npy"

        if os.path.exists(target_path):
            arr = np.load(target_path)
            if arr.shape != (n_cls, dim):
                raise ValueError(f"{target_path} has shape {arr.shape}, expected {(n_cls, dim)}.")
            return torch.tensor(arr).float()

        print(
            f"[KCLOneViewModel] Target file not found: {target_path}. "
            f"Generating uniform targets for n_cls={n_cls}, dim={dim}."
        )
        return generate_uniform_targets(
            n_cls=n_cls,
            dim=dim,
            tau=tau,
            steps=steps,
            lr=lr,
            seed=seed,
            device="cpu",
        )

    def _encode_query(self, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Robust parser for encoder output.

        Original implementation expects:
            q, feat_q = encoder_q(im, True)
        """
        try:
            out = self.encoder_q(im, True)
        except TypeError:
            out = self.encoder_q(im)

        if torch.is_tensor(out):
            return out, out

        if isinstance(out, (tuple, list)):
            if len(out) == 1:
                return out[0], out[0]
            return out[0], out[1]

        if isinstance(out, dict):
            q = None
            feat = None
            for key in ["projection", "proj", "logits", "q", "z", "out"]:
                if key in out:
                    q = out[key]
                    break
            for key in ["feat", "feature", "features", "embedding", "emb"]:
                if key in out:
                    feat = out[key]
                    break
            if q is None and feat is not None:
                q = feat
            if feat is None and q is not None:
                feat = q
            if q is not None and feat is not None:
                return q, feat

        raise TypeError("Cannot parse encoder output. Please edit `_encode_query()` for your backbone.")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Queue update. Supports single GPU/CPU and DDP.
        """
        keys = concat_all_gather(keys.detach())
        labels = concat_all_gather(labels.detach().contiguous().view(-1, 1).long())

        batch_size = keys.shape[0]

        if batch_size >= self.K:
            keys = keys[-self.K:]
            labels = labels[-self.K:]
            batch_size = self.K

        ptr = int(self.queue_ptr.item())

        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_labels[:, ptr:ptr + batch_size] = labels.T
        else:
            first_len = self.K - ptr
            second_len = batch_size - first_len

            self.queue[:, ptr:] = keys[:first_len].T
            self.queue_labels[:, ptr:] = labels[:first_len].T

            self.queue[:, :second_len] = keys[first_len:].T
            self.queue_labels[:, :second_len] = labels[first_len:].T

        self.queue_ptr[0] = (ptr + batch_size) % self.K

    @torch.no_grad()
    def _update_class_centroids_and_assignment(
        self,
        q: torch.Tensor,
        im_labels: torch.Tensor,
    ) -> None:
        """
        Update class centroids and solve online target assignment.
        """
        im_labels_all = concat_all_gather(im_labels.detach().contiguous().view(-1, 1).long())
        features_all = concat_all_gather(q.detach())
        features_all = _l2_normalize(features_all, dim=1)

        for one_label in torch.unique(im_labels_all).tolist():
            one_label = int(one_label)
            if one_label < 0 or one_label >= self.n_cls:
                continue

            mask = im_labels_all[:, 0].eq(one_label)
            if mask.sum() == 0:
                continue

            batch_center = _l2_normalize(features_all[mask].mean(dim=0), dim=0)
            self.class_centroid[one_label] = (
                self.center_momentum * self.class_centroid[one_label]
                + (1.0 - self.center_momentum) * batch_center
            )
            self.class_centroid[one_label] = _l2_normalize(self.class_centroid[one_label], dim=0)

        # Maximize centroid-target similarity.
        sim = torch.einsum("nc,ck->nk", [self.class_centroid, self.optimal_target_unique])
        rows, cols = _hungarian(-sim)
        assignment = torch.empty(self.n_cls, dtype=torch.long, device=self.target_assignment.device)
        assignment[rows.to(assignment.device)] = cols.to(assignment.device)
        self.target_assignment.copy_(assignment)

    def _sample_queue_positives(self, mask_queue: torch.Tensor) -> torch.Tensor:
        """
        Sample at most self.num_positive positives from queue for each anchor.
        """
        mask_queue = mask_queue.float()
        if self.num_positive <= 0:
            return mask_queue

        sampled = torch.zeros_like(mask_queue)

        for i in range(mask_queue.size(0)):
            pos_idx = torch.where(mask_queue[i] > 0)[0]
            if pos_idx.numel() == 0:
                continue
            n_select = min(self.num_positive, pos_idx.numel())
            perm = torch.randperm(pos_idx.numel(), device=mask_queue.device)[:n_select]
            sampled[i, pos_idx[perm]] = 1.0

        return sampled

    def forward(self, im: torch.Tensor, im_labels: torch.Tensor, eval: bool = False) -> Dict[str, Any]:
        """
        Forward pass without loss computation.

        Returns a dict:
            logits:      [B, K] or [B, K + C*tr]
            mask_class:  same-class positive mask, same shape as logits
            mask_target: assigned-target positive mask, same shape as logits
            mask_total:  mask_class + mask_target
            q:           normalized projection
            feat_q:      feature from encoder
            labels:      dummy labels for compatibility
        """
        q, feat_q = self._encode_query(im)
        q = _l2_normalize(q, dim=1)

        if eval:
            return {"q": q, "feat_q": feat_q}

        device = q.device
        B = q.size(0)

        im_labels = im_labels.contiguous().view(-1, 1).long().to(device)
        if im_labels.min().item() < 0 or im_labels.max().item() >= self.n_cls:
            raise ValueError(f"Labels must be in [0, {self.n_cls - 1}].")

        queue_features = self.queue.clone().detach().to(device)        # [D, K]
        queue_labels = self.queue_labels.clone().detach().to(device)   # [1, K]

        if self.targeted:
            target_features = self.optimal_target.to(device).transpose(0, 1)  # [D, C*tr]
            bank_features = torch.cat([queue_features, target_features], dim=1)
        else:
            bank_features = queue_features

        logits = torch.einsum("nc,ck->nk", [q, bank_features]) / self.T

        # Class positives from queue.
        mask_queue = torch.eq(im_labels, queue_labels).float()  # [B, K]
        mask_queue = self._sample_queue_positives(mask_queue)

        if self.targeted:
            with torch.no_grad():
                self._update_class_centroids_and_assignment(q, im_labels)

            n_targets_total = self.optimal_target.size(0)
            mask_target = torch.zeros(B, n_targets_total, device=device)

            labels_in_batch = torch.unique(im_labels[:, 0])
            for one_label in labels_in_batch.tolist():
                one_label = int(one_label)
                assigned_target = int(self.target_assignment[one_label].item())

                rows = torch.where(im_labels[:, 0] == one_label)[0]
                cols = torch.arange(
                    assigned_target * self.tr,
                    (assigned_target + 1) * self.tr,
                    device=device,
                    dtype=torch.long,
                )

                if rows.numel() > 0 and cols.numel() > 0:
                    mask_target[rows[:, None], cols[None, :]] = 1.0

            mask_class = torch.cat([mask_queue, torch.zeros_like(mask_target)], dim=1)
            mask_target_full = torch.cat([torch.zeros_like(mask_queue), mask_target], dim=1)
        else:
            mask_class = mask_queue
            mask_target_full = torch.zeros_like(mask_class)

        mask_total = mask_class + mask_target_full
        labels = torch.zeros(B, dtype=torch.long, device=device)

        # Update queue after creating logits/masks.
        if self.enqueue:
            self._dequeue_and_enqueue(q.detach(), im_labels.detach())

        return {
            "logits": logits,
            "mask_class": mask_class,
            "mask_target": mask_target_full,
            "mask_total": mask_total,
            "labels": labels,
            "q": q,
            "feat_q": feat_q,
            "im_labels": im_labels.squeeze(1),
            "target_assignment": self.target_assignment.detach().clone(),
        }


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Safe all_gather for both DDP and single-process training.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    return tensor
