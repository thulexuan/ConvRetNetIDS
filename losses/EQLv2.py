from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class EQLv2Loss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        gamma: float = 12.0,
        mu: float = 0.8,
        alpha: float = 4.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.reduction = reduction

        self.map_func = partial(self._sigmoid_map, gamma=gamma, mu=mu)

        # --- persistent state (NOT learnable parameters) ---
        # register_buffer ensures these tensors:
        #   - move with the module (.to(device), .cuda(), etc.)
        #   - are included in state_dict() for checkpoint save/load
        #   - are NOT updated by the optimizer (no gradient tracked)
        self.register_buffer("_pos_grad", torch.zeros(num_classes))
        self.register_buffer("_neg_grad", torch.zeros(num_classes))
        self.register_buffer("pos_neg", torch.ones(num_classes))
        # tracks whether at least one batch of statistics has been
        # collected yet (mirrors the `self._pos_grad is None` check in
        # the original Keras code, but without relying on Python-level
        # None checks that don't survive graph tracing / TorchScript)
        self.register_buffer("_initialized", torch.tensor(False))

    @staticmethod
    def _sigmoid_map(x: torch.Tensor, gamma: float, mu: float) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-gamma * (x - mu)))

    def get_weight(self):
        """Compute per-class positive/negative loss weights.

        Before any statistics have been collected (first ever forward
        call), both weights default to 1.0, i.e. behaves like plain
        BCE for the very first batch.
        """
        if not bool(self._initialized):
            pos_w = torch.ones(self.num_classes, device=self.pos_neg.device)
            neg_w = torch.ones(self.num_classes, device=self.pos_neg.device)
        else:
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1.0 + self.alpha * (1.0 - neg_w)
        return pos_w, neg_w

    @torch.no_grad()
    def collect_grad(
        self,
        cls_score: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        """Update the accumulated positive/negative gradient statistics.

        Must run without gradient tracking: it only maintains running
        statistics used to reweight *future* batches, and must not
        itself be part of the autograd graph.
        """
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1.0) + (1.0 - target) * prob
        grad = torch.abs(grad)

        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1.0 - target) * weight, dim=0)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)
        self._initialized.fill_(True)

    def forward(self, cls_score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_score: (N, C) raw logits, BEFORE sigmoid.
            target:    (N, C) one-hot (or multi-hot) ground-truth, float.

        Returns:
            scalar loss tensor.
        """
        if cls_score.shape != target.shape:
            raise ValueError(
                f"cls_score shape {tuple(cls_score.shape)} must match "
                f"target shape {tuple(target.shape)} (one-hot targets expected)."
            )

        target = target.float()

        pos_w, neg_w = self.get_weight()
        weight = pos_w.unsqueeze(0) * target + neg_w.unsqueeze(0) * (1.0 - target)

        # elementwise BCE-with-logits, no reduction yet
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_score, target, reduction="none"
        )
        cls_loss = torch.sum(cls_loss * weight) / self.num_classes

        # update running gradient statistics for the NEXT batch
        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        if self.reduction == "sum":
            return cls_loss * cls_score.shape[0]
        # 'mean' and 'none' both just return the per-class-normalized
        # batch loss here, consistent with the original implementation
        return cls_loss




# from functools import partial

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class EQLv2Loss(nn.Module):

#     def __init__(
#         self,
#         num_classes: int,
#         gamma: float = 12.0,
#         mu: float = 0.8,
#         alpha: float = 4.0,
#         reduction: str = "mean",
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.gamma = gamma
#         self.mu = mu
#         self.alpha = alpha
#         self.reduction = reduction

#         self._map_func = partial(self._sigmoid_map, gamma=gamma, mu=mu)

#         # --- persistent gradient-ratio bookkeeping state (Eq. 6-7, 12) ---
#         # These are NOT learnable parameters (no gradient flows into
#         # them via backprop); they are running statistics, analogous
#         # to the running_mean/running_var buffers of BatchNorm.
#         self.register_buffer("_pos_grad", torch.zeros(num_classes))
#         self.register_buffer("_neg_grad", torch.zeros(num_classes))
#         self.register_buffer("pos_neg", torch.ones(num_classes))
#         # tracks whether Eq. (6)-(7) have been evaluated at least once;
#         # before that, weights default to 1 (plain BCE), matching the
#         # `if self._pos_grad is None` branch of the reference code.
#         self.register_buffer("_initialized", torch.tensor(False))

#     @staticmethod
#     def _sigmoid_map(x: torch.Tensor, gamma: float, mu: float) -> torch.Tensor:
#         """Eq. (8): f(x) = 1 / (1 + exp(-gamma * (x - mu)))."""
#         return 1.0 / (1.0 + torch.exp(-gamma * (x - mu)))

#     def get_weight(self):
#         """Eq. (9): compute per-class positive/negative loss weights
#         q_j^(t) (pos_w) and r_j^(t) (neg_w) from the current
#         accumulated ratio `pos_neg` (g_j^(t))."""
#         if not bool(self._initialized):
#             # first-ever batch: no statistics yet -> behave like plain BCE
#             pos_w = torch.ones(self.num_classes, device=self.pos_neg.device)
#             neg_w = torch.ones(self.num_classes, device=self.pos_neg.device)
#         else:
#             neg_w = self._map_func(self.pos_neg)               # r_j^(t)
#             pos_w = 1.0 + self.alpha * (1.0 - neg_w)            # q_j^(t)
#         return pos_w, neg_w

#     @torch.no_grad()
#     def collect_grad(
#         self,
#         cls_score: torch.Tensor,
#         target: torch.Tensor,
#         weight: torch.Tensor,
#     ) -> None:
#         """Eq. (6): accumulate positive/negative gradient magnitude per
#         class over this batch, then update the accumulated ratio
#         `pos_neg` (Eq. 7, updated per Eq. 12 for the next iteration).

#         Must run without gradient tracking: this only maintains
#         bookkeeping statistics used to reweight *future* batches and
#         must not itself be part of the autograd graph.
#         """
#         prob = torch.sigmoid(cls_score)
#         # |y*(p-1) + (1-y)*p| : magnitude of the sigmoid-BCE gradient
#         grad = torch.abs(target * (prob - 1.0) + (1.0 - target) * prob)

#         pos_grad = torch.sum(grad * target * weight, dim=0)
#         neg_grad = torch.sum(grad * (1.0 - target) * weight, dim=0)

#         self._pos_grad += pos_grad
#         self._neg_grad += neg_grad
#         self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)  # Eq. (12)
#         self._initialized.fill_(True)

#     def forward(self, cls_score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             cls_score: (N, C) raw logits, BEFORE sigmoid (analogous to
#                 `y_pred` / `cls_score` in the paper's implementation).
#             target: (N, C) one-hot (or multi-hot) ground-truth, float
#                 (analogous to `y_true` / `label`, produced e.g. via
#                 one-hot encoding of the class index as in the paper's
#                 `to_categorical()` usage).

#         Returns:
#             scalar loss tensor.
#         """
#         if cls_score.shape != target.shape:
#             raise ValueError(
#                 f"cls_score shape {tuple(cls_score.shape)} must match "
#                 f"target shape {tuple(target.shape)} (one-hot targets expected)."
#             )
#         target = target.float()
#         n_i = cls_score.shape[0]

#         # Eq. (9): per-class positive/negative weights for this iteration
#         pos_w, neg_w = self.get_weight()
#         weight = pos_w.unsqueeze(0) * target + neg_w.unsqueeze(0) * (1.0 - target)

#         # per-class binary cross-entropy (sigmoid), elementwise
#         cls_loss = F.binary_cross_entropy_with_logits(
#             cls_score, target, reduction="none"
#         )
#         # reweight by q_j/r_j (Eq. 10-11), then normalize as in the paper
#         cls_loss = torch.sum(cls_loss * weight) / self.num_classes

#         # Eq. (6)-(7),(12): update running gradient-ratio statistics
#         # for the NEXT batch, using this batch's logits/target/weight
#         self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

#         if self.reduction == "sum":
#             return cls_loss
#         # 'mean': additionally average over the batch size for a
#         # training-friendly, batch-size-invariant scalar
#         return cls_loss / n_i


# from functools import partial
 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
 
 
# class EQLv2(nn.Module):
#     def __init__(
#         self,
#         use_sigmoid=True,
#         reduction="mean",
#         class_weight=None,
#         loss_weight=1.0,
#         num_classes=10,
#         gamma=12,
#         mu=0.8,
#         alpha=4.0,
#         vis_grad=False,
#         per_class_grad=True,
#     ):
#         super().__init__()
#         self.use_sigmoid = True
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.class_weight = class_weight
#         self.num_classes = num_classes
#         self.group = True
 
#         # cfg for eqlv2
#         self.vis_grad = vis_grad
#         self.gamma = gamma
#         self.mu = mu
#         self.alpha = alpha
#         self.per_class_grad = per_class_grad
 
#         # initial variables -- buffers instead of `None` / plain tensors,
#         # so that they persist, move with the module and are checkpointed
#         self.register_buffer("_pos_grad", torch.zeros(num_classes))
#         self.register_buffer("_neg_grad", torch.zeros(num_classes))
#         self.register_buffer("pos_neg", torch.ones(num_classes))
#         # replaces the `if self._pos_grad is None` check of the original
#         self.register_buffer("_initialized", torch.tensor(False))
 
#         def _func(x, gamma, mu):
#             return 1 / (1 + torch.exp(-gamma * (x - mu)))
 
#         self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
 
#     def forward(self, cls_score, label):
#         """Equivalent of Keras `call(y_true, y_pred)`.
 
#         Args:
#             cls_score: (N, C) raw logits, BEFORE sigmoid.
#             label: (N, C) one-hot / multi-hot target, float.
#         """
#         self.n_i, self.n_c = cls_score.shape[0], cls_score.shape[1]
#         self.gt_classes = label
#         self.pred_class_logits = cls_score
 
#         target = label.float()
#         pos_w, neg_w = self.get_weight(cls_score)
 
#         weight = pos_w * target + neg_w * (1 - target)          # (N, C)
 
#         cls_loss = F.binary_cross_entropy_with_logits(
#             cls_score, target, reduction="none"
#         )
#         cls_loss = torch.sum(cls_loss * weight) / self.n_c
 
#         self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
 
#         return self.loss_weight * cls_loss
 
#     def get_channel_num(self, num_classes):
#         num_channel = num_classes + 1
#         return num_channel
 
#     def get_activation(self, cls_score):
#         """Rescale foreground scores by (1 - background score).
 
#         The original used `.size()` / `.view()` on a TF tensor and
#         mutated `cls_score` in place; rewritten here as valid,
#         non-mutating PyTorch. Only meaningful if the last channel is a
#         background/objectness class.
#         """
#         cls_score = torch.sigmoid(cls_score)
#         n_i, n_c = cls_score.shape
#         bg_score = cls_score[:, -1].view(n_i, 1)
#         cls_score = torch.cat(
#             [cls_score[:, :-1] * (1 - bg_score), bg_score], dim=1
#         )
#         return cls_score
 
#     @torch.no_grad()
#     def collect_grad(self, cls_score, target, weight):
#         prob = torch.sigmoid(cls_score)
#         grad = target * (prob - 1) + (1 - target) * prob
#         grad = torch.abs(grad)
 
#         if self.per_class_grad:
#             # sum over the batch dimension only -> vector of length C,
#             # matching Eq. (6) of the paper (per-class statistics)
#             pos_grad = torch.sum(grad * target * weight, dim=0)
#             neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)
#         else:
#             # original behaviour: scalar, broadcast to every class
#             pos_grad = torch.sum(grad * target * weight)
#             neg_grad = torch.sum(grad * (1 - target) * weight)
 
#         self._pos_grad += pos_grad
#         self._neg_grad += neg_grad
#         self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)
#         self._initialized.fill_(True)
 
#     def get_weight(self, cls_score):
#         # we do not have information about pos grad and neg grad at beginning
#         if not bool(self._initialized):
#             device = cls_score.device
#             neg_w = torch.ones(self.n_c, device=device)
#             pos_w = torch.ones(self.n_c, device=device)
#         else:
#             # the negative weight for objectiveness is always 1
#             neg_w = self.map_func(self.pos_neg)
#             pos_w = 1 + self.alpha * (1 - neg_w)
#             neg_w = neg_w.view(1, -1)
#             pos_w = pos_w.view(1, -1)
#         return pos_w, neg_w
 
#     def reset_stats(self):
#         """Clear the accumulators, e.g. between StratifiedKFold folds.
 
#         The original Keras script built the model once *outside* the
#         k-fold loop, so statistics from earlier folds leaked into later
#         ones. Call this at the start of every fold.
#         """
#         self._pos_grad.zero_()
#         self._neg_grad.zero_()
#         self.pos_neg.fill_(1.0)
#         self._initialized.fill_(False)
 
#     def extra_repr(self):
#         return (
#             f"num_classes={self.num_classes}, gamma={self.gamma}, "
#             f"mu={self.mu}, alpha={self.alpha}, "
#             f"loss_weight={self.loss_weight}, "
#             f"per_class_grad={self.per_class_grad}"
#         )