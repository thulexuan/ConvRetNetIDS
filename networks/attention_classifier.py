"""
attention_classifier.py
==========================

Attention-Weighted MIL Classifier for window-level CAN intrusion
classification, with a per-class attention branch that doubles as a
weakly-supervised frame-level localization signal (no frame-level
labels needed during training).

Theoretical basis
-------------------
The gated attention mechanism (tanh-branch * sigmoid-branch) is from:
    Ilse, M., Tomczak, J. M., & Welling, M. (2018). "Attention-based
    Deep Multiple Instance Learning." ICML 2018 / PMLR 80.
    arXiv:1802.04712.  Official code: github.com/AMLab-Amsterdam/AttentionDeepMIL

The per-class (multi-branch) extension -- one independent attention
branch AND one independent linear head per class, so each class
gathers and judges its own evidence without mixing information from
other classes -- follows:
    Lu, M. Y. et al. (2021). "Data-efficient and weakly supervised
    computational pathology on whole-slide images" (CLAM). Nature
    Biomedical Engineering, 5, 555-570.

How this maps onto CAN-bus intrusion detection
-------------------------------------------------
    CLAM (whole-slide image)          CAN IDS (this module)
    ---------------------------       ---------------------------
    bag  = 1 WSI                      bag  = 1 window (e.g. 32 frames)
    instance = 1 image patch          instance = 1 CAN frame
    only slide-level label known      only window-level label known
    (which patches show tumor?)       (which frames are the attack?)

Only the WINDOW-LEVEL label is ever used for training (standard
cross-entropy on `logits`). The per-class attention weights this
module produces as a byproduct can, AFTER training, be read off and
compared against frame-level ground truth (if available, purely for
EVALUATION -- never for training) to check whether the model's
attention happens to concentrate on the actual attack frames. See
`detect_attack_frames` at the bottom of this file for that step.

Input / output
---------------
Input:
    h: (batch, window_size, dim)  -- token representations from the
       Retention encoder (this module does not care what produced them)

Output of `AttentionMILClassifier.forward`:
    {
      "logits":       (batch, num_classes)                -- for CrossEntropyLoss
      "pooled":       (batch, num_classes, dim)            -- per-class pooled vectors
      "attn_weights": (batch, num_classes, window_size)    -- for interpretability /
                                                               frame-level localization
    }
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Gated attention pooling (Ilse et al. 2018), one branch per class (CLAM-style)
# --------------------------------------------------------------------------- #
class AttentionWeightedPooling(nn.Module):
    """
    Multi-class, gated attention-weighted pooling over a window of
    token representations.

    For each class n, a dedicated attention branch scores every
    frame in the window:

        score_n[k] = w_n^T ( tanh(V h_k) . sigmoid(U h_k) )   (gated attention, Ilse et al. 2018)
        attn_n     = softmax_k( score_n )                      (normalized over the window)
        pooled_n   = sum_k attn_n[k] * h_k                     (class-specific pooled vector)

    V and U (the two gate branches) are SHARED across classes -- only
    the final linear projection to `num_classes` raw scores is
    class-specific. This keeps parameter count low while still giving
    each class its own normalized attention distribution over the
    window (because softmax is applied per-class, along the window
    dimension, independently for each of the `num_classes` rows).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        attn_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        hidden_dim = attn_hidden_dim or max(dim // 2, 1)

        # gated attention (Ilse et al. 2018): tanh branch * sigmoid branch
        self.attn_V = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attn_U = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
        # class-specific projection: (batch, L, hidden_dim) -> (batch, L, num_classes)
        self.attn_class_proj = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> dict:
        """
        Args:
            h:         (batch, window_size, dim)
            attn_mask: optional (batch, window_size), 1 = valid frame, 0 = padding
        Returns:
            {"pooled": (batch, num_classes, dim),
             "attn_weights": (batch, num_classes, window_size)}
        """
        b, l, d = h.shape
        assert d == self.dim, f"expected last dim {self.dim}, got {d}"

        gated = self.attn_V(h) * self.attn_U(h)  # (b, l, hidden_dim)
        raw_scores = self.attn_class_proj(gated).transpose(1, 2)  # (b, num_classes, l)

        if attn_mask is not None:
            key_mask = attn_mask[:, None, :].to(dtype=torch.bool)  # (b, 1, l)
            raw_scores = raw_scores.masked_fill(~key_mask, float("-inf"))

        attn_weights = torch.softmax(raw_scores, dim=-1)  # (b, num_classes, l), each row sums to 1

        # pooled[b, n, :] = sum_k attn_weights[b, n, k] * h[b, k, :]
        pooled = torch.einsum("bnl,bld->bnd", attn_weights, h)  # (b, num_classes, d)

        return {"pooled": pooled, "attn_weights": attn_weights}

    @staticmethod
    def get_pseudo_labels(attn_weights: torch.Tensor, k: int) -> torch.Tensor:
        """
        Optional CLAM-style pseudo instance labels for an auxiliary
        instance-clustering loss (CLAM Eq. 6-7): top-k attended frames
        -> label 1 ("critical evidence"), bottom-k -> label 0 ("noise"),
        the rest -> -100 (ignored by CrossEntropyLoss's default
        ignore_index).

        Args:
            attn_weights: (batch, num_classes, window_size)
            k: number of top/bottom frames to label (keep small for
               short windows, e.g. 2-4 for a 32-frame window; CLAM's
               original default of 8 assumes bags with hundreds of
               instances).
        Returns:
            (batch, num_classes, window_size) long tensor in {0,1,-100}.
        """
        b, n, l = attn_weights.shape
        k = min(k, l // 2) if l >= 2 else 0
        pseudo_labels = torch.full(
            (b, n, l), fill_value=-100, dtype=torch.long, device=attn_weights.device
        )
        if k <= 0:
            return pseudo_labels
        topk_idx = attn_weights.topk(k, dim=-1, largest=True).indices
        bottomk_idx = attn_weights.topk(k, dim=-1, largest=False).indices
        pseudo_labels.scatter_(-1, topk_idx, 1)
        pseudo_labels.scatter_(-1, bottomk_idx, 0)
        return pseudo_labels


# --------------------------------------------------------------------------- #
# Full classifier: pooling + N independent per-class linear heads -> logits
# --------------------------------------------------------------------------- #
class AttentionMILClassifier(nn.Module):
    """
    Retention-encoder output -> per-class attention pooling -> per-class
    linear head -> logits.

    Each class n has:
      (a) its own attention branch (inside AttentionWeightedPooling),
          deciding which frames in the window are evidence for class n;
      (b) its own linear head `bag_classifiers[n]`, which looks ONLY at
          `pooled[:, n, :]` (class n's own pooled vector) and outputs a
          single score for class n.

    logits[:, n] therefore depends only on evidence class n's own
    attention branch gathered -- it never sees what another class's
    branch attended to. The N per-class scores are concatenated into
    the final logits vector, shape-wise identical to a standard shared
    Linear(dim, num_classes) classifier, but built from independently
    gathered, class-specific evidence.

    Set `classifier_hidden_dim` to replace each Linear(dim,1) head with
    a small MLP (Linear -> ReLU -> Dropout -> Linear); see the
    docstring note in earlier discussion for the trade-offs (more
    expressive per-class decision boundary vs. higher overfitting risk
    for a small model / limited data).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        attn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        classifier_hidden_dim: int | None = None,
        classifier_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        self.pooling = AttentionWeightedPooling(
            dim=dim, num_classes=num_classes, attn_hidden_dim=attn_hidden_dim, dropout=dropout,
        )

        if classifier_hidden_dim is None:
            self.bag_classifiers = nn.ModuleList(
                [nn.Linear(dim, 1) for _ in range(num_classes)]
            )
        else:
            self.bag_classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim, classifier_hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(classifier_dropout),
                        nn.Linear(classifier_hidden_dim, 1),
                    )
                    for _ in range(num_classes)
                ]
            )

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> dict:
        """
        Args:
            h:         (batch, window_size, dim) -- Retention encoder output
            attn_mask: optional (batch, window_size) padding mask
        Returns:
            {"logits": (batch, num_classes),
             "pooled": (batch, num_classes, dim),
             "attn_weights": (batch, num_classes, window_size)}
        """
        pooling_out = self.pooling(h, attn_mask=attn_mask)
        pooled = pooling_out["pooled"]  # (batch, num_classes, dim)

        per_class_scores = [
            self.bag_classifiers[n](pooled[:, n, :])  # (batch, 1)
            for n in range(self.num_classes)
        ]
        logits = torch.cat(per_class_scores, dim=1)  # (batch, num_classes)

        return {"logits": logits, "pooled": pooled, "attn_weights": pooling_out["attn_weights"]}


# --------------------------------------------------------------------------- #
# Post-training utility: frame-level attack detection from attn_weights
# --------------------------------------------------------------------------- #
@torch.no_grad()
def detect_attack_frames(
    model_output: dict,
    normal_class_idx: int = 0,
    use_predicted_class: bool = True,
    true_labels: torch.Tensor | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """
    Given a batch's model output (from a model whose classifier is an
    AttentionMILClassifier), determine which frames within each
    predicted-attack window are the likely attack frames -- WITHOUT
    ever needing frame-level labels.

    Two-stage logic per window:
      1. Is this window an attack at all? (window-level classification,
         already computed in `logits`)
      2. If yes, which frames within it look suspicious? (read off the
         attention branch for that class)

    Args:
        model_output: dict with "logits" (batch, num_classes) and
            "attn_weights" (batch, num_classes, window_size), as
            returned by AttentionMILClassifier.forward().
        normal_class_idx: index of the "Normal" class; windows
            predicted (or labeled) as this class are skipped.
        use_predicted_class: if True (deployment scenario), use the
            model's own argmax prediction to pick which class's
            attention branch to read. If False, use `true_labels`
            instead (useful for offline analysis / localization
            evaluation where you want to decouple localization quality
            from classification correctness).
        true_labels: (batch,) required if use_predicted_class=False.
        threshold: fixed attention-score threshold above which a frame
            is flagged suspicious. Defaults to 1/window_size (i.e.
            "more attention than a uniform distribution would give").
            Ignored if `top_k` is set.
        top_k: if set, flags exactly the top-k highest-attention frames
            per window instead of using a threshold.

    Returns:
        A list (length = batch_size) of dicts:
            {"predicted_class": int, "is_attack": bool,
             "suspicious_frame_indices": list[int],
             "frame_scores": list[float] | None}
    """
    logits = model_output["logits"]
    attn_weights = model_output["attn_weights"]
    batch_size, window_size = attn_weights.shape[0], attn_weights.shape[-1]

    predicted_classes = logits.argmax(dim=1)
    if use_predicted_class:
        class_idx_per_sample = predicted_classes
    else:
        assert true_labels is not None, "true_labels required when use_predicted_class=False"
        class_idx_per_sample = true_labels

    default_threshold = 1.0 / window_size
    results = []
    for b in range(batch_size):
        class_idx = class_idx_per_sample[b].item()
        pred_class = predicted_classes[b].item()

        if class_idx == normal_class_idx:
            results.append({
                "predicted_class": pred_class,
                "is_attack": False,
                "suspicious_frame_indices": [],
                "frame_scores": None,
            })
            continue

        frame_score = attn_weights[b, class_idx, :]  # (window_size,)

        if top_k is not None:
            indices = frame_score.topk(min(top_k, window_size)).indices.tolist()
        else:
            thr = threshold if threshold is not None else default_threshold
            indices = (frame_score > thr).nonzero().squeeze(-1).tolist()

        results.append({
            "predicted_class": pred_class,
            "is_attack": True,
            "suspicious_frame_indices": sorted(indices),
            "frame_scores": frame_score.tolist(),
        })

    return results


# =========================================================================== #
# Integration snippet (for reference -- shows how this plugs into a backbone)
# =========================================================================== #
"""
Example integration into a full CAN-IDS model (replacing a Mean+Max Pool
classifier with this module):

    class CANIDSModel(nn.Module):
        def __init__(self, ..., num_classes):
            super().__init__()
            self.local_encoder = PayloadLocalEncoderPerFrame(...)
            self.retention_encoder = BidirectionalRetentionEncoder(...)
            self.final_ln = nn.LayerNorm(d_model)
            self.classifier = AttentionMILClassifier(d_model, num_classes)  # <-- here

        def forward(self, x, mask=None):
            h = self.local_encoder(x)
            h = self.retention_encoder(h, attn_mask=mask)
            h = self.final_ln(h)
            return self.classifier(h, attn_mask=mask)  # {"logits", "pooled", "attn_weights"}

    # training: only out["logits"] and the window-level label are used
    out = model(x)
    loss = F.cross_entropy(out["logits"], window_label)

    # post-training localization (frame_labels used ONLY here, for evaluation)
    detections = detect_attack_frames(out, normal_class_idx=0, use_predicted_class=False,
                                       true_labels=window_label, top_k=3)
"""


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, window_size, dim = 16, 32, 64
    num_classes = 4  # e.g. Normal, DoS, RPM, Speed
    normal_idx = 0

    h = torch.randn(batch_size, window_size, dim)
    true_labels = torch.randint(0, num_classes, (batch_size,))

    classifier = AttentionMILClassifier(dim=dim, num_classes=num_classes, dropout=0.1)
    out = classifier(h)

    print(f"Input h shape:        {tuple(h.shape)}")
    print(f"logits shape:         {tuple(out['logits'].shape)}")
    print(f"pooled shape:         {tuple(out['pooled'].shape)}")
    print(f"attn_weights shape:   {tuple(out['attn_weights'].shape)}")
    assert out["logits"].shape == (batch_size, num_classes)
    assert out["pooled"].shape == (batch_size, num_classes, dim)
    assert out["attn_weights"].shape == (batch_size, num_classes, window_size)

    # each attention row must sum to 1 (softmax over the window)
    row_sums = out["attn_weights"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    print("attn_weights rows sum to 1: True")

    # standard training step
    loss = nn.functional.cross_entropy(out["logits"], true_labels)
    loss.backward()
    print(f"\nCrossEntropyLoss: {loss.item():.4f}")
    grad_norms = [
        classifier.bag_classifiers[n].weight.grad.norm().item() for n in range(num_classes)
    ]
    print(f"Per-class FC head gradient norms (should differ -> independent heads): "
          f"{[f'{g:.4f}' for g in grad_norms]}")

    # padding mask
    mask = torch.ones(batch_size, window_size)
    mask[:, 28:] = 0
    out_masked = classifier(h, attn_mask=mask)
    assert out_masked["attn_weights"][:, :, 28:].max().item() < 1e-4
    print("Padding mask: masked positions receive ~0 attention: True")

    # pseudo-label utility
    pseudo = AttentionWeightedPooling.get_pseudo_labels(out["attn_weights"], k=4)
    print(f"\npseudo_labels shape: {tuple(pseudo.shape)}, unique values: {pseudo.unique().tolist()}")

    # --- frame-level localization utility ---
    print("\n=== detect_attack_frames demo ===")
    detections_by_true_label = detect_attack_frames(
        out, normal_class_idx=normal_idx, use_predicted_class=False,
        true_labels=true_labels, top_k=3,
    )
    for i in range(3):
        d = detections_by_true_label[i]
        print(f"window {i}: true_label={true_labels[i].item()}, "
              f"is_attack={d['is_attack']}, "
              f"suspicious_frames={d['suspicious_frame_indices']}")

    detections_by_pred = detect_attack_frames(
        out, normal_class_idx=normal_idx, use_predicted_class=True, threshold=None,
    )
    n_flagged_attack = sum(d["is_attack"] for d in detections_by_pred)
    print(f"\n(using predicted class) windows flagged as attack: {n_flagged_attack}/{batch_size}")

    # MLP head variant
    print("\n=== MLP bag-classifier head variant ===")
    classifier_mlp = AttentionMILClassifier(
        dim=dim, num_classes=num_classes, classifier_hidden_dim=32, classifier_dropout=0.1,
    )
    out_mlp = classifier_mlp(h)
    assert out_mlp["logits"].shape == (batch_size, num_classes)
    print(f"MLP-head logits shape: {tuple(out_mlp['logits'].shape)}")

    n_params_linear = sum(p.numel() for p in classifier.parameters())
    n_params_mlp = sum(p.numel() for p in classifier_mlp.parameters())
    print(f"\nParameters (Linear heads): {n_params_linear:,}")
    print(f"Parameters (MLP heads):    {n_params_mlp:,}")
