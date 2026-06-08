"""
Stage-2 linear classifier training after TSC-1View representation learning.

The encoder is frozen. A linear classifier is trained on top of embeddings.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_tsc_oneview_stage1 import get_embedding, parse_batch


@torch.no_grad()
def evaluate_linear(
    encoder: nn.Module,
    classifier: nn.Module,
    data_loader: Iterable,
    device: str | torch.device = "cuda",
) -> float:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    for batch in data_loader:
        x, y = parse_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        z = get_embedding(encoder, x)
        z = F.normalize(z, p=2, dim=1)
        logits = classifier(z)

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return correct / max(1, total)


def train_linear_stage2(
    encoder: nn.Module,
    train_loader: Iterable,
    val_loader: Optional[Iterable],
    emb_dim: int,
    num_classes: int,
    device: str | torch.device = "cuda",
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> Tuple[nn.Module, float]:
    """
    Freeze encoder and train a linear classifier.
    """
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    encoder = encoder.to(device)
    classifier = nn.Linear(emb_dim, num_classes).to(device)

    for p in encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    best_acc = 0.0

    for epoch in range(epochs):
        encoder.eval()
        classifier.train()

        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y = parse_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            with torch.no_grad():
                z = get_embedding(encoder, x)
                z = F.normalize(z, p=2, dim=1)

            logits = classifier(z)
            loss = F.cross_entropy(logits, y, weight=class_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        if val_loader is not None:
            val_acc = evaluate_linear(encoder, classifier, val_loader, device=device)
        else:
            val_acc = 0.0

        best_acc = max(best_acc, val_acc)

        print(
            f"[Linear Epoch {epoch+1:03d}/{epochs:03d}] "
            f"loss={total_loss/max(1,n_batches):.4f}, "
            f"val_acc={val_acc:.4f}, best={best_acc:.4f}"
        )

        if save_path is not None:
            torch.save(
                {
                    "classifier": classifier.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch + 1,
                },
                save_path,
            )

    return classifier, best_acc
