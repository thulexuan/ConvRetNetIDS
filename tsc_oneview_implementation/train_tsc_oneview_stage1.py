"""
Generic stage-1 training loop for TSC-1View.

Expected dataloader output:
    (images, labels)
or:
    (images, labels, extra...)

Expected encoder output:
    features
or:
    (logits, features)
or:
    {"features": features}
or:
    {"embedding": features}

The stage-1 goal is representation learning. The classifier can be trained later in stage 2.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

from tsc_oneview_loss import TSCOneViewLoss


def get_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract embedding from a model output.

    Modify this function if your cct_retnet returns features in a different format.
    """
    out = model(x)

    if torch.is_tensor(out):
        return out

    if isinstance(out, (tuple, list)):
        # Common case: return logits, features
        return out[-1]

    if isinstance(out, dict):
        for key in ["features", "feature", "feat", "embedding", "emb", "z"]:
            if key in out:
                return out[key]

    raise TypeError(
        "Cannot extract embedding from model output. "
        "Please edit get_embedding() to match your model output."
    )


def parse_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parse batch from dataloader.

    Supports:
        (images, labels)
        (images, labels, ...)
        {"image": images, "label": labels}
    """
    if isinstance(batch, dict):
        x = batch.get("image", batch.get("images", batch.get("x", None)))
        y = batch.get("label", batch.get("labels", batch.get("y", None)))
        if x is None or y is None:
            raise ValueError("Dict batch must contain image/images/x and label/labels/y.")
        return x, y

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise ValueError("Batch must be (images, labels) or dict with image/label.")


def train_tsc_oneview_stage1(
    encoder: nn.Module,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    emb_dim: int,
    device: str | torch.device = "cuda",
    epochs: int = 100,
    tau: float = 0.1,
    k: int = 6,
    lambda_target: float = 1.0,
    center_momentum: float = 0.9,
    warmup_epochs: int = 0,
    target_generation_steps: int = 2000,
    target_generation_lr: float = 0.05,
    save_path: Optional[str] = None,
    print_freq: int = 50,
) -> Dict[str, Any]:
    """
    Train encoder with one-view TSC loss.
    """
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    encoder = encoder.to(device)

    criterion = TSCOneViewLoss(
        num_classes=num_classes,
        emb_dim=emb_dim,
        tau=tau,
        k=k,
        lambda_target=lambda_target,
        center_momentum=center_momentum,
        target_generation_steps=target_generation_steps,
        target_generation_lr=target_generation_lr,
    ).to(device)

    history = []

    for epoch in range(epochs):
        encoder.train()
        use_target_term = epoch >= warmup_epochs

        total_loss = 0.0
        total_same = 0.0
        total_target = 0.0
        total_pos = 0
        n_batches = 0

        for it, batch in enumerate(train_loader):
            x, y = parse_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            features = get_embedding(encoder, x)
            output = criterion(
                features=features,
                labels=y,
                update_centers=True,
                use_target_term=use_target_term,
                use_same_class_term=True,
            )

            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            optimizer.step()

            total_loss += float(output.loss.detach().cpu())
            total_same += float(output.same_class_loss.cpu())
            total_target += float(output.target_loss.cpu())
            total_pos += int(output.num_positive_pairs)
            n_batches += 1

            if print_freq and (it % print_freq == 0):
                print(
                    f"Epoch [{epoch+1:03d}/{epochs:03d}] "
                    f"Iter [{it:04d}] "
                    f"loss={output.loss.item():.4f} "
                    f"same={output.same_class_loss.item():.4f} "
                    f"target={output.target_loss.item():.4f} "
                    f"pos_pairs={output.num_positive_pairs} "
                    f"use_target={use_target_term}"
                )

        epoch_log = {
            "epoch": epoch + 1,
            "loss": total_loss / max(1, n_batches),
            "same_class_loss": total_same / max(1, n_batches),
            "target_loss": total_target / max(1, n_batches),
            "positive_pairs_per_batch": total_pos / max(1, n_batches),
            "use_target_term": use_target_term,
        }
        history.append(epoch_log)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"loss={epoch_log['loss']:.4f}, "
            f"same={epoch_log['same_class_loss']:.4f}, "
            f"target={epoch_log['target_loss']:.4f}, "
            f"pos/batch={epoch_log['positive_pairs_per_batch']:.1f}"
        )

        if save_path is not None:
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "criterion": criterion.state_dict(),
                    "history": history,
                    "epoch": epoch + 1,
                    "num_classes": num_classes,
                    "emb_dim": emb_dim,
                    "tau": tau,
                    "k": k,
                    "lambda_target": lambda_target,
                },
                save_path,
            )

    return {
        "encoder": encoder,
        "criterion": criterion,
        "history": history,
    }
