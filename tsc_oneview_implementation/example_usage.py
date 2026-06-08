"""
Example usage for TSC-1View.

Replace `YourEncoder` and dataloaders with your own cct_retnet pipeline.

Your train_loader should return:
    images, labels

Your encoder should return:
    embeddings
or:
    logits, embeddings
or:
    {"features": embeddings}
"""

import torch
import torch.nn as nn

from train_tsc_oneview_stage1 import train_tsc_oneview_stage1
from train_linear_stage2 import train_linear_stage2


class YourEncoder(nn.Module):
    """
    Dummy encoder example.
    Replace this with cct_retnet backbone.
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return z


def main():
    num_classes = 10
    emb_dim = 128

    # Replace these with your actual dataloaders.
    train_loader = None
    val_loader = None

    encoder = YourEncoder(emb_dim=emb_dim)

    optimizer = torch.optim.SGD(
        encoder.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=1e-4,
    )

    # Stage 1: train embedding space with one-view TSC.
    result = train_tsc_oneview_stage1(
        encoder=encoder,
        train_loader=train_loader,
        optimizer=optimizer,
        num_classes=num_classes,
        emb_dim=emb_dim,
        device="cuda",
        epochs=100,
        tau=0.1,
        k=6,
        lambda_target=1.0,
        warmup_epochs=0,
        save_path="tsc_oneview_stage1.pt",
    )

    # Stage 2: freeze encoder and train classifier.
    classifier, best_acc = train_linear_stage2(
        encoder=result["encoder"],
        train_loader=train_loader,
        val_loader=val_loader,
        emb_dim=emb_dim,
        num_classes=num_classes,
        device="cuda",
        epochs=50,
        lr=0.01,
        save_path="tsc_oneview_linear.pt",
    )

    print("Best validation accuracy:", best_acc)


if __name__ == "__main__":
    main()
