"""
Example training loop with model and loss separated.
"""

import torch
import torch.nn as nn

from builder_oneview_model import KCLOneViewModel
from losses_oneview import KCLTSCCriterion


class YourBaseEncoder(nn.Module):
    """
    Dummy encoder. Replace this with your real backbone.
    It follows original builder.py convention:
        base_encoder(num_classes=dim)
        forward(x, True) returns q, feat
    """

    def __init__(self, num_classes=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 512),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.backbone(x)
        q = self.fc(feat)
        if return_feat:
            return q, feat
        return q


def main():
    model = KCLOneViewModel(
        base_encoder=YourBaseEncoder,
        dim=128,
        K=1024,
        T=0.07,
        num_positive=6,
        targeted=True,
        n_cls=10,
        tr=1,
    )

    criterion = KCLTSCCriterion(
        target_weight=1.0,
        original_style_weighting=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.03,
        momentum=0.9,
        weight_decay=1e-4,
    )

    images = torch.randn(32, 3, 64, 64)
    labels = torch.randint(0, 10, (32,))

    output = model(images, labels)
    loss_output = criterion(
        logits=output["logits"],
        mask_class=output["mask_class"],
        mask_target=output["mask_target"],
    )

    loss = loss_output["loss"]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss:", float(loss.detach()))
    print("loss_class:", float(loss_output["loss_class"]))
    print("loss_target:", float(loss_output["loss_target"]))


if __name__ == "__main__":
    main()
