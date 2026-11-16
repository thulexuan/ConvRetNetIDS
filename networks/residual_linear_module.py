import torch
import torch.nn as nn


class ResidualLinearBlock(nn.Module):
    """
    Per-frame residual block using nn.Linear.

    Equivalent to ResidualConvBlock1D with kernel_size=1, but operates
    directly on (B, T, C) without transposing to channel-first.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)

        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Projection shortcut only when feature dimensions differ
        # (e.g. the very first block: in_features=9 -> out_features).
        if in_features != out_features:
            self.shortcut_fc = nn.Linear(in_features, out_features)
            self.shortcut_bn = nn.BatchNorm1d(out_features)
        else:
            self.shortcut_fc = None
            self.shortcut_bn = None

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _bn(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> BatchNorm over all B*T frames per channel -> (B, T, C)
        B, T, C = x.shape
        return bn(x.reshape(B * T, C)).reshape(B, T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_features)
        if self.shortcut_fc is not None:
            identity = self._bn(self.shortcut_bn, self.shortcut_fc(x))  # (B, T, out_features)
        else:
            identity = x

        out = self.relu(self._bn(self.bn1, self.fc1(x)))
        out = self._bn(self.bn2, self.fc2(out))

        out = out + identity
        out = self.relu(out)
        out = self.dropout(out)
        return out


class PayloadLocalEncoderPerFrameLinear(nn.Module):
    """
    Local payload encoder operating independently on each CAN frame,
    implemented with nn.Linear (residual MLP per frame).

    Equivalent to PayloadLocalEncoderPerFrame with kernel_size=1.

    Input:
        payload_features: (B, T, 9)

    Output:
        h: (B, T, out_features)
    """

    def __init__(
        self,
        in_features: int = 9,
        out_features: int = 128,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        blocks = []
        for i in range(n_res_blocks):
            block_in_features = in_features if i == 0 else out_features
            blocks.append(
                ResidualLinearBlock(
                    in_features=block_in_features,
                    out_features=out_features,
                    dropout=dropout,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, payload_features):
        """
        Args:
            payload_features:
                Tensor with shape (B, T, 9)

        Returns:
            h:
                Tensor with shape (B, T, out_features)
        """

        h = self.res_blocks(payload_features)  # (B, T, out_features)

        return h
