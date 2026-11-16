import torch
import torch.nn as nn


class PayloadFeatureConvEncoder(nn.Module):
    """
    Local payload encoder operating independently on each CAN frame.

    Per frame:
        (1, 9)  --Linear-->  (1, 32)  --Conv1d(k=3, no padding)-->  (32, 30)
                --flatten-->  (960,)  --Linear-->  (128,)

    Frames are folded into the batch dimension, so there is no mixing along
    the time axis.

    Input:
        payload_features: (B, T, in_channels)

    Output:
        h: (B, T, out_channels)

    """

    def __init__(
        self,
        in_channels: int = 9,  # number of per-frame features (CAN ID + 8 payload bytes)
        out_channels: int = 128,
        embed_dim: int = 32,  # length after the first Linear
        conv_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, embed_dim)

        # No padding: length goes from embed_dim to embed_dim - kernel_size + 1 (32 -> 30)
        self.conv = nn.Conv1d(1, conv_channels, kernel_size)
        self.bn = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        conv_out_len = embed_dim - kernel_size + 1
        self.output_proj = nn.Linear(conv_channels * conv_out_len, out_channels)

    def forward(self, payload_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            payload_features:
                Tensor with shape (B, T, in_channels)

        Returns:
            h:
                Tensor with shape (B, T, out_channels)
        """
        B, T, n_features = payload_features.shape

        x = payload_features.reshape(B * T, n_features)  # (B*T, 9)
        x = self.input_proj(x)  # (B*T, 32)
        x = x.unsqueeze(1)  # (B*T, 1, 32)

        h = self.dropout(self.relu(self.bn(self.conv(x))))  # (B*T, 32, 30)

        h = h.flatten(1)  # (B*T, 32 * 30)
        h = self.output_proj(h)  # (B*T, out_channels)

        h = h.reshape(B, T, -1)  # (B, T, out_channels)

        return h
