import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock1D(nn.Module):
    """
    Single-channel residual convolutional block used inside every branch
    of the MC-1DCNN (Fig. 2 / Eq. 4-5 of the paper).
 
    Two Conv1d -> BatchNorm -> ReLU stages, with a residual (skip)
    connection added around them, followed by dropout.
    'same' padding is used so the temporal length T never changes,
    which is required so the residual add is shape-compatible along T.
 
    NOTE:
        When `in_channels != out_channels`, a plain identity skip is no
        longer shape-compatible on the CHANNEL dimension (only T is kept
        the same by padding). In that case a 1x1 Conv1d + BatchNorm
        "shortcut" projection is used to map the input to `out_channels`
        before the residual addition (standard ResNet-style projection
        shortcut, He et al., 2016). When `in_channels == out_channels`,
        the shortcut is a plain identity (no extra parameters), matching
        Eq. (5) of the paper exactly.
    """
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2  # keeps T unchanged for odd kernel sizes
 
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
 
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
 
        # Projection shortcut only when channel dimensions differ
        # (e.g. the very first block: in_channels=9 -> out_channels).
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
 
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T)
        identity = self.shortcut(x)  # (B, out_channels, T)
 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
 
        out = out + identity
        out = self.relu(out)
        out = self.dropout(out)
        return out
 
 
class PayloadLocalEncoderPerFrame(nn.Module):
    """
    Local payload encoder operating independently on each CAN frame.
 
    Input:
        payload_features: (B, T, 9)
 
    Output:
        h: (B, T, out_channels)
 
    Each CAN frame contains 9 features (e.g. 1 CAN ID + 8 payload bytes).
    All Conv1d layers use kernel_size=1, therefore no temporal mixing
    occurs between neighboring CAN frames.
 
    Channel schedule of the stacked residual blocks:
        - 1st block : in_channels=9            -> out_channels
        - blocks 2..n_res_blocks : in_channels=out_channels -> out_channels
    """
 
    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 128,
        kernel_size: int = 1,
        n_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
 
        blocks = []
        for i in range(n_res_blocks):
            block_in_channels = in_channels if i == 0 else out_channels
            blocks.append(
                ResidualConvBlock1D(
                    in_channels=block_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)
 
        self.norm = nn.BatchNorm1d(out_channels)
 
    def forward(self, payload_features):
        """
        Args:
            payload_features:
                Tensor with shape (B, T, 9)
 
        Returns:
            h:
                Tensor with shape (B, T, out_channels)
        """
 
        x = payload_features.transpose(1, 2)  # (B, 9, T)
 
        h = self.res_blocks(x)  # (B, out_channels, T)
 
        h = h.transpose(1, 2)  # (B, T, out_channels)
 
        return h