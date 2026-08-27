import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style Channel Attention Mechanism (CAM), as used
    in STDLM to dynamically re-weight fused spatio-temporal feature channels
    before classification, emphasizing attack-relevant channels while
    attenuating redundant ones.
    """

    def __init__(self, d_model: int, reduction: int = 8):
        super().__init__()
        hidden = max(d_model // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, d_model) fused, channel-attended vector."""

        squeeze = x.mean(dim=1)                 # global average pool over time
        weights = self.fc(squeeze)                # (B, d_model) channel weights
        scaled = x * weights.unsqueeze(1)          # re-weight every timestep
        return scaled                 # pooled fused feature (B, T, d_model) - can be further pooled if needed