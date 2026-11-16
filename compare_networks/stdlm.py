"""
STDLM: Spatio-Temporal Deep Learning Method for anomaly detection in
in-vehicle CAN networks.

Reference:
Xiong, X., Zhang, J., Wu, Y., Huang, Q., Guo, X. (2026).
"STDLM: A spatio-temporal deep learning method for anomaly detection in
in-vehicle CAN networks." Journal of Network and Computer Applications, 251, 104489.

Architecture (4 modules, per Fig. 1 of the paper):
    1. Data preprocessing (done outside this file -> produces normalized
       windows of shape (t, 9): 1 CAN-ID channel + 8 payload-byte channels).
    2. MC-1DCNN  : Multi-Channel 1D CNN with adaptive kernels (Sec. 3.2)
                   -> spatial feature extraction, one branch per input
                   channel, residual blocks (Fig. 2).
    3. MC-GTCN   : Multi-Channel Gated Temporal Convolutional Network
                   (Sec. 3.3) -> dilated causal convolutions with a Gated
                   Linear Unit in the first residual block (Fig. 3/4),
                   preserving the "multi-channel" identity of the 9
                   branches via grouped convolutions (Eq. 9).
    4. Channel Attention + Classifier (Sec. 3.4) -> SE-style channel
       re-weighting, followed by a fully-connected softmax classifier.

INPUT  : tensor of shape (batch, t, 9)  -- normalized CAN window
         (t = number of time steps / frames in the sliding window,
          9 = 1 CAN-ID channel + 8 payload-byte channels, values in [0,1])
OUTPUT : tensor of shape (batch, num_classes) -- raw logits
         (apply Softmax externally, or use nn.CrossEntropyLoss which
          expects logits directly)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class STDLMConfig:
    num_channels: int = 9          # 1 CAN ID + 8 payload bytes (fixed by protocol)
    window_size: int = 32          # t, number of time steps per window
    num_classes: int = 10           # 2 for binary (normal/attack), N for multi-class

    # MC-1DCNN (spatial) params
    spatial_hidden: int = 7       # feature maps produced per branch
    id_kernel: int = 3             # kernel size for the CAN ID channel   (Eq. 3)
    payload_kernel: int = 5        # kernel size for the 8 payload channels (Eq. 3)
    spatial_dropout: float = 0.2

    # MC-GTCN (temporal) params
    temporal_hidden: int = 7      # feature maps per branch inside MC-GTCN
    tcn_kernel: int = 3            # K in Eq. 6-8
    tcn_levels: int = 4            # number of stacked dilated residual blocks
    temporal_dropout: float = 0.2

    # Channel attention params
    attn_reduction: int = 4        # squeeze ratio in the SE-style block


# --------------------------------------------------------------------------- #
# 1) MC-1DCNN -- Spatial feature extraction (Sec. 3.2, Fig. 2)
# --------------------------------------------------------------------------- #
class SpatialResidualBranch(nn.Module):
    """
    Single-channel residual block (Fig. 2): two stacked 1D-conv layers with
    an adaptive kernel size K_m (Eq. 3-4), BatchNorm, ReLU, a residual
    (skip) connection (Eq. 5), and Dropout before the next stage.
    """

    def __init__(self, kernel_size: int, hidden_dim: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding to preserve sequence length t

        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 1x1 projection so the residual add works (in_channels=1 -> hidden_dim)
        self.shortcut = nn.Conv1d(1, hidden_dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, t)
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = F.relu(out + identity)      # Eq. 5: y_block = x_input + F(x_input)
        out = self.dropout(out)
        return out                        # (B, hidden_dim, t)


class MC1DCNN(nn.Module):
    """
    Multi-Channel 1D CNN (Sec. 3.2.1). Runs `num_channels` independent
    parallel branches (one per input feature channel: CAN ID + 8 payload
    bytes), each with its own adaptive kernel size, then concatenates all
    branch outputs to form the unified spatial feature map used as input
    to the temporal stage.
    """

    def __init__(self, cfg: STDLMConfig):
        super().__init__()
        self.num_channels = cfg.num_channels

        kernels = [cfg.id_kernel] + [cfg.payload_kernel] * (cfg.num_channels - 1)
        self.branches = nn.ModuleList(
            [
                SpatialResidualBranch(k, cfg.spatial_hidden, cfg.spatial_dropout)
                for k in kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_channels, t)  -- one raw scalar sequence per channel
        branch_outputs = [
            branch(x[:, i : i + 1, :]) for i, branch in enumerate(self.branches)
        ]
        # concat along channel dim -> (B, num_channels * spatial_hidden, t)
        return torch.cat(branch_outputs, dim=1)


# --------------------------------------------------------------------------- #
# 2) MC-GTCN -- Temporal feature extraction (Sec. 3.3, Fig. 3/4)
# --------------------------------------------------------------------------- #
class CausalConv1d(nn.Module):
    """Dilated causal 1D convolution (Eq. 6-8): output at t depends only on
    inputs at time steps <= t. Implemented via left-only zero padding.

    NOTE (stability fix, round 2): weight_norm was removed. weight_norm
    reparameterizes w = g * v/||v||, and `g` (magnitude) is otherwise
    unconstrained -- across many epochs it can drift upward slowly (only
    weakly discouraged by weight_decay). Once `g` grows large enough, the
    raw conv output can overflow float32 *before* GroupNorm ever sees it;
    GroupNorm computes mean/var over its input, so if that input already
    contains inf, GroupNorm itself turns it into NaN instead of fixing it.
    This explained why the previous fix (GroupNorm only) pushed the NaN
    from epoch 4 to epoch 47 instead of eliminating it: GroupNorm helped
    a lot, but the unconstrained weight_norm magnitude was still slowly
    working against it. A plain nn.Conv1d + GroupNorm is sufficient here
    since GroupNorm already provides the activation-scale stabilization
    that weight_norm was originally meant to help with.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, groups=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=self.pad,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.pad != 0:
            out = out[:, :, : -self.pad]  # trim the right side -> causal
        return out


class GatedResidualBlock(nn.Module):
    """
    One residual unit of the MC-GTCN (Fig. 4).

    - Sub-layer 1 (gated): multi-channel dilated causal conv -> GroupNorm
      -> ReLU -> Dropout, run through TWO parallel paths (value path +
      gate path). The gate path is squashed with Sigmoid; a simplified
      Gated Linear Unit combines them (Eq. 12): y = (W1*x+b1) * sigmoid(W2*x+b2).
    - Sub-layer 2 (plain): another dilated causal conv -> GroupNorm ->
      ReLU -> Dropout, WITHOUT gating (preserves feature identity, Sec. 3.3.2).
    - Residual connection: a 1x1 conv projects the block input to match
      channel dims, then is added to the sub-layer-2 output.

    `groups` = num_channels keeps the "multi-channel" branch identity
    (Eq. 9: per-branch dilated conv, concatenated) via grouped convolution,
    so ID-branch and payload-branch dynamics are learned semi-independently.

    NOTE (stability fix): the original draft relied only on WeightNorm
    inside CausalConv1d, with no normalization on the *activations*
    flowing through the residual stack. WeightNorm only constrains the
    *direction* of a weight (w = g * v/||v||); the magnitude g is
    unconstrained and can grow across epochs. Combined with 4 stacked
    residual adds (dilation 1-2-4-8) and no activation normalization,
    this let activation magnitudes drift upward over training until they
    overflowed float32 -> inf -> NaN loss (loss decreasing steadily, then
    suddenly NaN -- classic exploding-activation signature). Adding
    GroupNorm after every dilated conv keeps activation scale bounded at
    every block, independent of the learning rate/weight-decay settings
    used for the optimizer.
    """

    def __init__(self, channels, kernel_size, dilation, dropout, groups, use_gate):
        super().__init__()
        self.use_gate = use_gate

        self.conv1_val = CausalConv1d(channels, channels, kernel_size, dilation, groups)
        self.norm1_val = nn.GroupNorm(groups, channels)
        self.drop1 = nn.Dropout(dropout)

        if use_gate:
            self.conv1_gate = CausalConv1d(channels, channels, kernel_size, dilation, groups)
            self.norm1_gate = nn.GroupNorm(groups, channels)
            self.drop1_gate = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation, groups)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        val = self.drop1(F.relu(self.norm1_val(self.conv1_val(x))))

        if self.use_gate:
            gate = self.drop1_gate(F.relu(self.norm1_gate(self.conv1_gate(x))))
            out = val * torch.sigmoid(gate)          # Eq. 12 (simplified GLU)
        else:
            out = val

        out = self.drop2(F.relu(self.norm2(self.conv2(out))))
        return F.relu(out + identity)                # residual add


class MCGTCN(nn.Module):
    """
    Multi-Channel Gated Temporal Convolutional Network (Sec. 3.3).
    Stacks `tcn_levels` GatedResidualBlocks with exponentially increasing
    dilation d = 2^l (Eq. 6), growing the receptive field logarithmically
    (Eq. 7). Only the FIRST block uses the GLU gate (Sec. 3.3.2); later
    blocks use a plain residual path to preserve feature identity.
    """

    def __init__(self, in_channels: int, cfg: STDLMConfig):
        super().__init__()
        # project spatial features down to the per-branch temporal_hidden size
        groups = cfg.num_channels
        proj_channels = cfg.num_channels * cfg.temporal_hidden

        self.input_proj = nn.Conv1d(in_channels, proj_channels, kernel_size=1, groups=groups)

        blocks = []
        for level in range(cfg.tcn_levels):
            dilation = 2 ** level  # Eq. 6: d^(l) = 2^l
            blocks.append(
                GatedResidualBlock(
                    channels=proj_channels,
                    kernel_size=cfg.tcn_kernel,
                    dilation=dilation,
                    dropout=cfg.temporal_dropout,
                    groups=groups,
                    use_gate=(level == 0),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = proj_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        return self.blocks(x)  # (B, proj_channels, t)


# --------------------------------------------------------------------------- #
# 3) Channel Attention Mechanism (Sec. 3.4, Fig. 7) -- SE-style
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    """
    Squeeze-and-excitation style channel attention (Ning et al., 2021a /
    Hu et al., 2018, as cited in the paper). Learns a per-channel weight
    vector that emphasizes attack-relevant channels and suppresses
    redundant ones, before the final classification layer.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # F_sq: global average pool
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, t)
        b, c, _ = x.shape
        z = self.squeeze(x).view(b, c)      # F_sq
        scale = self.excite(z).view(b, c, 1)  # F_ex -> per-channel weights
        return x * scale                    # F_scale: reweight channels


# --------------------------------------------------------------------------- #
# 4) Full STDLM model
# --------------------------------------------------------------------------- #
class STDLM(nn.Module):
    """
    Full pipeline:
        (B, t, 9) --permute--> (B, 9, t)
            --> MC-1DCNN   --> (B, 9*spatial_hidden, t)
            --> MC-GTCN    --> (B, 9*temporal_hidden, t)
            --> ChannelAttention (reweight channels)
            --> Global Average Pool over time --> (B, C)
            --> Linear classifier --> (B, num_classes)  [logits]
    """

    def __init__(self, cfg: STDLMConfig | None = None):
        super().__init__()
        self.cfg = cfg or STDLMConfig()

        self.mc1dcnn = MC1DCNN(self.cfg)
        spatial_out_channels = self.cfg.num_channels * self.cfg.spatial_hidden

        self.mcgtcn = MCGTCN(spatial_out_channels, self.cfg)
        temporal_out_channels = self.mcgtcn.out_channels

        self.attention = ChannelAttention(temporal_out_channels, self.cfg.attn_reduction)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(temporal_out_channels, self.cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, t, 9) normalized CAN window
               (t = window_size time steps, 9 = CAN-ID + 8 payload bytes)
        Returns:
            logits: (batch, num_classes)
        """
        x = x[:, :, 0:9]
        x = x.permute(0, 2, 1)          # -> (B, 9, t) for Conv1d
        spatial = self.mc1dcnn(x)       # -> (B, 9*spatial_hidden, t)
        temporal = self.mcgtcn(spatial) # -> (B, 9*temporal_hidden, t)
        attended = self.attention(temporal)
        pooled = self.pool(attended).squeeze(-1)  # (B, C)
        logits = self.classifier(pooled)          # (B, num_classes)
        return logits


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = STDLMConfig(window_size=29, num_classes=2)
    model = STDLM(cfg)

    batch_size = 8
    dummy_input = torch.rand(batch_size, cfg.window_size, cfg.num_channels)  # in [0,1]
    logits = model(dummy_input)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"Input shape : {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Total trainable parameters: {n_params:,}")

    # Example loss computation (binary classification, matches Eq. 15/16 via CE)
    labels = torch.randint(0, cfg.num_classes, (batch_size,))
    loss = F.cross_entropy(logits, labels)
    print(f"Sample cross-entropy loss: {loss.item():.4f}")

    # ----------------------------------------------------------------- #
    # Reference training step (recommended defaults after the NaN-loss
    # debugging above). GroupNorm + plain conv (no weight_norm) now keep
    # activations bounded through many epochs. Gradient clipping is kept
    # as a cheap extra safety net, and a decaying LR schedule further
    # reduces the risk of a late-training instability once the loss has
    # mostly converged and gradients could otherwise take large steps.
    # ----------------------------------------------------------------- #
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    print(f"Gradient norm (post-clip target 1.0): {grad_norm:.4f}")