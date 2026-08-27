"""
model.py
========
Spatio-Temporal Intrusion Detection Model for In-Vehicle CAN Networks.

This module combines two architectures from prior work into a single hybrid
model for network-wide CAN anomaly detection:

  * MC-1DCNN  (Multi-Channel 1D CNN, from STDLM [Xiong et al., 2026])
    Extracts per-field SPATIAL features from a CAN window. The CAN ID field
    and the 8 payload bytes are each processed by an independent 1D-conv
    branch with an adaptive kernel size (small kernel for the fast-changing
    CAN ID, larger kernel for the more correlated payload bytes), following
    residual convolution blocks for stable deep feature extraction.

  * Multi-Scale Retention (from RetNet [Sun et al., 2023])
    Replaces STDLM's MC-GTCN for TEMPORAL modeling. Retention offers a dual
    form: a parallel representation (used here for training on fixed CAN
    windows, fully parallelizable like self-attention) and a recurrent
    representation with O(1) per-step cost and constant memory, which is a
    natural fit for streaming, real-time intrusion detection on
    resource-constrained ECUs.

Pipeline
--------
    Raw CAN window (B, T, 9)                      T = sliding-window length
                                                    9 = 1 CAN-ID + 8 payload bytes
        -> MC1DCNN                 (B, T, d_model)  spatial feature per timestep
        -> [RetentionBlock] x L    (B, T, d_model)  temporal feature per timestep
        -> ChannelAttention (CAM)  (B, d_model)     attention-weighted fusion
        -> Classifier               (B, num_classes) logits

Two inference modes are provided:
  - `forward(x)`      : batched / parallel, for training and offline evaluation.
  - `stream_step(...)`: O(1) recurrent single-frame update, for on-vehicle
                        real-time deployment.

Expected preprocessing (as in STDLM): CAN ID and payload bytes are decoded
from hex to decimal and normalized with logarithmic min-max normalization to
[0, 1] before being fed to this model. Channel 0 = CAN ID, channels 1..8 =
payload bytes d1..d8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ #
# Configuration
# ============================================================================ #

@dataclass
class CANIDSConfig:
    # --- input ---
    num_channels: int = 9          # 1 CAN-ID channel + 8 payload-byte channels
    window_size: int = 32          # T, CAN frames per sliding window

    # --- MC-1DCNN (spatial feature extraction) ---
    spatial_hidden: int = 16       # feature maps per channel branch
    id_kernel_size: int = 3        # kernel for CAN ID channel      (m = 0)
    payload_kernel_size: int = 5   # kernel for payload byte channels (m = 1..8)
    spatial_num_blocks: int = 2    # residual conv blocks per branch
    spatial_dropout: float = 0.2

    # --- shared model / temporal dimension ---
    d_model: int = 144

    # --- Multi-Scale Retention (temporal feature extraction) ---
    num_retention_layers: int = 2
    num_heads: int = 4
    ffn_hidden_mult: int = 2
    retention_dropout: float = 0.1

    # --- Channel Attention (CAM) + classifier ---
    cam_reduction: int = 8
    num_classes: int = 10           # 2 = binary (normal/attack); >2 = attack-type multiclass
    classifier_dropout: float = 0.2

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, (
            "d_model must be divisible by num_heads for multi-scale retention"
        )


# ============================================================================ #
# MC-1DCNN: Multi-Channel 1D CNN spatial feature extractor
# ============================================================================ #

class ResidualConv1DBlock(nn.Module):
    """Residual 1D-conv block used inside one channel branch of the MC-1DCNN.

    conv -> BN -> ReLU -> conv -> BN -> (+ input) -> ReLU -> Dropout

    Same-padding is used throughout so the sequence length T is preserved,
    which keeps the branch's output alignable with the other branches.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return self.dropout(out)


class ChannelBranch(nn.Module):
    """One parallel branch of the MC-1DCNN, processing a single CAN field
    (the CAN ID or one payload byte) across the time dimension with a
    field-specific ("adaptive") kernel size.
    """

    def __init__(self, hidden: int, kernel_size: int, num_blocks: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.input_conv = nn.Conv1d(1, hidden, kernel_size, padding=padding)
        self.input_bn = nn.BatchNorm1d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            ResidualConv1DBlock(hidden, kernel_size, dropout) for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) -> (B, hidden, T)
        out = self.act(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            out = block(out)
        return out


class MC1DCNN(nn.Module):
    """Multi-Channel 1D CNN spatial feature extractor.

    Field 0 is the CAN ID (fast-changing, single scalar -> small receptive
    field); fields 1..num_channels-1 are payload bytes (patterns often span
    multiple positions -> larger receptive field):

        K_m = id_kernel_size        if m == 0  (CAN ID)
        K_m = payload_kernel_size   if m >= 1  (payload bytes)

    All channel branches are computed in parallel and concatenated at every
    timestep, then linearly projected to `d_model` to form the input sequence
    for the temporal (retention) stage.
    """

    def __init__(self, cfg: CANIDSConfig):
        super().__init__()
        self.num_channels = cfg.num_channels
        self.branches = nn.ModuleList()
        for m in range(cfg.num_channels):
            k = cfg.id_kernel_size if m == 0 else cfg.payload_kernel_size
            self.branches.append(
                ChannelBranch(cfg.spatial_hidden, k, cfg.spatial_num_blocks, cfg.spatial_dropout)
            )
        fused_dim = cfg.num_channels * cfg.spatial_hidden
        self.proj = nn.Linear(fused_dim, cfg.d_model)
        self.proj_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, num_channels) normalized CAN window.
        returns: (B, T, d_model) spatial feature sequence.
        """
        assert x.size(-1) == self.num_channels, (
            f"expected {self.num_channels} input channels (CAN ID + payload bytes), "
            f"got {x.size(-1)}"
        )
        feats = []
        for m, branch in enumerate(self.branches):
            xm = x[:, :, m].unsqueeze(1)   # (B, 1, T)
            feats.append(branch(xm))        # (B, hidden, T)
        fused = torch.cat(feats, dim=1)      # (B, num_channels * hidden, T)
        fused = fused.transpose(1, 2)         # (B, T, num_channels * hidden)
        # return self.proj_norm(self.proj(fused))  # (B, T, d_model)
        return self.proj_norm(fused)  # (B, T, num_channels * hidden)


# ============================================================================ #
# Multi-Scale Retention (RetNet) — temporal feature extraction
# ============================================================================ #

class MultiScaleRetention(nn.Module):
    """Multi-Scale Retention module (parallel + recurrent dual form).

    Each of the `num_heads` heads uses a fixed decay rate gamma_h (multi-scale
    decay), so different heads naturally specialize in different temporal
    receptive fields — analogous to how MC-GTCN used different dilation
    factors across layers, but here it is content-aware (query/key driven)
    and learned jointly with the rest of the network.

    forward()            : parallel form, O(T^2) but fully parallelizable,
                            used during training on a fixed CAN window.
    forward_recurrent()   : O(1) recurrent form, used for streaming,
                            frame-by-frame inference on the vehicle.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.v_head_dim = 2 * self.head_dim  # widened value dim, as in RetNet

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.g_proj = nn.Linear(d_model, 2 * d_model, bias=False)  # swish gate
        self.out_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, 2 * d_model)
        self.dropout = nn.Dropout(dropout)

        # fixed multi-scale decay rates, gamma_h = 1 - 2^(-5-h)
        gamma = 1.0 - torch.pow(2.0, -5 - torch.arange(0, num_heads).float())
        self.register_buffer("gamma", gamma)  # (num_heads,)

    def _decay_mask(self, T: int, device, dtype) -> torch.Tensor:
        """Builds D in R^{h x T x T}: causal mask combined with exponential
        decay along the relative distance, per head."""
        n = torch.arange(T, device=device).unsqueeze(1).float()
        m = torch.arange(T, device=device).unsqueeze(0).float()
        diff = (n - m).clamp(min=0)                     # (T, T)
        causal = (n >= m).float()                        # (T, T)
        gamma = self.gamma.to(device=device).view(-1, 1, 1)  # (h, 1, 1)
        mask = (gamma ** diff.unsqueeze(0)) * causal.unsqueeze(0)  # (h, T, T)
        return mask.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, _ = x.shape
        h, hd, vhd = self.num_heads, self.head_dim, self.v_head_dim

        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)   # (B, h, T, hd)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)   # (B, h, T, hd)
        v = self.v_proj(x).view(B, T, h, vhd).transpose(1, 2)  # (B, h, T, vhd)

        decay = self._decay_mask(T, x.device, q.dtype)          # (h, T, T)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)  # (B, h, T, T)
        scores = scores * decay.unsqueeze(0)
        out = torch.matmul(scores, v)                            # (B, h, T, vhd)

        out = out.transpose(1, 2).contiguous().view(B, T, h * vhd)  # (B, T, 2*d_model)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)   # normalize per head-group
        gate = F.silu(self.g_proj(x))
        out = gate * out
        return self.dropout(self.out_proj(out))

    @torch.no_grad()
    def init_recurrent_state(self, batch_size: int, device) -> torch.Tensor:
        """S_0 = 0, shape (B, h, head_dim, v_head_dim)."""
        return torch.zeros(batch_size, self.num_heads, self.head_dim, self.v_head_dim, device=device)

    def forward_recurrent(
        self, x_t: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One O(1) recurrent update.

        x_t   : (B, d_model)              -- fused spatial feature of one CAN frame
        state : (B, h, head_dim, v_head_dim) -- previous state S_{n-1}
        returns (out_t, new_state)
        """
        B = x_t.size(0)
        h, hd, vhd = self.num_heads, self.head_dim, self.v_head_dim

        q = self.q_proj(x_t).view(B, h, hd)
        k = self.k_proj(x_t).view(B, h, hd)
        v = self.v_proj(x_t).view(B, h, vhd)

        gamma = self.gamma.to(x_t.device).view(1, h, 1, 1)
        new_state = gamma * state + torch.einsum("bhk,bhv->bhkv", k, v)  # S_n = gamma*S_{n-1} + K^T V
        out = torch.einsum("bhk,bhkv->bhv", q, new_state)                 # O_n = Q_n S_n

        out = out.reshape(B, h * vhd)
        out = self.group_norm(out.unsqueeze(-1)).squeeze(-1)  # same GroupNorm as parallel form
        gate = F.silu(self.g_proj(x_t))
        out = gate * out
        return self.out_proj(out), new_state


class RetentionBlock(nn.Module):
    """Pre-norm residual block: Retention + FFN, matching RetNet's overall
    layer layout (Eq. 9): Y = MSR(LN(X)) + X ;  Out = FFN(LN(Y)) + Y.
    """

    def __init__(self, cfg: CANIDSConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.msr = MultiScaleRetention(cfg.d_model, cfg.num_heads, cfg.retention_dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.d_model * cfg.ffn_hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Dropout(cfg.retention_dropout),
            nn.Linear(hidden, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.msr(self.ln1(x)) + x
        return self.ffn(self.ln2(y)) + y

    def forward_recurrent(
        self, x_t: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        msr_out, new_state = self.msr.forward_recurrent(self.ln1(x_t), state)
        y_t = msr_out + x_t
        out_t = self.ffn(self.ln2(y_t)) + y_t
        return out_t, new_state


# ============================================================================ #
# Channel Attention Mechanism (CAM) — feature fusion before classification
# ============================================================================ #

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
        return scaled.mean(dim=1)                   # pooled fused feature


# ============================================================================ #
# Top-level model
# ============================================================================ #

class CANIntrusionDetector(nn.Module):
    """STDLM x RetNet hybrid intrusion detection model for in-vehicle CAN buses.

    MC-1DCNN extracts per-field spatial features (adaptive kernels for CAN ID
    vs. payload bytes) at every timestep; the resulting sequence is fed to a
    stack of Retention blocks (RetNet's multi-scale retention), which replace
    STDLM's MC-GTCN for modeling long-range temporal dependencies. A Channel
    Attention Mechanism (CAM) fuses the final spatio-temporal representation
    before a small MLP classifier produces logits.

    Loss: for `num_classes == 2` (normal vs. attack) or `num_classes > 2`
    (specific attack types), use `nn.CrossEntropyLoss` on the returned logits
    against integer class labels.
    """

    def __init__(self, cfg: Optional[CANIDSConfig] = None):
        super().__init__()
        self.cfg = cfg or CANIDSConfig()
        self.spatial_encoder = MC1DCNN(self.cfg)
        self.temporal_blocks = nn.ModuleList(
            RetentionBlock(self.cfg) for _ in range(self.cfg.num_retention_layers)
        )
        self.channel_attention = ChannelAttention(self.cfg.d_model, self.cfg.cam_reduction)
        self.classifier = nn.Sequential(
            nn.Dropout(self.cfg.classifier_dropout),
            nn.Linear(self.cfg.d_model, self.cfg.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.classifier_dropout),
            nn.Linear(self.cfg.d_model // 2, self.cfg.num_classes),
        )

    # -- training / offline (parallel) path ------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, num_channels) normalized CAN window.
           channel 0    = CAN ID       (log-min-max normalized)
           channels 1-8 = payload bytes d1..d8 (log-min-max normalized)
        returns logits: (B, num_classes)
        """
        x = x[:, :, 0:9]
        z = self.spatial_encoder(x)              # (B, T, d_model)
        for block in self.temporal_blocks:
            z = block(z)
        pooled = self.channel_attention(z)         # (B, d_model)
        return F.normalize(pooled, dim=1), self.classifier(pooled)

    # -- real-time / streaming (recurrent) path ---------------------------- #

    @property
    def spatial_receptive_field(self) -> int:
        """Number of past raw CAN frames the spatial encoder needs buffered
        to compute one timestep's spatial feature (same-padding convs, so
        this only depends on kernel size and branch depth, not on how long
        the vehicle has been running)."""
        k = max(self.cfg.id_kernel_size, self.cfg.payload_kernel_size)
        num_convs = 1 + 2 * self.cfg.spatial_num_blocks  # input conv + 2 per residual block
        return num_convs * (k - 1) + 1

    @torch.no_grad()
    def init_stream_buffer(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        R = self.spatial_receptive_field
        return torch.zeros(batch_size, R, self.cfg.num_channels, device=device)

    @torch.no_grad()
    def init_stream_state(self, batch_size: int, device=None) -> List[torch.Tensor]:
        device = device or next(self.parameters()).device
        return [block.msr.init_recurrent_state(batch_size, device) for block in self.temporal_blocks]

    def stream_step(
        self,
        frame: torch.Tensor,
        buffer: torch.Tensor,
        states: List[torch.Tensor],
    ):
        """One incremental, O(1)-amortized update for real-time, frame-by-
        frame CAN monitoring on the ECU.

        frame  : (B, num_channels)             the newest normalized CAN frame.
        buffer : (B, R, num_channels)           rolling buffer of the last R raw
                 frames (R = `spatial_receptive_field`), used only to compute
                 THIS step's local spatial feature — constant-size regardless
                 of total elapsed time.
        states : list of per-layer recurrent retention states, as returned by
                 `init_stream_state`.

        returns (logits, new_buffer, new_states)

        Note: the Channel Attention Mechanism pools over the time dimension,
        so it is only meaningful over a window of features; in streaming mode
        we skip CAM and classify directly on the current retention output,
        trading a small amount of the CAM's noise-filtering benefit for O(1)
        latency and memory that do not grow with sequence length.
        """
        new_buffer = torch.cat([buffer[:, 1:], frame.unsqueeze(1)], dim=1)
        z_seq = self.spatial_encoder(new_buffer)   # (B, R, d_model)
        z_t = z_seq[:, -1, :]                        # spatial feature of newest frame

        new_states = []
        for block, state in zip(self.temporal_blocks, states):
            z_t, new_state = block.forward_recurrent(z_t, state)
            new_states.append(new_state)

        logits = self.classifier(z_t)
        return logits, new_buffer, new_states

    # -- utilities ----------------------------------------------------------- #

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # ============================================================================ #
# # Smoke test
# # ============================================================================ #

# if __name__ == "__main__":
#     torch.manual_seed(0)

#     cfg = CANIDSConfig(num_classes=2)
#     model = CANIntrusionDetector(cfg)
#     model.eval()

#     B, T, C = 8, cfg.window_size, cfg.num_channels
#     x = torch.rand(B, T, C)  # already-normalized CAN window in [0, 1]

#     # 1) parallel (training/offline) forward
#     logits = model(x)
#     print(f"[parallel]  input {tuple(x.shape)} -> logits {tuple(logits.shape)}")
#     print(f"model parameters: {model.num_parameters():,}")

#     # 2) streaming (recurrent) forward, fed frame-by-frame, should be usable
#     #    online with O(1) state regardless of how many frames have passed.
#     buf = model.init_stream_buffer(B)
#     states = model.init_stream_state(B)
#     stream_logits = None
#     for t in range(T):
#         stream_logits, buf, states = model.stream_step(x[:, t, :], buf, states)
#     print(f"[streaming] processed {T} frames -> logits {tuple(stream_logits.shape)}")
