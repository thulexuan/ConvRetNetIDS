import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class UnifiedCANTokenizer(nn.Module):
    """
    Unified tokenizer cho 9 loại CAN attack.

    Input : (B, 3, 64, 64)
              height = 64 CAN messages (time axis)
              width  = 64 bits per message
              ch 0   = CAN ID  (11 bits pad -> 64)
              ch 1   = Payload (8 bytes x 8 bits = 64 bits)
              ch 2   = Timestamp (z-score x 2^32 -> 64 bits)

    Output: (B, N_tokens, embedding_dim)  — drop-in thay Tokenizer gốc

    Thiết kế 3 nhánh song song:
      A. Short-range  (kernel_h=3,  dilation_h=1)  -> RF = 3  msgs  : Fuzzing, Gear spoof
      B. Medium-range (kernel_h=3,  dilation_h=4)  -> RF = 9  msgs  : RPM/Speed spoof, Interval
      C. Long-range   (kernel_h=3,  dilation_h=12) -> RF = 25 msgs  : DoS, Standstill, Systematic

    Sau đó:
      - Channel-wise attention: tự chọn channel quan trọng (CAN ID / Payload / Timestamp)
      - Gated fusion: sigmoid gate trộn 3 nhánh
      - Bit-precision branch riêng: Conv1D theo width để bắt subtle value change (Speed spoof subtle)
    """

    def __init__(
        self,
        n_input_channels: int = 3,
        n_output_channels: int = 128,   # embedding_dim
        in_planes: int = 64,
        conv_bias: bool = False,
    ):
        super().__init__()
        d   = n_output_channels
        mid = in_planes             # intermediate channels mỗi nhánh

        # ── 0. Channel-wise attention (trước khi conv) ──────────────────────
        # Học trọng số cho từng channel (CAN ID / Payload / Timestamp)
        # tùy vào loại attack, model tự focus channel nào quan trọng.
        # SE-style: global avg pool -> FC -> sigmoid
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),               # (B, 3, 1, 1)
            nn.Flatten(),                               # (B, 3)
            nn.Linear(n_input_channels, n_input_channels),
            nn.Sigmoid(),
        )

        # ── 1. Branch A: Short-range (RF=3) ──────────────────────────────────
        # Bắt: Fuzzing (random 1-2 consecutive msgs), Gear/Speed spoof (sudden 1-frame change)
        self.branch_short = nn.Sequential(
            nn.Conv2d(n_input_channels, mid,
                      kernel_size=(3, 3),
                      dilation=(1, 1),
                      padding=(1, 1),
                      bias=conv_bias),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # ── 2. Branch B: Medium-range (RF=9 msgs) ────────────────────────────
        # Bắt: RPM spoof, Speed spoof sustained, Combined spoof, Interval
        # dilation_h=4: kernel tại row t nhìn t-4, t, t+4 -> span 9 msgs
        self.branch_med = nn.Sequential(
            nn.Conv2d(n_input_channels, mid,
                      kernel_size=(3, 3),
                      dilation=(4, 1),     # dilation chỉ theo time (height)
                      padding=(4, 1),
                      bias=conv_bias),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # ── 3. Branch C: Long-range (RF=25 msgs) ─────────────────────────────
        # Bắt: DoS (20-40 msgs liên tiếp), Standstill (RPM=0 sustained), Systematic
        # dilation_h=12: t-12, t, t+12 -> span 25 msgs
        self.branch_long = nn.Sequential(
            nn.Conv2d(n_input_channels, mid,
                      kernel_size=(3, 3),
                      dilation=(12, 1),    # dilation lớn theo time
                      padding=(12, 1),
                      bias=conv_bias),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # ── 4. Bit-precision branch (Conv theo width only) ───────────────────
        # Bắt: Speed spoof subtle (1-2 mph = vài bits thay đổi)
        # Chỉ nhìn theo width (bit dimension), kernel_h=1 để không mix messages
        self.branch_bit = nn.Sequential(
            nn.Conv2d(n_input_channels, mid,
                      kernel_size=(1, 7),  # kernel_h=1: không mix time rows
                      dilation=(1, 1),
                      padding=(0, 3),
                      bias=conv_bias),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # ── 5. Gated fusion ──────────────────────────────────────────────────
        # Input: concat 4 nhánh = mid*4 channels
        # Gate: học sigmoid weight cho từng nhánh tại mỗi spatial position
        # -> model tự chọn nhánh nào dominant cho mỗi vùng trong image
        fused = mid * 4
        self.gate = nn.Sequential(
            nn.Conv2d(fused, fused, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # ── 6. Projection -> embedding_dim ───────────────────────────────────
        self.proj = nn.Sequential(
            nn.Conv2d(fused, d, kernel_size=1, bias=conv_bias),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )

        # ── 7. Spatial pooling ───────────────────────────────────────────────
        # Pool (2,4): giảm height x2, width x4
        # -> output spatial size: (32, 16) -> 512 tokens
        # Nếu muốn ít token hơn, dùng (4,8) -> (16, 8) -> 128 tokens
        self.pool = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.apply(self._init_weights)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 64, 64)

        # 0. Channel attention: reweight từng channel (CAN ID / Payload / TS)
        ca = self.channel_attn(x)           # (B, 3)
        ca = ca.view(ca.shape[0], 3, 1, 1)  # (B, 3, 1, 1) broadcast
        x  = x * ca                         # channel-wise scale

        # 1-4. Bốn nhánh song song
        a = self.branch_short(x)   # (B, mid, 64, 64) — RF=3
        b = self.branch_med(x)     # (B, mid, 64, 64) — RF=9
        c = self.branch_long(x)    # (B, mid, 64, 64) — RF=25
        bp = self.branch_bit(x)    # (B, mid, 64, 64) — bit precision

        # 5. Gated fusion
        fused = torch.cat([a, b, c, bp], dim=1)  # (B, mid*4, 64, 64)
        gate  = self.gate(fused)                  # (B, mid*4, 64, 64)
        fused = fused * gate                       # element-wise gate

        # 6. Project -> embedding dim
        out = self.proj(fused)    # (B, d, 64, 64)

        # 7. Spatial pool
        out = self.pool(out)      # (B, d, 32, 16)

        # 8. Flatten spatial -> token sequence
        return rearrange(out, 'b c h w -> b (h w) c')  # (B, 512, d)

    def sequence_length(self, n_channels=3, height=64, width=64):
        """Tương thích với CCTRetNet.tokenizer.sequence_length(...)"""
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, height, width)
            return self.forward(dummy).shape[1]

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
