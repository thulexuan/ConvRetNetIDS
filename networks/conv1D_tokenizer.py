import torch
import torch.nn as nn

class SingleChannelBlock(nn.Module):

    """
    1 nhánh Conv1D cho 1 feature channel (CAN ID hoặc 1 payload byte).
 
    Lấy ý tưởng từ STDLM Section 3.2:
      - Conv1D với kernel size phù hợp từng field
      - Residual connection để tránh vanishing gradient
      - BatchNorm sau mỗi conv để stabilize training
 
    Input : (B, 1, T)   — 1 channel, T time steps
    Output: (B, out_ch, T)
    """
 
    def __init__(
        self,
        out_channels : int,
        kernel_size  : int,   # k=3 cho CAN ID, k=5 cho payload bytes
        conv_bias    : bool = False,
    ):
        super().__init__()
 
        # Conv layer 1
        self.conv1 = nn.Conv1d(
            in_channels  = 1,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            padding      = kernel_size // 2,   # same padding → giữ T
            bias         = conv_bias,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
 
        # Conv layer 2 (residual mapping)
        self.conv2 = nn.Conv1d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            padding      = kernel_size // 2,
            bias         = conv_bias,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
 
        # Residual projection: 1 → out_channels (vì channel count thay đổi)
        self.residual_proj = nn.Conv1d(1, out_channels, kernel_size=1, bias=False)
 
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        # x: (B, 1, T)
        residual = self.residual_proj(x)      # (B, out_channels, T)
 
        out = self.relu(self.bn1(self.conv1(x)))   # (B, out_channels, T)
        out = self.bn2(self.conv2(out))             # (B, out_channels, T)
 
        out = self.relu(out + residual)        # residual connection
        return out                             # (B, out_channels, T)
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  MC-1DCNN TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
 
class MC1DCNNTokenizer(nn.Module):
    """
    Multi-Channel 1D CNN Tokenizer — lấy ý tưởng từ STDLM paper.
 
    Input : (B, T, F)
              T = window_size  (số messages = số time steps)
              F = n_features   (1 CAN ID + 8 payload bytes = 9)
 
    Ý tưởng từ STDLM:
      - Xử lý CAN ID và payload bytes qua các nhánh Conv1D riêng biệt
      - CAN ID: kernel_size=3  (thay đổi nhanh, cần RF nhỏ)
      - Payload bytes: kernel_size=5  (pattern trải rộng hơn)
      - Residual connection trong mỗi nhánh
      - Concat tất cả nhánh → unified feature vector per time step
 
    Sau đó project lên embedding_dim để đưa vào RetNet.
 
    Output: (B, T, embedding_dim)   ← T tokens, mỗi token = 1 time step
    """
 
    def __init__(
        self,
        n_features    : int = 9,      # 1 CAN ID + 8 payload bytes
        out_ch_id     : int = 3,     # output channels cho CAN ID branch
        out_ch_payload: int = 5,     # output channels cho mỗi payload branch
        dropout_rate  : float = 0.1,
        conv_bias     : bool = False,
        embedding_dim : int = 128,
    ):
        super().__init__()
 
        self.n_features     = n_features
        self.n_payload      = n_features - 1   # = 8 payload bytes
        self.out_ch_id      = out_ch_id
        self.out_ch_payload = out_ch_payload
        self.embedding_dim  = embedding_dim
 
        # ── Branch 0: CAN ID (feature index 0) ───────────────────────────
        # kernel=3: CAN ID thay đổi nhanh, cần RF nhỏ để bắt sudden change
        self.branch_can_id = SingleChannelBlock(
            out_channels = out_ch_id,
            kernel_size  = 3,
            conv_bias    = conv_bias,
        )
 
        # ── Branches 1-8: Payload bytes (feature index 1..8) ─────────────
        # kernel=5: payload bytes có pattern trải rộng hơn qua nhiều time steps
        self.branches_payload = nn.ModuleList([
            SingleChannelBlock(
                out_channels = out_ch_payload,
                kernel_size  = 5,
                conv_bias    = conv_bias,
            )
            for _ in range(self.n_payload)
        ])
 
        # ── Tổng channels sau concat ──────────────────────────────────────
        self.total_ch = out_ch_id + out_ch_payload * self.n_payload
        # = 16 + 16×8 = 144  (với default params)
 
        # ── Projection: total_ch → embedding_dim ─────────────────────────
        # Tách rời conv channels và token dim (như đã phân tích)
        # self.proj = nn.Sequential(
        #     nn.Linear(self.total_ch, embedding_dim),
        #     nn.LayerNorm(embedding_dim),
        #     nn.Dropout(p=dropout_rate),
        # )
 
        self.apply(self._init_weights)
 
    def sequence_length(self, window_size: int, **kwargs) -> int:
        """Số tokens output = window_size (giữ nguyên T)."""
        return window_size
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)   — CAN windows từ DataLoader ~ (128, 29, 9)
        Returns: (B, T, embedding_dim) ~ (128, 29, 144)
        """
        B, T, F = x.shape
 
        # Transpose sang (B, F, T) để Conv1D xử lý
        # Conv1D expect: (B, C_in, L)
        x = x.transpose(1, 2)    # (B, F, T) ~ (batch_size, n_features, window_size)
 
        # ── Branch CAN ID: lấy feature 0 ─────────────────────────────────
        x_id = x[:, 0:1, :]                        # (B, 1, T)
        feat_id = self.branch_can_id(x_id)         # (B, out_ch_id, T)
 
        # ── Branches Payload: lấy features 1..8 ──────────────────────────
        feat_payload_list = []
        for i, branch in enumerate(self.branches_payload):
            x_byte = x[:, i+1 : i+2, :]           # (B, 1, T)
            feat_payload_list.append(branch(x_byte))  # (B, out_ch_payload, T)
 
        # ── Concat tất cả branches ────────────────────────────────────────
        # (B, total_ch, T)
        feat_all = torch.cat([feat_id] + feat_payload_list, dim=1)
 
        # ── Transpose về (B, T, total_ch) để dùng Linear ─────────────────
        feat_all = feat_all.transpose(1, 2)        # (B, T, total_ch)
 
        # ── Project → embedding_dim ───────────────────────────────────────
        # tokens = self.proj(feat_all)               # (B, T, embedding_dim)
 
        return feat_all
 
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)