from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from networks.conv1D_tokenizer import MC1DCNNTokenizer
from networks.torchscale.architecture.retnet import RetNetDecoder
from networks.torchscale.config.config import RetNetConfig

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# modules

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from networks.torchscale.architecture.retnet import RetNetDecoder
from networks.torchscale.config.config import RetNetConfig


def exists(val):
    return val is not None


                   # (batch, seq_len, 144) - shape KHÔNG đổi


import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from networks.torchscale.architecture.retnet import RetNetDecoder
from networks.torchscale.config.config import RetNetConfig


def exists(val):
    return val is not None


class CrossChannelAttention(nn.Module):
    """
    Self-Attention học quan hệ GIỮA các field (can_id, data0, ..., data7).

    Coi mỗi field là 1 "token" (dim_per_channel chiều), num_channels field
    tạo thành 1 "sequence" độ dài num_channels. Áp dụng self-attention
    TẠI MỖI TIMESTEP độc lập -> giữ nguyên độ phân giải thời gian, không
    pooling trước khi học cross-channel dependency.

    Input:  x ∈ (batch, seq_len, num_channels * dim_per_channel)
    Output: x' ∈ (batch, seq_len, num_channels * dim_per_channel) - cùng shape
    """

    def __init__(self, num_channels, dim_per_channel, n_heads=4,
                 ffn_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.dim_per_channel = dim_per_channel
        embed_dim = dim_per_channel

        assert embed_dim % n_heads == 0, \
            f"dim_per_channel ({embed_dim}) phải chia hết cho n_heads ({n_heads})"

        # ---- Standard Multi-Head Self-Attention (giữa 9 field) ----
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # ---- Feed-Forward Network (giống Transformer Encoder chuẩn) ----
        ffn_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, num_channels * dim_per_channel)
        b, t, d = x.shape

        # ---- Reshape: coi (batch*seq_len) là batch mới,
        #      num_channels là "sequence" cho self-attention ----
        x = x.view(b, t, self.num_channels, self.dim_per_channel)
        x = x.reshape(b * t, self.num_channels, self.dim_per_channel)
        # -> (b*t, num_channels=9, dim_per_channel=16)

        # ---- Self-Attention GIỮA 9 field (Q,K,V đều từ chính x) ----
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)          # residual + norm

        # ---- Feed-Forward ----
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)           # residual + norm

        # ---- Reshape lại về shape gốc ----
        x = x.reshape(b, t, self.num_channels, self.dim_per_channel)
        x = x.reshape(b, t, self.num_channels * self.dim_per_channel)

        return x, attn_weights  # attn_weights: (b*t, 9, 9) - dùng để diễn giải


class TransformerClassifier(nn.Module):
    """
    Pipeline:
    (batch, t, 9*16) -> RetNet (share weight, học temporal RIÊNG từng field)
                     -> CrossChannelAttention (self-attention GIỮA 9 field)
                     -> seq_pool -> Classifier
    """

    def __init__(self,
                 seq_pool=True,
                 embedding_dim=144,
                 num_channels=9,
                 n_layers=2,
                 n_heads=2,
                 cross_attn_heads=4,
                 num_classes=64,
                 dropout_rate=0.1,
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()

        assert embedding_dim % num_channels == 0

        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.dim_per_channel = embedding_dim // num_channels
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        self.dropout = nn.Dropout(p=dropout_rate)

        # ---- Bước 1: RetNet share weight, học temporal RIÊNG từng field ----
        config = RetNetConfig(
            vocab_size=-1,
            decoder_embed_dim=self.dim_per_channel,
            decoder_value_embed_dim=self.dim_per_channel * 2,
            decoder_retention_heads=n_heads,
            decoder_ffn_embed_dim=self.dim_per_channel * 2,
            decoder_layers=n_layers,
        )
        self.retnet_shared = RetNetDecoder(args=config)

        # ---- Bước 2: Cross-Channel Self-Attention (GIỮA các field) ----
        self.cross_channel_attn = CrossChannelAttention(
            num_channels=num_channels,
            dim_per_channel=self.dim_per_channel,
            n_heads=cross_attn_heads,
            dropout=dropout_rate,
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.apply(self.init_weight)

    def _channels_to_batch(self, x):
        b, t, d = x.shape
        x = x.view(b, t, self.num_channels, self.dim_per_channel)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b * self.num_channels, t, self.dim_per_channel)
        return x, b

    def _batch_to_channels(self, x, b):
        bn, t, d = x.shape
        x = x.view(b, self.num_channels, t, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, t, self.num_channels * d)
        return x

    def get_feature_before_fc(self, x, return_attn=False):
        b = x.shape[0]

        if x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.sequence_length - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_token, x), dim=1)

        # ---- Bước 1: RetNet - temporal dependency RIÊNG từng field ----
        x_batched, b = self._channels_to_batch(x)         # (b*9, t, 16)
        x_batched, _ = self.retnet_shared(
            prev_output_tokens=x_batched, token_embeddings=x_batched
        )                                                   # (b*9, t, 16) - vẫn TÁCH BIỆT
        x = self._batch_to_channels(x_batched, b)          # (b, t, 144)

        # ---- Bước 2: Cross-Channel Attention - quan hệ GIỮA các field ----
        x, attn_weights = self.cross_channel_attn(x)       # (b, t, 144)

        # ---- Bước 3: Pooling cuối cùng ----
        if self.seq_pool:
            attn_pool_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_pool_weights.softmax(dim=1), x)
        else:
            x = x[:, -1]

        if return_attn:
            return x, attn_weights
        return x

    def forward(self, x):
        return self.get_feature_before_fc(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# 1D-CNN-RetNet Main model

class CNN1DModifiedRetNet(nn.Module):
    def __init__(
        self,
        window_size=32,
        dropout_rate=0.,
        attention_dropout=0.1,
        stochastic_depth_rate=0.1,
        projection_head='mlp',
        projection_dim=64,
        n_layers=2,
        n_heads=2,
        num_classes=10,
        positional_embedding='none',
        num_branches=9,
        *args, **kwargs
    ):
        super().__init__()
        self.window_size = window_size

        self.tokenizer = MC1DCNNTokenizer(
            n_features=9,
            out_ch_id=16,
            out_ch_payload=16,
            dropout_rate=0.1,
            conv_bias=False,
        )

        embedding_dim = self.tokenizer.total_ch  # 144

        self.encoder = TransformerClassifier(
            sequence_length=window_size,
            embedding_dim=embedding_dim,
            num_branches=num_branches,
            seq_pool=True,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            stochastic_depth_rate=stochastic_depth_rate,
            n_layers=n_layers,
            n_heads=n_heads,
            num_classes=projection_dim,
            positional_embedding=positional_embedding,
            *args, **kwargs)

        self.fc = nn.Linear(embedding_dim, num_classes)

        if projection_head == 'linear':
            self.head = nn.Linear(embedding_dim, projection_dim)
        elif projection_head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, projection_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(projection_head))

    def forward(self, x):  # (128, 16, 9)
        x = self.tokenizer(x)                  # (128, 16, 144)
        x_before_fc = self.encoder(x)          # (128, 144)
        x_after_fc = self.head(x_before_fc)    # (128, 64)

        return x_before_fc, F.normalize(x_after_fc, dim=1)