import torch
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


# positional

def sinusoidal_embedding(n_channels, dim): # 1D positional embedding, cho biết token index chứ k cho biết token nằm ở hàng nào, cột nào
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

def sinusoidal_embedding_2d(height, width, dim):
    if dim <= 1:
        return sinusoidal_embedding(height * width, dim)

    row_dim = dim // 2
    col_dim = dim - row_dim

    row_pe = sinusoidal_embedding(height, row_dim).squeeze(0)
    col_pe = sinusoidal_embedding(width, col_dim).squeeze(0)

    row_pe = repeat(row_pe, 'h d -> h w d', w=width)
    col_pe = repeat(col_pe, 'w d -> h w d', h=height)

    pe = torch.cat([row_pe, col_pe], dim=-1)
    return rearrange(pe, 'h w d -> 1 (h w) d')

# modules

class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=128, # 256
                 n_layers=2, # 2
                 n_heads=2, # 2
                 mlp_ratio=4.0, # 1
                 num_classes=64,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 sequence_length=None,
                 sequence_shape=None,
                 mlp=True,
                 *args, **kwargs):
        super().__init__()


        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.sequence_shape = sequence_shape
        self.seq_pool = seq_pool

        if not seq_pool:  # if not using sequence pooling, add a class token and increase the sequence length by 1
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        self.dropout = nn.Dropout(p=dropout_rate)

        config = RetNetConfig(
            vocab_size=-1,
            decoder_embed_dim=embedding_dim, 
            decoder_value_embed_dim=embedding_dim * 2,
            decoder_retention_heads=n_heads,
            decoder_ffn_embed_dim=embedding_dim * 2,
            decoder_layers=n_layers
        )
        
        self.retnet_blocks = RetNetDecoder(args=config)

        self.norm = nn.LayerNorm(embedding_dim)
        
        self.apply(self.init_weight)

    def get_feature_before_fc(self, x): # when call Transformer Classifier, put x is a sequence of tokens
        # print("Investigating the input x of Transformer Classifier: ", x.shape) # expectedshape is (batch_size, sequence_length, embedding_dim)
        b = x.shape[0]

        if x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.sequence_length - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_token, x), dim=1)
      
        # with RetNet
        x, _ = self.retnet_blocks(prev_output_tokens=x, token_embeddings=x) # (batch_size, sequence_length, embedding_dim)
                
        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim = 1), x)
        else:
            x = x[:, -1] # get the last token, shape is (batch_size, embedding_dim)

        # print("Print shape of feature before fc: ", x.shape)

        return x # shape is (128, 128) ~ (batch_size, embedding_dim)

    def forward(self, x): # (128, 16, 128) ~ (batch_size, sequence_length, token_embedding_dim)

        x_before_fc = self.get_feature_before_fc(x) # (128, 144)
        return x_before_fc

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

class CNN1DRetNet(nn.Module):
    def __init__(
        self,
        window_size=29,
        dropout_rate=0.,
        attention_dropout=0.1,
        stochastic_depth_rate=0.1,
        projection_head='mlp',
        projection_dim=64,
        n_layers=2,
        n_heads=2,
        num_classes=10,
        positional_embedding='none',
        *args, **kwargs
    ):
        super().__init__()
        self.window_size=window_size,

        ####------ Conv1D tokenizer
        self.tokenizer = MC1DCNNTokenizer(
            n_features = 9,      # 1 CAN ID + 8 payload bytes
            out_ch_id  = 16,     # output channels cho CAN ID branch
            out_ch_payload = 16,     # output channels cho mỗi payload branch
            dropout_rate  = 0.1,
            conv_bias     = False,
        )

        embedding_dim = self.tokenizer.total_ch # 144
        
        self.encoder = TransformerClassifier(
            sequence_length=window_size, # 16
            embedding_dim=embedding_dim, # 144
            seq_pool=True,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            stochastic_depth_rate=stochastic_depth_rate,
            n_layers=n_layers,
            n_heads=n_heads,
            num_classes=projection_dim,
            positional_embedding=positional_embedding,
            *args, **kwargs)

        
        self.fc = nn.Linear(embedding_dim, num_classes) # for supervised loss, shape is (128, 10)
        
        # add projection head, projection_dim: feature dimension used for contrastive learning
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
            raise NotImplementedError(
                'head not supported: {}'.format(projection_head))
    
    def forward(self, x): # (128, 16, 9)
        
        x = self.tokenizer(x) # x shape is (128, 16, 144) ~ (batch_size, sequence_length, token_embedding_dim)
        x_before_fc = self.encoder(x) # x_before_fc (128, 144)
        x_after_fc = self.head(x_before_fc) # x_after_fc shape is (128, 64)
        # print("check")
        # print(x_before_fc.shape, x_after_fc.shape)

        return x_before_fc, F.normalize(x_after_fc, dim=1) # x_after_fc shape is (128, 64), L2-normalized
    