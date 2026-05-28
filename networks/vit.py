import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.): # dim_head = d_model/num_heads
        # dim: the dimension of the input and output of the transformer, or the dimension of the token embeddings
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        # self.args=RetNetConfig(vocab_size=-1)
        # self.retnet_rel_pos = RetNetRelPos(self.args)
        # self.retention = self.build_retention(self.dim, self.args)

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                # MultiScaleRetention(embed_dim=dim, value_dim=dim * 2, num_heads=heads, args=self.args),
                # dim: dimension of token embeddings 
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x): # x: (batch_size, sequence_length, embedding_dim)
        # x ~ list of token embeddings ~ an input sequence
        # slen = x.size(1) # the number of tokens in one input sequence
        # retention_rel_pos = self.retnet_rel_pos(slen, False, False)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # config = RetNetConfig(vocab_size=-1)
        # self.transformer = RetNetDecoder(config)
        print(self.transformer)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None
        # self.mlp_head = None

    def forward(self, img):
        batch = img.shape[0]
        # print('Finding out batch: ', batch)
        # print('Finding out shape of original image: ', img.shape)
        x = self.to_patch_embedding(img)

        # print('Finding out shape of x after to_patch_embedding: ', x.shape)
        # print(x[0])

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        # print('Finding out class token: ', cls_tokens.shape)
        
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        # prev_output_tokens = torch.zeros(
        #     x.shape[0],
        #     x.shape[1],
        #     dtype=torch.long,
        #     device=x.device
        # )
        # print('Finding out shape of x after adding positional embedding: ', x.shape)
        x = self.dropout(x)
        # print('Finding out shape of x after dropping out: ', x.shape)

        # NOW, x is list of token embeddings
        x = self.transformer(x)
        # x, additional_info = self.transformer(prev_output_tokens=prev_output_tokens, token_embeddings=x)
        # print('Finding out shape of input after transformer: ', x.shape)
        # print('Finding out additional info: ', additional_info)

        # print('Finding out shape of x: ', x.shape)

        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # print('Finding out shape of x after mlp_head: ', self.mlp_head(x).shape)
        # print(self.mlp_head(x).shape) [64, 10]
        return self.mlp_head(x)
