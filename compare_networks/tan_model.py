"""
TAN: Transformer-Based Attention Network for In-Vehicle Intrusion Detection.

Reference:
Nguyen, T.P., Nam, H., Kim, D. (2023). "Transformer-Based Attention Network
for In-Vehicle Intrusion Detection." IEEE Access, 11, 55389-55403.

Architecture (Fig. 8 + Algorithm 1 of the paper):
    Input (raw scalar sequence, length L)
        -> Input Embedding + Positional Encoding  (Algorithm 1, line 2-3:
           the SAME embedding function f_embed is applied both to the raw
           values and to the position indices [0..L-1], then summed --
           this is NOT the standard sinusoidal positional encoding.)
        -> Multi-Head Attention (Eq. 2-3, 5-8)
        -> Add & Normalization  (residual + LayerNorm)
        -> Feed Forward
        -> Add & Normalization
        -> Max Pooling (over the sequence dimension)
        -> Dense(32)
        -> Classifier (Linear -> logits; apply Softmax externally or via
           CrossEntropyLoss)

Two input types share this SAME architecture (only preprocessing differs,
done outside this file, per Sec. IV-B/IV-C of the paper):

    1) Single message   : L = 10 tokens = [CAN_ID, DLC, byte1, ..., byte8]
                           (missing payload bytes padded with 256).
                           Detects DoS / Fuzzy / Spoofing well, but NOT
                           replay attacks (no temporal context).
    2) Sequential CAN ID: L = window_size (e.g. 16/32/64/128/256) tokens,
                           each token = a single CAN ID value at time t
                           (Eq. 1: X_i = [x_i^1, ..., x_i^w]).
                           Segment-level label (Algorithm 2) is enough --
                           no per-message labels needed. Enables replay-
                           attack detection via loss of periodicity.

INPUT  : tensor of shape (batch, L) -- raw scalar values (CAN ID / DLC /
         payload bytes, decimal-converted, NOT yet embedded)
OUTPUT : tensor of shape (batch, num_classes) -- raw logits
         (use nn.CrossEntropyLoss directly, or apply Softmax externally)

Additionally, `identify_malicious_tokens()` implements Sec. IV-C: even
though the model is trained with segment-level labels only, the
per-token attention scores can be summed, passed through Sigmoid, and
thresholded to flag WHICH individual token(s) in the segment are
malicious -- without ever using message-level labels during training.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TANConfig:
    seq_len: int = 32       # L: 10 for single-message, or window_size for sequential CAN ID
    embed_dim: int = 128     # Table 5: Embedding dimension = 32
    num_heads: int = 4      # Table 5: Number of heads = 4
    ff_dim: int = 256        # Table 5: Feed forwarding dimension = 32
    dense_dim: int = 128     # Fig. 8: Dense(32) before the classifier
    num_classes: int = 10    # e.g. Normal + DoS + Fuzzy + Gear + RPM (Car Hacking dataset)
    dropout: float = 0.15   # paper's dropout rate, interleaved with normalization
    num_layers: int = 2     # Fig. 8 depicts a single encoder block; stackable if desired


# --------------------------------------------------------------------------- #
# 1) Input Embedding + Positional Encoding (Algorithm 1, lines 2-3)
# --------------------------------------------------------------------------- #
class TokenPositionEmbedding(nn.Module):
    """
    Embeds a sequence of raw scalar values (CAN ID / DLC / payload byte /
    or a single CAN ID at time t) into `embed_dim`-dimensional vectors, and
    adds a positional signal.

    Per Algorithm 1 (lines 2-3):
        input = f_embed([x_0, ..., x_w]) + f_embed([0, ..., w])
    i.e. the SAME linear embedding function is reused for both the token
    values and the position indices, then summed -- rather than the
    standard fixed sinusoidal positional encoding used in the original
    Transformer (Vaswani et al., 2017).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # continuous scalar -> embed_dim vector (values are decimal CAN
        # fields, not a fixed vocabulary, so a linear projection is used
        # in place of a discrete nn.Embedding lookup table)
        self.f_embed = nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) raw scalar values
        B, L = x.shape
        value_emb = self.f_embed(x.unsqueeze(-1))  # (B, L, embed_dim)

        positions = torch.arange(L, device=x.device, dtype=x.dtype)  # (L,)
        pos_emb = self.f_embed(positions.unsqueeze(-1))              # (L, embed_dim)

        return value_emb + pos_emb.unsqueeze(0)  # broadcast -> (B, L, embed_dim)


# --------------------------------------------------------------------------- #
# 2) Transformer Encoder Block (Fig. 8: MHA -> Add&Norm -> FF -> Add&Norm)
# --------------------------------------------------------------------------- #
class TANEncoderLayer(nn.Module):
    """
    One encoder block of the TAN architecture (Fig. 8):
        x -> MultiHeadAttention -> Dropout -> Add(x) -> LayerNorm
          -> FeedForward        -> Dropout -> Add(.) -> LayerNorm
    Also returns the attention weights (averaged over heads by PyTorch's
    nn.MultiheadAttention), needed later for Sec. IV-C malicious-token
    identification.
    """

    def __init__(self, cfg: TANConfig):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.dropout1 = nn.Dropout(cfg.dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.embed_dim),
        )
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        # --- Multi-Head Attention + Add & Norm (Eq. 2-3, 5-8) ---
        attn_out, attn_weights = self.mha(x, x, x, need_weights=True)
        x = self.norm1(x + self.dropout1(attn_out))

        # --- Feed Forward + Add & Norm ---
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x, attn_weights  # attn_weights: (B, L, L), averaged over heads


# --------------------------------------------------------------------------- #
# 3) Full TAN model
# --------------------------------------------------------------------------- #
class TAN(nn.Module):
    """
    Full pipeline (Fig. 8):
        (B, L) --Embedding+PosEnc--> (B, L, embed_dim)
               --N x TANEncoderLayer--> (B, L, embed_dim)
               --Max Pooling (over L)--> (B, embed_dim)
               --Dense(32) + ReLU + Dropout--> (B, dense_dim)
               --Classifier--> (B, num_classes)  [logits]

    Works identically for both paper input types -- only `cfg.seq_len`
    (and the data fed in) differs:
        - single message   : cfg.seq_len = 10
        - sequential CAN ID: cfg.seq_len = window_size (16/32/64/128/256)
    """

    def __init__(self, cfg: TANConfig | None = None):
        super().__init__()
        self.cfg = cfg or TANConfig()

        self.embedding = TokenPositionEmbedding(self.cfg.embed_dim)
        self.layers = nn.ModuleList(
            [TANEncoderLayer(self.cfg) for _ in range(self.cfg.num_layers)]
        )

        self.dense = nn.Linear(self.cfg.embed_dim, self.cfg.dense_dim)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.classifier = nn.Linear(self.cfg.dense_dim, self.cfg.num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (batch, L) raw scalar sequence (decimal CAN fields)
            return_attention: if True, also return the list of per-layer
                attention weight tensors (each (B, L, L)), used for
                Sec. IV-C intrusion-message identification.
        Returns:
            logits: (batch, num_classes)
            [attn_weights_per_layer]: optional, list of (B, L, L) tensors
        """
        x = x[:, :, 0]  # only the first column (CAN ID) is used for TAN
        h = self.embedding(x)  # (B, L, embed_dim)

        attn_weights_per_layer = []
        for layer in self.layers:
            h, attn_w = layer(h)
            attn_weights_per_layer.append(attn_w)

        pooled = torch.max(h, dim=1).values  # Max Pooling over sequence -> (B, embed_dim)

        dense_out = F.relu(self.dense(pooled))
        dense_out = self.dropout(dense_out)
        logits = self.classifier(dense_out)  # (B, num_classes)

        if return_attention:
            return logits, attn_weights_per_layer
        return logits

    @torch.no_grad()
    def identify_malicious_tokens(self, x: torch.Tensor, threshold: float = 0.5):
        """
        Sec. IV-C of the paper: identify WHICH tokens (CAN IDs) in a
        segment are malicious, using only the attention scores learned
        from segment-level labels (no message-level labels needed).

        Procedure (as described in the paper):
            1. Take the last encoder layer's attention weights (B, L, L).
            2. Sum over the query dimension -> per-key (per-token) score
               ("attention scores of each segment were flattened by the
               sum function", Sec. IV-C).
            3. Pass through Sigmoid.
            4. Threshold (paper sweeps 0.3-0.6; default 0.5) -> binary
               flag per token: 1 = malicious, 0 = normal.

        Args:
            x: (batch, L) raw scalar sequence (typically a sequential
               CAN ID segment, L = window_size)
            threshold: sigmoid threshold for flagging a token malicious
        Returns:
            logits: (batch, num_classes)          -- segment-level prediction
            token_scores: (batch, L) in [0, 1]     -- per-token sigmoid score
            token_flags: (batch, L) in {0, 1}      -- per-token malicious flag
        """
        logits, attn_weights_per_layer = self.forward(x, return_attention=True)

        last_attn = attn_weights_per_layer[-1]      # (B, L, L): query x key
        token_raw_score = last_attn.sum(dim=1)      # sum over queries -> (B, L)
        token_scores = torch.sigmoid(token_raw_score)
        token_flags = (token_scores > threshold).long()

        return logits, token_scores, token_flags


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    # ----------------------------------------------------------------- #
    # 1) Single-message mode: L = 10 tokens [CAN_ID, DLC, byte1..byte8]
    # ----------------------------------------------------------------- #
    cfg_single = TANConfig(seq_len=10, num_classes=5)  # e.g. 5 classes on Car Hacking
    model_single = TAN(cfg_single)

    batch_size = 8
    # emulate raw decimal CAN fields (unnormalized, as in Fig. 9)
    dummy_single = torch.rand(batch_size, cfg_single.seq_len) * 256
    logits_single = model_single(dummy_single)
    print("== Single-message mode ==")
    print(f"Input shape : {tuple(dummy_single.shape)}")
    print(f"Output shape: {tuple(logits_single.shape)}")

    # ----------------------------------------------------------------- #
    # 2) Sequential CAN ID mode: L = window_size (e.g. 16)
    # ----------------------------------------------------------------- #
    cfg_seq = TANConfig(seq_len=32, num_classes=10)
    model_seq = TAN(cfg_seq)

    dummy_seq = torch.randint(0, 2048, (batch_size, cfg_seq.seq_len)).float()
    logits_seq, token_scores, token_flags = model_seq.identify_malicious_tokens(
        dummy_seq, threshold=0.5
    )
    print("\n== Sequential CAN ID mode ==")
    print(f"Input shape         : {tuple(dummy_seq.shape)}")
    print(f"Segment logits shape: {tuple(logits_seq.shape)}")
    print(f"Token scores shape  : {tuple(token_scores.shape)}")
    print(f"Token flags (sample): {token_flags[0].tolist()}")

    n_params = sum(p.numel() for p in model_seq.parameters())
    print(f"\nTotal trainable parameters (seq model): {n_params:,}")

    # Example loss + training step (matches Algorithm 1: optimize L_cross_entropy)
    # NOTE: use plain forward() for training -- identify_malicious_tokens() is
    # wrapped in @torch.no_grad() since it's meant for inference/interpretability
    # (Sec. IV-C), not for the training loop.
    logits_train = model_seq(dummy_seq)
    labels = torch.randint(0, cfg_seq.num_classes, (batch_size,))
    loss = F.cross_entropy(logits_train, labels)
    optimizer = torch.optim.Adam(model_seq.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model_seq.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"Sample cross-entropy loss: {loss.item():.4f}")
    print(f"Gradient norm (post-clip target 1.0): {grad_norm:.4f}")
