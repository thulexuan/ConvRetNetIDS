import torch
import torch.nn as nn
import torch.nn.functional as F

from compare_networks.transformer_baseline import TransformerEncoderBlock
from networks.bidirectional_retention_with_swiGLU import BidirectionalRetentionEncoder
from networks.payload_local_encoder_per_frame import PayloadLocalEncoderPerFrame

class ResConvTransformer(nn.Module):

    def __init__(
        self,
        num_features: int = 9,
        d_model: int = 64,
        out_channels_payload: int = 64,
        window_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 64 * 2,
        num_classes: int = 10,
        dropout: float = 0.1,
        kernel_size: int = 1,
        n_res_blocks: int = 2,
    ):
        super().__init__()
       
        self.num_features = num_features
        self.d_model = d_model
        self.window_size = window_size  # số frame trong mỗi window (T=window_size) - cố định, không thay đổi

        # self.payload_local_encoder = PayloadLocalEncoderPerFrame(
        #     in_channels=9,  # can_id + 8 byte payload 
        #     out_channels=out_channels_payload,
        #     dropout=dropout,
        #     kernel_size=kernel_size,
        #     n_res_blocks=n_res_blocks,
        # )

        self.linear = nn.Linear(num_features, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, ffn_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        
        self.final_ln = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        x: (B, T, 9) = [can_id, data0..7]
        """

        id_payload_features = x[:, :, 0:9]  # cột can_id + data0..7 

        # final_encode_feat = self.payload_local_encoder(id_payload_features) # (B, T, d_model)
        final_encode_feat = self.linear(id_payload_features)  # (B, T, d_model)

        all_attn_weights = []
        for layer in self.layers:
            final_encode_feat, attn_weights = layer(final_encode_feat, attn_mask=mask)
            all_attn_weights.append(attn_weights)

        final_encode_feat = self.final_ln(final_encode_feat)

        mean_pooled = final_encode_feat.mean(dim=1)
        max_pooled = final_encode_feat.max(dim=1).values
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)

        logits = self.classifier(pooled)
        return logits    

    



