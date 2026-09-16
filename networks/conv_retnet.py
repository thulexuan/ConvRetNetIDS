import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.bidirectional_retention_with_swiGLU import BidirectionalRetentionEncoder
from networks.payload_local_encoder_per_frame import PayloadLocalEncoderPerFrame

class ConvRetNet(nn.Module):

    def __init__(
        self,
        num_features: int = 9,
        d_model: int = 128,
        out_channels_payload: int = 128,
        window_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
       
        self.num_features = num_features
        self.d_model = d_model
        self.window_size = window_size  # số frame trong mỗi window (T=window_size) - cố định, không thay đổi

        self.payload_local_encoder = PayloadLocalEncoderPerFrame(
            in_channels=9,  # can_id + 8 byte payload 
            out_channels=out_channels_payload,
            dropout=dropout
        )

        # self.temporal_encoder = RetNetEncoder(
        #     hidden_size=d_model,
        #     num_layers=num_layers,
        #     num_heads=num_heads,
        #     ffn_size=ffn_dim,
        #     dropout=dropout,
        #     double_v_dim=double_v_dim,
        # ) # uni-retention

        self.temporal_encoder = BidirectionalRetentionEncoder(
            dim=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_gated_decay=False,
            gated_decay_value_dim=16,
        ) # có swiGLU

        # self.temporal_encoder = BidirectionalRetentionEncoder(
        #     dim=d_model, num_heads=num_heads, num_layers=num_layers,
        #     dropout=dropout, causal=True, # causal=False~bidirectional, causal=True~unidirectional
        # )

        # self.temporal_encoder = BidirectionalRetentionEncoder(
        #     dim=d_model, num_heads=num_heads, num_layers=num_layers,
        #     dropout=dropout, causal=True,
        # )

        # self.temporal_encoder = BidirectionalRetentionEncoder(
        #     dim=d_model, num_heads=num_heads, num_layers=num_layers, dropout=0.1,
        #     use_id_aware_decay=True, id_decay_delta=3.0,
        # ) # có id-aware decay

        # self.temporal_encoder = BidirectionalRetentionEncoder(
        #     dim=d_model, num_heads=num_heads, num_layers=num_layers,
        #     dropout=0.1, learnable_gamma=True, gamma_parameterization="sigmoid",
        # ) # learnable gamma

        # self.temporal_encoder = BidirectionalRetentionEncoder(
        #     dim=d_model, num_heads=num_heads, num_layers=num_layers, dropout=0.1,
        #     learnable_gamma=True, use_id_aware_decay=True, same_id_init_extra=0.9,
        # ) # learnable gamma + id-aware decay
        
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

        # self.classifier = AttentionMILClassifier(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        x: (B, T, 9) = [can_id, data0..7]
        """

        id_features = x[:, :, 0]  # cột can_id
        id_payload_features = x[:, :, 0:9]  # cột can_id + data0..7 

        final_encode_feat = self.payload_local_encoder(id_payload_features) # (B, T, d_model)

        final_encode_feat = self.temporal_encoder(final_encode_feat)  # (B, T, d_model)

        # id-aware
        # final_encode_feat = self.temporal_encoder(final_encode_feat, can_ids=id_features)  # (B, T, d_model)
        final_encode_feat = self.final_ln(final_encode_feat)

        mean_pooled_feat = final_encode_feat.mean(dim=1)  # (B, d_model)
        max_pooled_feat = final_encode_feat.max(dim=1).values  # (B, d_model)

        pooled_feat = torch.cat([mean_pooled_feat, max_pooled_feat], dim=1)  # (B, 2*d_model)

        logits = self.classifier(pooled_feat)  # (B, num_classes)

        ##### attention classifier
        # dict = self.classifier(final_encode_feat)  # (B, num_classes)
        # logits = dict['logits'] 
        # attn_weights = dict['attn_weights']  # (B, T, 1)
        
        return final_encode_feat, logits, None, None     

    



