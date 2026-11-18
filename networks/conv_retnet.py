import torch
import torch.nn as nn
import torch.nn.functional as F

from compare_networks.transformer_baseline import TransformerEncoderBlock
from networks.bidirectional_retention_with_swiGLU import BidirectionalRetentionEncoder
from networks.local_module import PayloadFeatureConvEncoder
from networks.payload_local_encoder_per_frame import PayloadLocalEncoderPerFrame
from networks.temporal_original_retnet import OriginalRetNetEncoder
# from networks.softmax_retention_encoder import BidirectionalRetentionEncoder

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
        kernel_size: int = 1,
        n_res_blocks: int = 2,
        temporal_type: str = "proposed",
    ):
        super().__init__()
       
        self.num_features = num_features
        self.d_model = d_model
        self.window_size = window_size  # số frame trong mỗi window (T=window_size) - cố định, không thay đổi

        # self.linear = nn.Linear(num_features, d_model) # ablation study of without payload local encoder

        self.payload_local_encoder = PayloadLocalEncoderPerFrame(
            in_channels=9,  # can_id + 8 byte payload 
            out_channels=out_channels_payload,
            dropout=dropout,
            kernel_size=kernel_size,
            n_res_blocks=n_res_blocks
        )

        # self.payload_local_encoder = PayloadFeatureConvEncoder(
        #     in_channels=9,  # can_id + 8 byte payload 
        #     out_channels=out_channels_payload,
        #     dropout=dropout
        # )

        # self.temporal_encoder = BidirectionalRetentionEncoder(
        #     dim=d_model,
        #     num_heads=num_heads,
        #     num_layers=num_layers,
        #     dropout=dropout,
        #     # causal=True,
        #     # ffn_type="swiglu"
        #     ffn_type="gelu"
        # ) # có swiGLU

        if temporal_type == "proposed":
            self.temporal_encoder = BidirectionalRetentionEncoder(
                dim=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                # causal=causal,
                ffn_type="swiglu",
            )
        elif temporal_type == "original_retnet":
            # original RetNet: causal decay, no softmax, GELU FFN (hidden 2d)
            self.temporal_encoder = OriginalRetNetEncoder(
                dim=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unknown temporal_type: {temporal_type}")
        
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

        final_encode_feat = self.payload_local_encoder(id_payload_features) # (B, T, d_model)


        # simple Linear first, ablation study of without payload local encoder
        # final_encode_feat = self.linear(id_payload_features)  # (B, T, d_model)

        final_encode_feat = self.temporal_encoder(final_encode_feat)  # (B, T, d_model)

        final_encode_feat = self.final_ln(final_encode_feat)

        mean_pooled_feat = final_encode_feat.mean(dim=1)  # (B, d_model)
        max_pooled_feat = final_encode_feat.max(dim=1).values  # (B, d_model)

        pooled_feat = torch.cat([mean_pooled_feat, max_pooled_feat], dim=1)  # (B, 2*d_model)

        logits = self.classifier(pooled_feat)  # (B, num_classes)
        
        return final_encode_feat, logits, None, None     

    



