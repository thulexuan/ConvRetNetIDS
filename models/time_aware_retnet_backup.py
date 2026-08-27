import torch
import torch.nn as nn
import torch.nn.functional as F

from models.payload_local_encoder_per_frame import PayloadLocalEncoderPerFrame
from models.original_retention_module import RetNetBlock, RetNetEncoder
from models.channel_attention import ChannelAttention

class TimeAwareRetNetCANIDS(nn.Module):

    # Vị trí cố định của các cột trong vector 11 chiều đầu vào.
    NUM_PROJ_FEATURES = 10   # can_id, data0..7, inter_arrival_ts -> vào input_proj
    INTER_ARRIVAL_COL = 9    # cột inter_arrival_ts (0-indexed, trong 11 cột)
    RELATIVE_TS_COL = 10     # cột relative_ts (0-indexed, trong 11 cột) - CHỈ dùng cho retention

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
        double_v_dim: bool = False,
        opt=None
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

        self.temporal_encoder = RetNetEncoder(
            hidden_size=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_size=ffn_dim,
            dropout=dropout,
            double_v_dim=double_v_dim,
        )
        
        self.final_ln = nn.LayerNorm(d_model)

        self.channel_attention = ChannelAttention(d_model=d_model, reduction=8)

        self.projection_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.proj = nn.Sequential(
            nn.Linear(
                self.window_size * d_model,
                d_model
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        self.class_net = nn.Sequential(
            nn.Linear(window_size * d_model, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(p=0.1),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(p=0.1),
            nn.Linear(d_model, num_classes)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        self.laynorm = nn.LayerNorm(d_model)

        # self.linear = nn.Linear(num_features, d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        x: (B, T, 9) = [can_id, data0..7]
        """
       
        id_payload_features = x[:, :, 0:9]  # cột can_id + data0..7 

        final_encode_feat = self.payload_local_encoder(id_payload_features) # (B, T, d_model)
        # retnet-only
        # final_encode_feat = self.linear(id_payload_features)  # (B, T, d_model)

        # local temporal - adding logic 180826
        # final_encode_feat = self.local_temporal_encoder(final_encode_feat) # (B, T, d_model)


        final_encode_feat = self.temporal_encoder(final_encode_feat)

        final_encode_feat = self.final_ln(final_encode_feat)

        final_encode_feat = self.channel_attention(final_encode_feat)

        mean_pooled_feat = final_encode_feat.mean(dim=1)  # (B, d_model)
        max_pooled_feat = final_encode_feat.max(dim=1).values  # (B, d_model)

        pooled_feat = torch.cat([mean_pooled_feat, max_pooled_feat], dim=1)  # (B, 2*d_model)

        logits = self.classifier(pooled_feat)  # (B, num_classes)

        # logits = self.class_net(final_encode_feat.view([final_encode_feat.shape[0], -1]))

        projected = F.normalize(self.projection_head(pooled_feat), dim=1)
        return final_encode_feat, logits, projected     

    



