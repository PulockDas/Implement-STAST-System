"""
Semantic lane encoder and fused trajectory model.

- SemanticLaneEncoder: LSTM over lane centerline points (x, y) -> 64-dim feature
- GraphSemanticTrajectoryModel: BiGRU + CGConv graph backbone fused with semantic feature
"""

from typing import Optional

import torch
import torch.nn as nn

from .temporal_encoder import BiGRUGraphTrajectoryEncoder
from .graph_layers import build_adjacency_matrix
from utils.config import FEATURE_DIM_POSITION, FUTURE_STEPS


class SemanticLaneEncoder(nn.Module):
    """
    Encode lane centerline coordinates into a fixed-size feature vector.

    Input:
        semantic_lane: (B, L, 2) — lane centerline points, normalized relative to target.

    Output:
        semantic_feature: (B, 64)
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM_POSITION,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, semantic_lane: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_lane: (B, L, 2)

        Returns:
            semantic_feature: (B, 64)
        """
        # We use the final hidden state as the lane embedding.
        _, (h_n, _) = self.lstm(semantic_lane)  # h_n: (num_layers, B, hidden_size)
        semantic_feature = h_n[-1]  # (B, hidden_size)
        return semantic_feature


class GraphSemanticTrajectoryModel(nn.Module):
    """
    BiGRU + CGConv backbone fused with semantic lane encoder.

    - Graph backbone: BiGRUGraphTrajectoryEncoder (without its decoder)
    - Semantic encoder: SemanticLaneEncoder
    - Fusion: concat[target_feature (128), semantic_feature (64)] -> 192
      then decoder: 192 -> 128 -> 60 -> (B, 30, 2)
    """

    def __init__(
        self,
        graph_backbone: Optional[BiGRUGraphTrajectoryEncoder] = None,
        semantic_hidden_size: int = 64,
    ) -> None:
        super().__init__()

        if graph_backbone is None:
            graph_backbone = BiGRUGraphTrajectoryEncoder()

        self.graph_backbone = graph_backbone
        self.semantic_encoder = SemanticLaneEncoder(
            input_size=FEATURE_DIM_POSITION,
            hidden_size=semantic_hidden_size,
        )

        self.target_dim = 128
        self.semantic_dim = semantic_hidden_size
        fused_dim = self.target_dim + self.semantic_dim  # 192

        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, FUTURE_STEPS * FEATURE_DIM_POSITION),  # 60
        )

        self.future_steps = FUTURE_STEPS

    def forward(
        self,
        past_traj: torch.Tensor,
        vehicle_mask: torch.Tensor,
        semantic_lane: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            past_traj: (B, N, 20, 2)
            vehicle_mask: (B, N)
            semantic_lane: (B, L, 2)

        Returns:
            pred_traj: (B, 30, 2)
        """
        # Graph backbone: BiGRU encoder + CGConv interaction
        vehicle_features = self.graph_backbone.encoder(past_traj)  # (B, N, 128)
        adj = build_adjacency_matrix(past_traj, vehicle_mask)
        interaction_features = self.graph_backbone.graph_layer(
            vehicle_features, adj
        )  # (B, N, 128)
        target_feature = interaction_features[:, 0, :]  # (B, 128)

        # Semantic lane encoding
        semantic_feature = self.semantic_encoder(semantic_lane)  # (B, 64)

        # Fusion and decoding
        fused = torch.cat([target_feature, semantic_feature], dim=-1)  # (B, 192)
        out = self.decoder(fused)  # (B, 60)
        return out.view(-1, self.future_steps, FEATURE_DIM_POSITION)  # (B, 30, 2)

