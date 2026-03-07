"""
BiGRU temporal encoder for trajectory prediction.

Encodes each vehicle's past trajectory into a 128-dim feature vector.
Temporary decoder predicts target vehicle future (30, 2) without graph interaction.
"""

import torch
import torch.nn as nn

try:
    from utils.config import PAST_STEPS, FUTURE_STEPS, FEATURE_DIM_POSITION
except ImportError:
    PAST_STEPS = 20
    FUTURE_STEPS = 30
    FEATURE_DIM_POSITION = 2


class BiGRUTrajectoryEncoder(nn.Module):
    """
    Bidirectional GRU encoder for past trajectories plus temporary MLP decoder
    (target vehicle only; no graph / CGConv / attention yet).
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM_POSITION,
        hidden_size: int = 64,
        num_layers: int = 1,
        past_steps: int = PAST_STEPS,
        future_steps: int = FUTURE_STEPS,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.embed_dim = hidden_size * 2  # bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Temporary decoder: target embedding -> future trajectory (no graph)
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, future_steps * FEATURE_DIM_POSITION),  # 60
        )

    def encoder(self, past_traj: torch.Tensor) -> torch.Tensor:
        """
        Encode all vehicles' past trajectories into feature vectors.

        Args:
            past_traj: (B, N, past_steps, 2)

        Returns:
            vehicle_features: (B, N, 128)
        """
        B, N, T, F = past_traj.shape
        x = past_traj.reshape(B * N, T, F)  # (B*N, 20, 2)

        output, h_n = self.gru(x)  # h_n: (num_layers*2, B*N, hidden_size)
        # Take last layer: forward and backward
        h_forward = h_n[-2]   # (B*N, hidden_size)
        h_backward = h_n[-1]  # (B*N, hidden_size)
        features = torch.cat([h_forward, h_backward], dim=-1)  # (B*N, 128)

        return features.reshape(B, N, self.embed_dim)

    def forward(self, past_traj: torch.Tensor) -> torch.Tensor:
        """
        Encode scene and predict target vehicle future trajectory only.

        Args:
            past_traj: (B, N, past_steps, 2)

        Returns:
            pred_traj: (B, future_steps, 2)
        """
        vehicle_features = self.encoder(past_traj)  # (B, N, 128)
        target_feature = vehicle_features[:, 0, :]   # (B, 128)

        out = self.decoder(target_feature)  # (B, 60)
        return out.view(-1, self.future_steps, FEATURE_DIM_POSITION)  # (B, 30, 2)
