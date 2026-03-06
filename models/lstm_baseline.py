"""
LSTM encoder-decoder baseline for trajectory prediction.

Predicts 30 future (x, y) positions from 20 past positions (and optional velocity).
No graph interaction, semantic encoder, or attention.
"""

import torch
import torch.nn as nn

try:
    from utils.config import PAST_STEPS, FUTURE_STEPS, FEATURE_DIM_POSITION
except ImportError:
    PAST_STEPS = 20
    FUTURE_STEPS = 30
    FEATURE_DIM_POSITION = 2


class LSTMTrajectoryPredictor(nn.Module):
    """
    LSTM encoder-decoder: encode past trajectory, autoregressively decode 30 future steps.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM_POSITION,
        hidden_size: int = 128,
        past_steps: int = PAST_STEPS,
        future_steps: int = FUTURE_STEPS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.past_steps = past_steps
        self.future_steps = future_steps

        # Encoder: (B, past_steps, input_dim) -> last hidden state
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Decoder: autoregressive; input is embedded (x,y) per step
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        # Embed past/predicted (x, y) to decoder input size for next step
        self.decoder_input_embed = nn.Linear(FEATURE_DIM_POSITION, hidden_size)
        # Decoder hidden -> (x, y)
        self.output_proj = nn.Linear(hidden_size, FEATURE_DIM_POSITION)

    def forward(self, past_traj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_traj: (B, past_steps, input_dim). Positions (x,y) and optionally velocity.

        Returns:
            (B, future_steps, 2) predicted future positions (x, y).
        """
        B = past_traj.size(0)
        device = past_traj.device

        # 1. Encode past trajectory
        _, (h, c) = self.encoder(past_traj)
        # h, c: (1, B, hidden_size)

        # 2. First decoder input: last position (x, y) from past
        last_xy = past_traj[:, -1, : self.input_dim].clone()
        if last_xy.size(-1) > FEATURE_DIM_POSITION:
            last_xy = last_xy[..., :FEATURE_DIM_POSITION]
        decoder_input = self.decoder_input_embed(last_xy)  # (B, hidden_size)

        predictions = []
        for _ in range(self.future_steps):
            decoder_input = decoder_input.unsqueeze(1)  # (B, 1, hidden_size)
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred_xy = self.output_proj(out.squeeze(1))  # (B, 2)
            predictions.append(pred_xy)
            decoder_input = self.decoder_input_embed(pred_xy)  # (B, hidden_size)

        return torch.stack(predictions, dim=1)  # (B, future_steps, 2)
