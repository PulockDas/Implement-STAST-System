"""
Graph layers for vehicle interaction modeling.

- build_adjacency_matrix: constructs spatial adjacency from last observed positions
- CGConvLayer: conditional graph convolution for message passing
"""

import torch
import torch.nn as nn


def build_adjacency_matrix(
    past_traj: torch.Tensor,
    vehicle_mask: torch.Tensor,
    distance_threshold: float = 10.0,
) -> torch.Tensor:
    """
    Build adjacency matrix from last observed vehicle positions.

    Vehicles are connected if pairwise distance < threshold.
    Padded vehicles (mask=False) are excluded from connections.

    Args:
        past_traj: (B, N, 20, 2) — past trajectories
        vehicle_mask: (B, N) — True for real vehicles, False for padded
        distance_threshold: connect if dist < this (meters), default 10.0

    Returns:
        adj: (B, N, N) — adjacency matrix, adj[b,i,j]=1 if connected
             adj[i,i]=1 for real vehicles; padded vehicles have no connections
    """
    # Last observed positions: (B, N, 2)
    pos = past_traj[:, :, -1, :]

    # Pairwise distances: pos_i (B,N,1,2) - pos_j (B,1,N,2) -> (B,N,N,2)
    pos_i = pos.unsqueeze(2)  # (B, N, 1, 2)
    pos_j = pos.unsqueeze(1)  # (B, 1, N, 2)
    diff = pos_i - pos_j
    dist = torch.norm(diff, dim=-1)  # (B, N, N)

    # Connection rule: dist < threshold or self-connection (i==j)
    eye = torch.eye(
        past_traj.size(1),
        device=past_traj.device,
        dtype=torch.bool,
    ).unsqueeze(0).expand(past_traj.size(0), -1, -1)
    connected = (dist < distance_threshold) | eye

    # Respect vehicle_mask: only connect real vehicles
    mask_i = vehicle_mask.unsqueeze(2)  # (B, N, 1)
    mask_j = vehicle_mask.unsqueeze(1)  # (B, 1, N)
    both_real = mask_i & mask_j  # (B, N, N)

    adj = (connected & both_real).float()
    return adj


class CGConvLayer(nn.Module):
    """
    Conditional Graph Convolution layer for vehicle interaction.

    Message passing: h_i_new = ReLU(W_self * h_i + sum_j(adj[i,j] * W_neighbor * h_j))
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W_self = nn.Linear(input_dim, output_dim)
        self.W_neighbor = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        vehicle_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply graph convolution.

        Args:
            vehicle_features: (B, N, 128) — per-vehicle embeddings
            adjacency: (B, N, N) — adjacency matrix

        Returns:
            interaction_features: (B, N, 128)
        """
        # Self contribution: W_self * h_i for each i
        self_out = self.W_self(vehicle_features)  # (B, N, 128)

        # Neighbor aggregation: sum over j of adj[i,j] * W_neighbor * h_j
        # W_neighbor * h_j: (B, N, 128)
        neighbor_transformed = self.W_neighbor(vehicle_features)  # (B, N, 128)

        # adj @ neighbor_transformed: (B, N, N) @ (B, N, 128)
        # For batch matmul: (B, N, N) @ (B, N, 128) -> (B, N, 128)
        neighbor_agg = torch.bmm(adjacency, neighbor_transformed)  # (B, N, 128)

        out = self_out + neighbor_agg
        return torch.relu(out)
