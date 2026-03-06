"""
Trajectory prediction metrics: ADE and FDE.
"""

import torch


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Average Displacement Error: mean Euclidean distance across all predicted timesteps.

    Args:
        pred: (B, T, 2) or (B, T, 2+) — predicted positions (x, y).
        target: (B, T, 2) — ground truth positions (x, y).

    Returns:
        Scalar tensor: mean over batch and timesteps of per-step L2 distance.
    """
    pred_xy = pred[..., :2]
    dist = torch.norm(pred_xy - target, dim=-1)  # (B, T)
    return dist.mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Final Displacement Error: Euclidean distance at the final timestep only.

    Args:
        pred: (B, T, 2) or (B, T, 2+) — predicted positions.
        target: (B, T, 2) — ground truth positions.

    Returns:
        Scalar tensor: mean over batch of L2 distance at last timestep.
    """
    pred_xy = pred[..., :2]
    pred_final = pred_xy[:, -1, :]   # (B, 2)
    target_final = target[:, -1, :]   # (B, 2)
    dist = torch.norm(pred_final - target_final, dim=-1)  # (B,)
    return dist.mean()
