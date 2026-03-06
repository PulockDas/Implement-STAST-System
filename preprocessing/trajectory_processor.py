"""
Trajectory preprocessing for Argoverse motion forecasting.

Extracts past/future windows, computes velocity, and converts to PyTorch tensors
with shapes suitable for trajectory prediction models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Use project config for step counts and feature dims
try:
    from utils.config import (
        PAST_STEPS,
        FUTURE_STEPS,
        PAST_FUTURE_SPLIT_INDEX,
        FEATURE_DIM_POSITION,
        FEATURE_DIM_WITH_VELOCITY,
    )
except ImportError:
    PAST_STEPS = 20
    FUTURE_STEPS = 30
    PAST_FUTURE_SPLIT_INDEX = 20
    FEATURE_DIM_POSITION = 2
    FEATURE_DIM_WITH_VELOCITY = 4


def extract_past_future(
    trajectory: List[List[float]],
    timestamps: Optional[List[Union[int, float]]] = None,
    past_steps: int = PAST_STEPS,
    future_steps: int = FUTURE_STEPS,
    split_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a single trajectory into past and future windows.

    Assumes trajectory is ordered by time and has at least (past_steps + future_steps)
    points. Uses split_index as the first index of the future; default is past_steps.

    Args:
        trajectory: List of [x, y] points in time order.
        timestamps: Optional; unused in split, kept for API consistency.
        past_steps: Number of steps in the past (observed) window.
        future_steps: Number of steps in the future (target) window.
        split_index: Index at which future starts. Defaults to past_steps.

    Returns:
        past: np.ndarray of shape (past_steps, 2), dtype float32.
        future: np.ndarray of shape (future_steps, 2), dtype float32.

    Raises:
        ValueError: If trajectory has insufficient length.
    """
    if split_index is None:
        split_index = past_steps
    required = split_index + future_steps
    if len(trajectory) < required:
        raise ValueError(
            f"Trajectory length {len(trajectory)} < required {required} "
            f"(past up to split {split_index} + future {future_steps})"
        )

    past = np.array(
        trajectory[:past_steps],
        dtype=np.float32,
    )
    future = np.array(
        trajectory[split_index : split_index + future_steps],
        dtype=np.float32,
    )
    return past, future


def compute_velocity(
    trajectory: List[List[float]],
    timestamps: List[Union[int, float]],
) -> np.ndarray:
    """
    Compute instantaneous velocity (vx, vy) per timestep using finite differences.

    First and last steps get forward/backward difference; interior steps use
    central difference. Assumes trajectory and timestamps are same length and
    ordered by time.

    Args:
        trajectory: List of [x, y] in time order.
        timestamps: List of timestamps (seconds or ms) in same order.

    Returns:
        velocities: np.ndarray of shape (len(trajectory), 2), (vx, vy) per step.
    """
    n = len(trajectory)
    if n != len(timestamps):
        raise ValueError("trajectory and timestamps must have same length")
    traj = np.array(trajectory, dtype=np.float64)
    ts = np.array(timestamps, dtype=np.float64)

    vel = np.zeros((n, 2), dtype=np.float32)
    if n <= 1:
        return vel

    # dt between consecutive steps (handle duplicate timestamps)
    dt = np.diff(ts)
    dt = np.where(dt <= 0, np.nan, dt)
    # Forward difference for first step
    if n > 1 and not np.isnan(dt[0]):
        vel[0] = (traj[1] - traj[0]) / dt[0]
    # Backward difference for last step
    if n > 1 and not np.isnan(dt[-1]):
        vel[-1] = (traj[-1] - traj[-2]) / dt[-1]
    # Central difference for interior
    for i in range(1, n - 1):
        if not np.isnan(dt[i - 1]) and not np.isnan(dt[i]):
            vel[i] = (traj[i + 1] - traj[i - 1]) / (dt[i - 1] + dt[i])
        elif not np.isnan(dt[i]):
            vel[i] = (traj[i + 1] - traj[i]) / dt[i]
        elif not np.isnan(dt[i - 1]):
            vel[i] = (traj[i] - traj[i - 1]) / dt[i - 1]

    return vel


def to_torch_tensors(
    past_list: List[np.ndarray],
    future_list: List[np.ndarray],
    add_velocity: bool = True,
    velocity_list: Optional[List[np.ndarray]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stack lists of past/future arrays into PyTorch tensors.

    Optionally concatenate velocity (vx, vy) to position (x, y), giving
    features of dim 4 per timestep.

    Args:
        past_list: List of (past_steps, 2) arrays.
        future_list: List of (future_steps, 2) arrays.
        add_velocity: If True, append (vx, vy) to each step (requires velocity_list).
        velocity_list: Per-agent velocities. If add_velocity, each element is
            (past_steps, 2) for past and we use same for past features; future
            velocity can be computed from future positions or set to 0 (here we
            only use past velocity for past tensor).
        device: Target device for tensors (e.g. "cuda", "cpu").

    Returns:
        past_tensor: (num_agents, past_steps, features), float32.
        future_tensor: (num_agents, future_steps, features), float32.
        Future features are (x, y) only by default; if add_velocity we could
        extend to (x,y,vx,vy) with future velocity — for now we keep future
        as (x, y) for target, and past as (x, y, vx, vy) when add_velocity.
    """
    if len(past_list) != len(future_list):
        raise ValueError("past_list and future_list must have same length")

    num_agents = len(past_list)
    past_steps = past_list[0].shape[0]
    future_steps = future_list[0].shape[0]

    if add_velocity and velocity_list is not None:
        if len(velocity_list) != num_agents:
            raise ValueError("velocity_list length must match number of agents")
        # Past: [x, y, vx, vy] — use velocity for past steps only
        past_blocks = [
            np.concatenate([past_list[i], velocity_list[i]], axis=1)
            for i in range(num_agents)
        ]
        feat_dim = FEATURE_DIM_WITH_VELOCITY
    else:
        past_blocks = list(past_list)
        feat_dim = FEATURE_DIM_POSITION

    past_stack = np.stack(past_blocks, axis=0).astype(np.float32)
    future_stack = np.stack(future_list, axis=0).astype(np.float32)
    # Future target: keep (x, y) only for prediction
    if future_stack.shape[2] != FEATURE_DIM_POSITION:
        future_stack = future_stack[..., :FEATURE_DIM_POSITION]

    past_t = torch.from_numpy(past_stack)
    future_t = torch.from_numpy(future_stack)
    if device is not None:
        past_t = past_t.to(device)
        future_t = future_t.to(device)

    return past_t, future_t


def prepare_agent_tensors(
    trajectories: List[Dict[str, Any]],
    past_steps: int = PAST_STEPS,
    future_steps: int = FUTURE_STEPS,
    add_velocity: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    min_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    From a list of trajectory dicts (from argoverse_loader), extract past/future,
    compute velocity, and return PyTorch tensors plus metadata.

    Only includes agents that have at least (past_steps + future_steps) points.
    Optionally filter by min_length (default: past_steps + future_steps).

    Args:
        trajectories: List of dicts with "trajectory", "timestamps", and optionally
            "object_type", "track_id".
        past_steps: Number of past steps.
        future_steps: Number of future steps.
        add_velocity: Whether to append (vx, vy) to past features.
        device: Device for output tensors.
        min_length: Minimum trajectory length to include; default past_steps + future_steps.

    Returns:
        Dict with:
            - past_tensor: (num_agents, past_steps, features) float32.
            - future_tensor: (num_agents, future_steps, 2) float32.
            - track_ids: list of track_id for each included agent.
            - object_types: list of object_type for each.
            - past_list, future_list: raw lists of arrays (for visualization).
    """
    if min_length is None:
        min_length = past_steps + future_steps

    past_list = []
    future_list = []
    velocity_list = []
    track_ids = []
    object_types = []

    for t in trajectories:
        traj = t["trajectory"]
        ts = t["timestamps"]
        if len(traj) < min_length:
            continue
        try:
            past, future = extract_past_future(
                traj,
                timestamps=ts,
                past_steps=past_steps,
                future_steps=future_steps,
            )
        except ValueError:
            continue

        past_list.append(past)
        future_list.append(future)

        if add_velocity:
            # Velocity for past window only (same length as past)
            vel_full = compute_velocity(traj, ts)
            vel_past = vel_full[:past_steps]
            velocity_list.append(vel_past)
        else:
            velocity_list.append(np.zeros((past_steps, 2), dtype=np.float32))

        track_ids.append(t.get("track_id", len(track_ids)))
        object_types.append(t.get("object_type", "UNKNOWN"))

    if not past_list:
        return {
            "past_tensor": torch.empty(0, past_steps, FEATURE_DIM_WITH_VELOCITY if add_velocity else FEATURE_DIM_POSITION),
            "future_tensor": torch.empty(0, future_steps, FEATURE_DIM_POSITION),
            "track_ids": [],
            "object_types": [],
            "past_list": [],
            "future_list": [],
        }

    past_tensor, future_tensor = to_torch_tensors(
        past_list,
        future_list,
        add_velocity=add_velocity,
        velocity_list=velocity_list if add_velocity else None,
        device=device,
    )

    return {
        "past_tensor": past_tensor,
        "future_tensor": future_tensor,
        "track_ids": track_ids,
        "object_types": object_types,
        "past_list": past_list,
        "future_list": future_list,
    }
