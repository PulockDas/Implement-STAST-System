"""
Scene-level dataset for Argoverse 1 Motion Forecasting.

Each item represents a single scenario (CSV file) with multiple vehicles.

Output per item:
    - past_traj:    (max_vehicles, past_steps, 2)  # relative to target's last observed position
    - future_target:(future_steps, 2)              # AGENT only, same origin
    - vehicle_mask: (max_vehicles,) bool           # True for real vehicles

Trajectories are normalized so the target vehicle's last past point is at (0, 0).
Vehicle at index 0 is always the target vehicle (AGENT).
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from .argoverse_loader import load_scenario_from_path
from preprocessing.trajectory_processor import extract_past_future
from utils.config import (
    PAST_STEPS,
    FUTURE_STEPS,
    OBJECT_TYPE_AGENT,
    OBJECT_TYPE_OTHERS,
)


class ArgoverseSceneDataset(Dataset):
    """
    Scene-level dataset: one sample per CSV scenario with multiple vehicles.

    For each scenario, we:
        - find the AGENT vehicle (target) -> index 0
        - collect nearby OTHERS vehicles (optional distance threshold)
        - extract past (20) and future (30) positions
        - pad/truncate to max_vehicles
    """

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        past_steps: int = PAST_STEPS,
        future_steps: int = FUTURE_STEPS,
        max_vehicles: int = 20,
        distance_threshold: Optional[float] = 50.0,
        min_length: Optional[int] = None,
    ) -> None:
        """
        Args:
            csv_paths: Iterable of scenario CSV paths.
            past_steps: Number of observed timesteps (default 20).
            future_steps: Number of target timesteps (default 30).
            max_vehicles: Max vehicles per scene (pad/truncate to this).
            distance_threshold: If set, only include OTHERS within this
                Euclidean distance (in meters) of the AGENT at the last
                observed past step. If None, keep all OTHERS.
            min_length: Minimum trajectory length to consider for a vehicle.
                Defaults to past_steps + future_steps.
        """
        super().__init__()
        self.csv_paths: List[Path] = [Path(p) for p in csv_paths]
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.max_vehicles = max_vehicles
        self.distance_threshold = distance_threshold
        self.min_length = min_length or (past_steps + future_steps)

        self._scenes: List[Dict[str, Any]] = []
        for csv_path in self.csv_paths:
            scene = self._build_scene(csv_path)
            if scene is not None:
                self._scenes.append(scene)

        if not self._scenes:
            raise RuntimeError(
                "ArgoverseSceneDataset: no valid scenes were constructed. "
                "Check csv_paths and trajectory lengths."
            )

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene = self._scenes[idx]
        return {
            "past_traj": scene["past_traj"],          # (max_vehicles, past_steps, 2)
            "future_target": scene["future_target"],  # (future_steps, 2)
            "vehicle_mask": scene["vehicle_mask"],    # (max_vehicles,)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scene(self, csv_path: Path) -> Optional[Dict[str, Any]]:
        """
        Build a single scene sample from a CSV path.

        Returns:
            Dict with padded tensors, or None if the scene is unusable.
        """
        scenario = load_scenario_from_path(csv_path, include_city=False)
        trajectories: List[Dict[str, Any]] = scenario["trajectories"]

        # 1. Find AGENT (target vehicle)
        agent = None
        for t in trajectories:
            if t.get("object_type") == OBJECT_TYPE_AGENT:
                agent = t
                break
        if agent is None:
            # Skip scenes without AGENT
            return None

        # Agent must have enough history
        if len(agent["trajectory"]) < self.min_length:
            return None

        agent_past, agent_future = extract_past_future(
            agent["trajectory"],
            timestamps=agent["timestamps"],
            past_steps=self.past_steps,
            future_steps=self.future_steps,
        )
        # Origin = last observed position of target vehicle (for relative coordinates)
        origin = agent_past[-1]  # (2,) in map coordinates

        past_list: List[torch.Tensor] = [torch.from_numpy(agent_past[:, :2].astype("float32"))]

        # 2. Collect OTHERS
        for t in trajectories:
            if t is agent:
                continue
            if t.get("object_type") != OBJECT_TYPE_OTHERS:
                continue
            if len(t["trajectory"]) < self.min_length:
                continue
            try:
                past, _ = extract_past_future(
                    t["trajectory"],
                    timestamps=t["timestamps"],
                    past_steps=self.past_steps,
                    future_steps=self.future_steps,
                )
            except ValueError:
                continue

            other_center = past[-1]
            if self.distance_threshold is not None:
                dist = float(((other_center - origin) ** 2).sum() ** 0.5)
                if dist > self.distance_threshold:
                    continue

            past_list.append(torch.from_numpy(past[:, :2].astype("float32")))

            if len(past_list) >= self.max_vehicles:
                break

        num_vehicles = len(past_list)
        if num_vehicles == 0:
            return None

        # 3. Normalize to relative motion: center at target's last observed position
        origin_t = torch.from_numpy(origin.astype("float32"))
        for i in range(len(past_list)):
            past_list[i] = past_list[i] - origin_t
        future_target = torch.from_numpy(agent_future[:, :2].astype("float32")) - origin_t

        # 4. Pad to max_vehicles
        past_tensor = torch.zeros(
            self.max_vehicles, self.past_steps, 2, dtype=torch.float32
        )
        vehicle_mask = torch.zeros(self.max_vehicles, dtype=torch.bool)

        for i, p in enumerate(past_list):
            past_tensor[i, :, :] = p
            vehicle_mask[i] = True

        return {
            "csv_path": str(csv_path),
            "past_traj": past_tensor,
            "future_target": future_target,
            "vehicle_mask": vehicle_mask,
            "num_vehicles": num_vehicles,
        }

