"""
Plot vehicle trajectories for a single scenario.

Supports past vs future coloring and highlighting the target (focal) vehicle.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    from utils.config import PAST_STEPS, FUTURE_STEPS
except ImportError:
    PAST_STEPS = 20
    FUTURE_STEPS = 30


def plot_scenario_trajectories(
    trajectories: List[Dict[str, Any]],
    past_steps: int = PAST_STEPS,
    future_steps: int = FUTURE_STEPS,
    target_track_id: Optional[Union[str, int]] = None,
    target_index: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = "Scenario trajectories",
    past_color: str = "steelblue",
    future_color: str = "coral",
    target_past_color: str = "darkblue",
    target_future_color: str = "red",
    other_alpha: float = 0.6,
    target_linewidth: float = 2.5,
    other_linewidth: float = 1.0,
    show_start_end: bool = True,
    figsize: Optional[tuple] = (10, 10),
) -> plt.Figure:
    """
    Plot each vehicle's path with past and future in different colors.

    Optionally highlight one vehicle as the target (e.g. AGENT) by track_id or
    index in the trajectory list.

    Args:
        trajectories: List of dicts with "track_id", "trajectory", "object_type".
            Each "trajectory" is list of [x, y] in time order.
        past_steps: Number of steps considered as past (plotted in past_color).
        future_steps: Number of steps considered as future (plotted in future_color).
        target_track_id: If set, the trajectory with this track_id is highlighted.
        target_index: If set and target_track_id is None, trajectory at this index is highlighted.
        ax: Matplotlib axes to draw on. If None, a new figure is created.
        title: Title of the plot.
        past_color: Color for past segment of non-target vehicles.
        future_color: Color for future segment of non-target vehicles.
        target_past_color: Color for past segment of target vehicle.
        target_future_color: Color for future segment of target vehicle.
        other_alpha: Alpha for non-target trajectories.
        target_linewidth: Line width for target vehicle.
        other_linewidth: Line width for other vehicles.
        show_start_end: If True, scatter plot start (green) and end (orange) points.
        figsize: Figure size when ax is None.

    Returns:
        The matplotlib Figure (so caller can save or show).
    """
    split = past_steps
    required_len = past_steps + future_steps

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    for i, t in enumerate(trajectories):
        traj = t.get("trajectory", [])
        if len(traj) < required_len:
            continue

        track_id = t.get("track_id", i)
        is_target = (
            (target_track_id is not None and track_id == target_track_id)
            or (target_index is not None and i == target_index)
        )

        past_traj = np.array(traj[:past_steps])
        future_traj = np.array(traj[split : split + future_steps])

        lw = target_linewidth if is_target else other_linewidth
        alpha = 1.0 if is_target else other_alpha
        pcolor = target_past_color if is_target else past_color
        fcolor = target_future_color if is_target else future_color

        ax.plot(
            past_traj[:, 0],
            past_traj[:, 1],
            color=pcolor,
            linewidth=lw,
            alpha=alpha,
        )
        ax.plot(
            future_traj[:, 0],
            future_traj[:, 1],
            color=fcolor,
            linewidth=lw,
            alpha=alpha,
        )

        if show_start_end:
            ax.scatter(
                past_traj[0, 0],
                past_traj[0, 1],
                color="green",
                s=30 if is_target else 15,
                zorder=5,
                alpha=alpha,
            )
            ax.scatter(
                future_traj[-1, 0],
                future_traj[-1, 1],
                color="orange",
                s=30 if is_target else 15,
                zorder=5,
                alpha=alpha,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Simple legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=past_color, linewidth=2, label="Past"),
        Line2D([0], [0], color=future_color, linewidth=2, label="Future"),
    ]
    if target_track_id is not None or target_index is not None:
        legend_elements.append(
            Line2D([0], [0], color=target_future_color, linewidth=target_linewidth, label="Target vehicle")
        )
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    return fig
