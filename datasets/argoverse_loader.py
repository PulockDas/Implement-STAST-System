"""
Argoverse 1 Motion Forecasting dataset loader.

Reads scenario CSV files (e.g. from Kaggle export), groups trajectories by TRACK_ID,
and returns them in a structured format suitable for preprocessing and visualization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


# -----------------------------------------------------------------------------
# Expected column names (Argoverse 1 / Kaggle CSV)
# -----------------------------------------------------------------------------
COL_TIMESTAMP = "TIMESTAMP"
COL_TRACK_ID = "TRACK_ID"
COL_OBJECT_TYPE = "OBJECT_TYPE"
COL_X = "X"
COL_Y = "Y"
COL_CITY = "CITY_NAME"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to expected uppercase names if needed.
    Some CSV exports may use different casing.
    """
    col_map = {
        c.lower(): c
        for c in [COL_TIMESTAMP, COL_TRACK_ID, COL_OBJECT_TYPE, COL_X, COL_Y, COL_CITY]
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns and df.columns.isin([k]).any()}
    if rename:
        return df.rename(columns=rename)
    return df


def load_scenario_csv(
    csv_path: Union[str, Path],
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a single Argoverse scenario CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to the scenario CSV file.
        required_columns: List of column names that must be present.
            Defaults to [TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y].

    Returns:
        DataFrame with at least TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y.
        Sorted by TRACK_ID then TIMESTAMP for consistent ordering.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing after normalization.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario CSV not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_columns(df)

    required = required_columns or [COL_TIMESTAMP, COL_TRACK_ID, COL_OBJECT_TYPE, COL_X, COL_Y]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. Found: {list(df.columns)}"
        )

    # Sort by track then time for stable trajectory order
    df = df.sort_values([COL_TRACK_ID, COL_TIMESTAMP]).reset_index(drop=True)
    return df


def get_trajectories_by_track(
    df: pd.DataFrame,
    include_city: bool = False,
) -> List[Dict[str, Any]]:
    """
    Group scenario DataFrame by TRACK_ID and build a list of trajectory dicts.

    Each trajectory has:
        - track_id: same as TRACK_ID
        - object_type: OBJECT_TYPE (e.g. AGENT, AV, OTHERS)
        - trajectory: list of [x, y] in time order
        - timestamps: list of TIMESTAMP values in same order
        - city_name: (optional) CITY_NAME if include_city and column present

    Args:
        df: DataFrame from load_scenario_csv (must have TRACK_ID, OBJECT_TYPE, X, Y, TIMESTAMP).
        include_city: If True and CITY_NAME exists, add city_name to each trajectory.

    Returns:
        List of trajectory dicts, one per track.
    """
    has_city = COL_CITY in df.columns and include_city
    out = []

    for track_id, group in df.groupby(COL_TRACK_ID, sort=False):
        group = group.sort_values(COL_TIMESTAMP)
        xs = group[COL_X].values.tolist()
        ys = group[COL_Y].values.tolist()
        timestamps = group[COL_TIMESTAMP].values.tolist()
        # First row's object type for this track
        object_type = group[COL_OBJECT_TYPE].iloc[0]

        traj_dict = {
            "track_id": track_id,
            "object_type": object_type,
            "trajectory": [[float(x), float(y)] for x, y in zip(xs, ys)],
            "timestamps": timestamps,
        }
        if has_city:
            traj_dict["city_name"] = group[COL_CITY].iloc[0]
        out.append(traj_dict)

    return out


def load_scenario_from_path(
    csv_path: Union[str, Path],
    include_city: bool = True,
) -> Dict[str, Any]:
    """
    Load one scenario from a CSV path and return trajectories in structured format.

    Convenience function that calls load_scenario_csv and get_trajectories_by_track.

    Args:
        csv_path: Path to the scenario CSV file.
        include_city: Whether to include city_name in each trajectory dict.

    Returns:
        Dict with:
            - "scenario_path": str path
            - "trajectories": list of trajectory dicts (track_id, object_type,
              trajectory, timestamps, optionally city_name)
            - "dataframe": raw DataFrame (for debugging or custom use)
    """
    path = Path(csv_path)
    df = load_scenario_csv(path)
    trajectories = get_trajectories_by_track(df, include_city=include_city)

    return {
        "scenario_path": str(path.resolve()),
        "trajectories": trajectories,
        "dataframe": df,
    }
