"""
Configuration constants for Argoverse dataset and trajectory processing.
Centralizes magic numbers for easy tuning and Colab compatibility.
"""

# -----------------------------------------------------------------------------
# Dataset (Argoverse 1 Motion Forecasting)
# -----------------------------------------------------------------------------
# Scenario length: 5 seconds at 10 Hz = 50 timesteps total
ARGOVERSE_HZ = 10
ARGOVERSE_SCENARIO_SEC = 5
ARGOVERSE_TOTAL_STEPS = ARGOVERSE_HZ * ARGOVERSE_SCENARIO_SEC  # 50

# Expected CSV columns (Kaggle / standard export)
ARGOVERSE_CSV_COLUMNS = [
    "TIMESTAMP",
    "TRACK_ID",
    "OBJECT_TYPE",
    "X",
    "Y",
    "CITY_NAME",
]

# Object type labels (focal agent is typically AGENT or AV)
OBJECT_TYPE_AGENT = "AGENT"
OBJECT_TYPE_AV = "AV"
OBJECT_TYPE_OTHERS = "OTHERS"

# -----------------------------------------------------------------------------
# Trajectory preprocessing (past / future split)
# -----------------------------------------------------------------------------
# Past: 2 seconds (20 steps) — observed history
# Future: 3 seconds (30 steps) — prediction target
PAST_STEPS = 20
FUTURE_STEPS = 30
# Split index: timestep at which we split past vs future (0-indexed)
PAST_FUTURE_SPLIT_INDEX = PAST_STEPS  # 20

# Feature dimension: (x, y) or (x, y, vx, vy)
FEATURE_DIM_POSITION = 2
FEATURE_DIM_WITH_VELOCITY = 4

# -----------------------------------------------------------------------------
# Paths (override in notebook/Colab)
# -----------------------------------------------------------------------------
# Default relative to project root; set DATA_ROOT in Colab after download
DEFAULT_DATA_ROOT = "data"
# Subfolder names under data root (Kaggle layout may vary)
TRAIN_FOLDER = "train"
VAL_FOLDER = "val"
TEST_FOLDER = "test"
