# Project Context — STAST Trajectory Prediction Pipeline

This file summarizes the current state of the project so a new session can quickly understand what exists and what remains to be done.

---

## Project Goal

Reproduce the trajectory prediction part of a research paper (STSAT-style). The pipeline uses the **Argoverse 1 Motion Forecasting dataset** and is designed to run in **Google Colab**.

---

## Current Repository Structure

```
project_root/
├── data/                          # Empty; dataset via kagglehub
├── checkpoints/                    # Saved model weights and metrics
│   └── .gitkeep
├── datasets/
│   ├── argoverse_loader.py         # Load single-scenario CSVs, group by TRACK_ID
│   └── argoverse_dataset.py        # Scene-level dataset (multi-vehicle, normalized)
├── preprocessing/
│   └── trajectory_processor.py    # Past/future split, velocity, PyTorch tensors
├── models/
│   ├── lstm_baseline.py            # LSTM encoder-decoder (single-vehicle)
│   ├── temporal_encoder.py        # BiGRU encoder + BiGRUGraphTrajectoryEncoder
│   └── graph_layers.py            # build_adjacency_matrix, CGConvLayer
├── metrics/
│   └── trajectory_metrics.py      # ADE, FDE
├── visualization/
│   └── plot_trajectories.py       # plot_scenario_trajectories, plot_trajectory
├── utils/
│   └── config.py                  # PAST_STEPS=20, FUTURE_STEPS=30, etc.
├── notebooks/
│   ├── explore_argoverse.ipynb    # Single-scenario exploration + LSTM baseline
│   └── train_trajectory_models.ipynb  # Scene-level BiGRU training
├── requirements.txt
├── README.md
└── CONTEXT.md                     # This file
```

---

## What Has Been Implemented

### 1. Dataset Loading

- **`argoverse_loader.py`**: Reads scenario CSVs, groups by `TRACK_ID`, returns list of dicts with `track_id`, `object_type`, `trajectory`, `timestamps`.
- **`argoverse_dataset.py`**: Scene-level dataset for multi-vehicle training.
  - One sample per CSV scenario.
  - Target vehicle (AGENT) at index 0; OTHERS within 50 m distance threshold.
  - **Normalized coordinates**: trajectories centered at target's last observed position `(0, 0)`.
  - Output: `past_traj (max_vehicles, 20, 2)`, `future_target (30, 2)`, `vehicle_mask (max_vehicles)`.

### 2. Preprocessing

- **`trajectory_processor.py`**: `extract_past_future`, `compute_velocity`, `prepare_agent_tensors`, `to_torch_tensors`.
- Past: 20 steps (2 s at 10 Hz); future: 30 steps (3 s).

### 3. Models

- **`LSTMTrajectoryPredictor`**: LSTM encoder-decoder, input `(B, 20, 2)` → output `(B, 30, 2)`. Used in explore notebook with single-scenario data.
- **`BiGRUTrajectoryEncoder`**: BiGRU encoder (input 2 → hidden 64 → output 128 per vehicle) + temporary MLP decoder (target vehicle only). Input `(B, N, 20, 2)` → output `(B, 30, 2)`.
- **`BiGRUGraphTrajectoryEncoder`**: BiGRU encoder + CGConv interaction layer + decoder. Pipeline: past_traj → BiGRU → vehicle_features → adjacency → CGConv → interaction_features → target_feature → decoder → `(B, 30, 2)`. Requires `vehicle_mask` for graph construction.
- **`graph_layers.py`**: `build_adjacency_matrix(past_traj, vehicle_mask)` — spatial adjacency (dist < 10 m); `CGConvLayer` — message passing with W_self and W_neighbor.

### 4. Metrics

- **ADE** (Average Displacement Error): mean L2 distance over all timesteps.
- **FDE** (Final Displacement Error): L2 distance at final timestep.

### 5. Visualization

- **`plot_scenario_trajectories`**: Past vs future, highlight target vehicle (explore notebook).
- **`plot_trajectory`**: Past (blue), gt future (green), pred future (red) for one sample.

### 6. Notebooks

- **`explore_argoverse.ipynb`**: Clone repo, kagglehub download, load one scenario, visualize, preprocessing, LSTM baseline on single-scenario agents.
- **`train_trajectory_models.ipynb`**: Clone repo, kagglehub download, scene dataset (max_scenes=5000), train/val split, BiGRU training, ADE/FDE logging, trajectory plot, save checkpoints and metrics.

### 7. Training Pipeline (train_trajectory_models.ipynb)

- Dataset: `ArgoverseSceneDataset` with `max_scenes`, train/val split (90/10).
- Hyperparameters: `batch_size=32`, `lr=1e-3`, `epochs=10`, Adam, MSELoss.
- BiGRU only: saves `checkpoints/bigru_temporal_model.pth`, `checkpoints/bigru_temporal_checkpoint.pth`, `checkpoints/bigru_training_metrics.json`.
- BiGRU+Graph (Step 5): saves `checkpoints/bigru_graph_model.pth`, `checkpoints/bigru_graph_checkpoint.pth`, `checkpoints/bigru_graph_training_metrics.json`.

---

## Key Design Decisions

1. **Relative coordinates**: All trajectories are normalized so the target vehicle's last past point is at `(0, 0)`. This avoids huge ADE/FDE from absolute map coordinates.
2. **Vehicle index 0 = target (AGENT)**: Always.
3. **Graph interaction (Step 5)**: `build_adjacency_matrix` (dist < 10 m) + `CGConvLayer` for vehicle interaction. No semantic encoder / attention yet.

---

## Dataset Format (Argoverse 1)

- CSV columns: `TIMESTAMP`, `TRACK_ID`, `OBJECT_TYPE`, `X`, `Y`, `CITY_NAME`.
- Object types: `AGENT` (target), `AV`, `OTHERS`.
- 5 s at 10 Hz = 50 timesteps; we use 20 past + 30 future.

---

## How to Run

1. **Colab**: Clone repo (or upload), run clone cell, set `PROJECT_ROOT`, pip install, kagglehub download.
2. **Explore notebook**: Single scenario, visualization, LSTM on agents from one CSV.
3. **Train notebook**: Scene dataset, BiGRU training, metrics, checkpoints.

---

## What Is NOT Implemented Yet

- Semantic feature encoder
- Multi-head attention fusion
- Federated learning, clustering, encryption

---

## Expected Performance (BiGRU on normalized data)

- ADE ≈ 2–4 m
- FDE ≈ 4–8 m
- Predicted trajectories should visually follow ground-truth motion.
