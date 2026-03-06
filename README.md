# Argoverse Trajectory Prediction — Foundation

A clean, modular Python research repo for **dataset loading**, **trajectory preprocessing**, and **visualization** using the **Argoverse 1 Motion Forecasting** dataset. This forms the foundation for later implementing a trajectory prediction model (e.g. STSAT-style).

## Repository structure

```
project_root/
├── data/                    # Dataset directory (populated after download)
├── datasets/
│   └── argoverse_loader.py  # Load scenario CSVs, group by TRACK_ID
├── preprocessing/
│   └── trajectory_processor.py  # Past/future split, velocity, PyTorch tensors
├── visualization/
│   └── plot_trajectories.py # Plot past vs future, highlight target vehicle
├── notebooks/
│   └── explore_argoverse.ipynb  # End-to-end exploration in Colab
├── utils/
│   └── config.py            # Constants (past/future steps, paths)
├── requirements.txt
└── README.md
```

## How to get the dataset

The project uses the **Argoverse 1 Motion Forecasting** data hosted on Kaggle. Download it programmatically with **kagglehub** (no manual Kaggle API keys needed for public datasets):

```python
import kagglehub
path = kagglehub.dataset_download("narendarmallireddy/argoverse1-motion-dataset")
# CSV scenario files will be under this path (e.g. train/val/test subdirs or flat)
```

- **First time**: `kagglehub` may prompt you to authenticate with Kaggle (e.g. browser or API credentials). See [kagglehub documentation](https://github.com/Kaggle/kagglehub) if needed.
- The downloaded path is typically under your user cache (e.g. `~/.cache/kagglehub`). Use the returned `path` in the notebook to locate scenario CSV files.

## How to run the notebook in Google Colab

1. **Upload or clone this project** into Colab (e.g. upload as ZIP and unzip, or clone from Git).
2. In the notebook, the first cell sets `PROJECT_ROOT` so that `datasets`, `preprocessing`, `visualization`, and `utils` are importable. If you unpacked the project at `/content/Implement-STAST-System`, set:
   ```python
   PROJECT_ROOT = "/content/Implement-STAST-System"
   ```
   and add it to `sys.path`.
3. Run the **pip install** cell to install `pandas`, `numpy`, `matplotlib`, `torch`, `tqdm`, and `kagglehub`.
4. Run the **kagglehub download** cell to download the dataset and get `path`.
5. The next cell discovers CSV files under `path` and picks one as `scenario_path`.
6. Run the remaining cells to load one scenario, visualize trajectories, and run the preprocessing pipeline (tensor shapes and sanity check).

Running all cells in order should give you trajectory plots and printed tensor shapes (e.g. `past_tensor: [num_agents, 20, 4]`, `future_tensor: [num_agents, 30, 2]`).

## How the dataset loader works

- **Input**: Path to a scenario CSV file. Each row is one object at one timestep, with columns such as `TIMESTAMP`, `TRACK_ID`, `OBJECT_TYPE`, `X`, `Y`, and optionally `CITY_NAME`.
- **Reading**: `load_scenario_csv(csv_path)` loads the CSV with pandas and normalizes column names (e.g. lowercase to uppercase). It sorts by `TRACK_ID` and `TIMESTAMP` for stable ordering.
- **Grouping**: `get_trajectories_by_track(df)` groups rows by `TRACK_ID` and builds one trajectory per track. Each trajectory is returned as a dict:
  - `track_id`: same as `TRACK_ID`
  - `object_type`: e.g. `AGENT`, `AV`, `OTHERS`
  - `trajectory`: list of `[x, y]` in time order
  - `timestamps`: list of timestamps in the same order
  - `city_name`: optional, if present in the CSV and requested
- **Convenience**: `load_scenario_from_path(csv_path)` does both steps and returns a dict with `"trajectories"` and the raw `"dataframe"`.

This structured format is what the preprocessing and visualization modules expect. No federated learning, clustering, or encryption is implemented; the scope is dataset loading, preprocessing, and visualization only.

## Dependencies

See `requirements.txt`:

- pandas, numpy, matplotlib, torch, tqdm, kagglehub

Install with:

```bash
pip install -r requirements.txt
```

## Next steps

- Use `prepare_agent_tensors()` output (`past_tensor`, `future_tensor`) as input to a trajectory prediction model.
- Extend the codebase with model training and evaluation when you implement the full STSAT-style pipeline.
