# Sensorium — metadata creation and data handling

A small collection of utilities to create FAIR metadata for the Sensorium dataset, and to load, inspect and preprocess Sensorium recordings (videos, neural responses, gaze, pupil, locomotion).

**Author:** Ana Fló

## Contents
- `utils/`: core dataset classes and helper functions (`DataSet`, `Video`, `VideoSegment`, `Neurons`, `Responses`, `Gaze`, `Pupil`, `Locomotion`).
- `scripts_to_generate_metadata/`: scripts to create and regenerate metadata (see *Regenerating metadata* below).
- `metadata/`: generated metadata output (per-recording and `global_meta/`).
- `intermediate_results/`: temporary outputs produced by processing scripts (classified videos, per-trial results).
- `examples_dataset_exploration/`: Jupyter notebooks demonstrating loading and using the data, and exploring video classification and label distributions.

## Requirements
- Python 3.8+ (3.9 or 3.10 recommended)
- Install dependencies from `requirements.txt`. See that file for exact package versions.

## Installation
1. Clone the repository:

```bash
git clone https://github.com/anaflom/sensorium.git
cd sensorium
```

2. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optionally: `pip install jupyterlab` to run the example notebooks.)

## Data (expected layout)
Download the Sensorium dataset and place it under `./data/`. Expected layout (example):

```
data/
  dynamic29513-3-5-Video-<hash>/
    data/
        videos/
            0.npy
            1.npy
        responses/
            0.npy
            1.npy
        pupil_center/
            0.npy
            1.npy
        bahavior/
            0.npy
            1.npy
    meta/
        neurons/
            cell_motor_coordinates.npy
            unit_ids.npy
        trials/
            tiers.npy
  dynamic29514-2-9-Video-<hash>/
    ...
```

Source: https://gin.g-node.org/pollytur/sensorium_2023_data/src/798ba8ad041d8f0f0ce879af396d52c7238c2730

## Quick usage

Notebook examples:
- `examples_dataset_exploration/example_load_data.ipynb` — loading a recording.
- `examples_dataset_exploration/examples_handle_data.ipynb` — workflows and visualizations.

Programmatic example (copy-pasteable):

```py
import sys
from pathlib import Path
repo_root = Path.cwd().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.dataset import DataSet

repo_root = Path.cwd()
data_root = repo_root / "data"

# Initialize dataset (adjust parameters to your setup)
ds = DataSet(data_root, folder_metadata=repo_root / "metadata")

# Get trials metadata (pandas.DataFrame)
df_trials = ds.get_trials_metadata()

# Filter trials and segments
df_filtered_trials = ds.filter_trials(recording="dynamic29514-2-9-Video-...", label="GaussianDot")
df_filtered_segments = ds.filter_segments(segment_ID="s760623")

# Load a video by recording and trial, or by video ID
video = ds.load_video_by_trial("dynamic29514-2-9-Video-...", "475")
video.display_video_clip()

video2 = ds.load_video_by_id("v054019")
segment = ds.load_segment_by_id("s760623")

# Load responses and associated trial DataFrame
responses, df_trials = ds.load_responses_by(video_ID="v002231")
responses, df_trials = ds.load_responses_by(recording="dynamic29514-2-9-Video-...", label="GaussianDot")

# Single response data (with optional normalization)
data = responses[0].get_data(normalization="by_mean")

# Load behavioral measures
pupil, df_trials = ds.load_behavior_by("pupil", recording="dynamic29514-2-9-Video-...", label="GaussianDot")
gaze, df_trials = ds.load_behavior_by("gaze", recording="dynamic29514-2-9-Video-...", label="GaussianDot")
locomotion, df_trials = ds.load_behavior_by("locomotion", recording="dynamic29514-2-9-Video-...", label="GaussianDot")
```

Notes:
- Replace the example recording/video IDs with the actual IDs present under `./data/`.

## Regenerating metadata — recommended order
Run these scripts in order when regenerating metadata:

1. `classify_videos.py` — classify/analyze trial videos. Outputs per-trial JSON in `intermediate_results/<recording>/`.
2. `define_videos_ids.py` — find equivalent videos and assign unique `videoID`s. Produces `metadata/global_meta/videos/<videoID>.json` and per-recording CSV summaries at `metadata/<recording>/meta-trials_<recording>.csv`. Run after `classify_videos.py`.
3. `define_segments_ids.py` — identify equivalent segments and produce `metadata/global_meta/segments/<segmentID>.json`. Run after `define_videos_ids.py`.
4. `generate_neurons_metadata.py` — create per-recording neuron metadata `metadata/<recording>/meta-neurons_<recording>.csv` (neuron IDs, coordinates, activity stats). This script is independent and may be run anytime.

Notes and tips:
- To regenerate from scratch, remove `metadata/` and `intermediate_results/` first:
```bash
rm -rf metadata/ intermediate_results/
```
- Regenerating from scratch will assign new `videoID`/`segmentID` values; update downstream code that relies on stable IDs.
- If the scripts `define_videos_ids.py` is run before removing the metadata folder, it will compare every trial against the existing video IDs (in `./metadata/global_meta/`), which significantly slows down execution.

## Running the scripts
From the repository root:

```bash
python scripts_to_generate_metadata/classify_videos.py
python scripts_to_generate_metadata/define_videos_ids.py
python scripts_to_generate_metadata/define_segments_ids.py
python scripts_to_generate_metadata/generate_neurons_metadata.py
```

## Contributing
- Please open issues or pull requests for improvements or bug reports.

## License
This project is licensed under the BSD 3-Clause "New" or "Revised" License. See `LICENSE` for details.
