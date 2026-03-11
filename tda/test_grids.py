import numpy as np
import json
import time

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.dataset import DataSet
from grids import Grid3D, get_ranges_from_positions



# path to the folder with the data as downloaded
folder_data = repo_root / 'data/'

# path to the metadata folder
folder_meta = repo_root / 'metadata'

# data normalization to apply before computing the grid activity
normalization = 'by_minmax'

# grid shape
num_grid=(15, 15, 10)

#recordings to use
recordings_to_use = [
    'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20',
    'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20',
    'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',
    'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20',
    'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20',
   ]

       
# initialize the object to handle the dataset
ds = DataSet(folder_data, 
             folder_metadata=folder_meta, 
             recording=recordings_to_use, 
             check=False,
             verbose=True)

# load a dataframe with all trials metadata
trials_df = ds.get_trials_metadata()


rec = recordings_to_use[0]


# Initialize Grid object for that recording
positions = ds.info[ds.recording[0]]["neurons"].coord_xyz
xyz_ranges = get_ranges_from_positions(positions)
grid = Grid3D(xyz_ranges, num_grid)

# Get a DataFrame with all the trials of that recording
trials_df = ds.filter_trials(recording=rec)

# Compute grid activity per trial and save it
trial = trials_df['trial'].unique()[0]
response = ds.load_response_by_trial(recording=rec, trial=trial)
activities = response.get_data(normalization=normalization)

# Benchmark configuration
n_runs = 10

# Warm-up run
grid_activity = grid.compute_grid_activity(positions, activities)

# Timed runs
times = []
for _ in range(n_runs):
    start = time.perf_counter()
    grid_activity = grid.compute_grid_activity(positions, activities)
    times.append(time.perf_counter() - start)

times = np.asarray(times)
print(
    "compute_grid_activity benchmark "
    f"(runs={n_runs}): "
    f"mean={times.mean():.6f} s, "
    f"std={times.std(ddof=1):.6f} s, "
    f"min={times.min():.6f} s, "
    f"max={times.max():.6f} s"
)
