import numpy as np
import json
from tqdm import tqdm

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.dataset import DataSet
from grids import Grid3D, get_ranges_from_positions


def main():       

    # path to the folder with the data as downloaded
    folder_data = repo_root / 'data/'

    # path to the metadata folder
    folder_meta = repo_root / 'metadata'

    # path to the output folder for the grids activity per trial
    folder_derivatives = repo_root / 'derivatives' 
    folder_derivatives.mkdir(exist_ok=True)

    # data normalization to apply before computing the grid activity
    normalization = 'by_minmax'

    # grid shape
    num_grid=(15, 15, 10)

    # recordings to use
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
    # create output folders
    folder_name = 'grid' 
    if normalization is not None:
        folder_name = folder_name + f"_{normalization}"
    folder_output = folder_derivatives / folder_name
    folder_output.mkdir(exist_ok=True)

    # initialize the object to handle the dataset
    ds = DataSet(folder_data, 
                folder_metadata=folder_meta, 
                recording=recordings_to_use, 
                check=False,
                verbose=True)

    trials_df = ds.get_trials_metadata()

    # Compute grid activity per trial and save it
    for rec in ds.recording:
        print(f"Computing grid activity for recording: {rec}")

        folder_output_rec = folder_output / rec
        folder_output_rec.mkdir(exist_ok=True)
        folder_output_rec_trials = folder_output_rec / 'trials'
        folder_output_rec_trials.mkdir(exist_ok=True)
        folder_output_rec_grid = folder_output_rec / 'grid'
        folder_output_rec_grid.mkdir(exist_ok=True)

        
        # Initialize Grid object for that recording
        positions = ds.info[ds.recording[0]]["neurons"].coord_xyz
        xyz_ranges = get_ranges_from_positions(positions)
        grid = Grid3D(xyz_ranges, num_grid)
        
        # Get a DataFrame with all the trials of that recording
        trials_df = ds.filter_trials(recording=rec)

        # Compute grid activity per trial and save it
        all_trials = trials_df['trial'].unique()
        for trial in tqdm(all_trials, total=len(all_trials), desc=f"Processing trials for recording {rec}"):
            response = ds.load_response_by_trial(recording=rec, trial=trial)
            activities = response.get_data(normalization=normalization)
            grid_activity = grid.compute_grid_activity(positions, activities)
            np.save(folder_output_rec_trials / f"{trial}.npy", grid_activity)
        
        # save the grid parameters for that recording
        grid.save(folder_output_rec_grid)

        # save a json file with the parameters used to generate grid activity for that recording
        params = {
            "num_grid": list(num_grid),
            "normalization": normalization,
        }
        with open(folder_output_rec / 'params.json', 'w') as f:
            json.dump(params, f, indent=4)


if __name__ == "__main__":
    try:
        main()
        print("\nComputation of activity over grids completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")
