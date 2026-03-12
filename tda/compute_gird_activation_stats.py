import numpy as np
import sys
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and save 3D grid activation stats across trials for selected recordings."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root path containing data/, metadata/, and derivatives/.",
    )
    parser.add_argument(
        "--folder-derivatives",
        type=str,
        default='derivatives/grid-15x15x10_normalization-by_minmax',
        help="Relative path to the derivatives folder where the grid stats will be saved for each recording.",
    )
    parser.add_argument(
        "--recordings",
        type=str,
        nargs="+",
        default='none',
        help="Recording IDs to process (space-separated). Use 'none' to apply it to all recordings.",
    )
    return parser.parse_args()


def main(repo_root, folder_derivatives, recordings=None):

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from grids import DataSetGrid

    # base folder
    folder_data = repo_root / "data"
    # results folder
    folder_metadata = repo_root / "metadata"
    # folder derivatives
    folder_derivatives = repo_root / folder_derivatives

    # initialize the object to handle the dataset
    ds = DataSetGrid(folder_data, 
                folder_metadata=folder_metadata, 
                folder_derivatives=folder_derivatives,
                recording=recordings, 
                check=False,
                verbose=True)

    # load a dataframe with all trials metadata
    trials_df = ds.get_trials_metadata()
    
    for rec in recordings:

        stats = ds.compute_grid_stats(rec)

        # save 
        output_folder = folder_derivatives / "grid_stats"
        output_folder.mkdir(parents=True, exist_ok=True)
        for stat_name, stat_value in stats.items():
            np.save(output_folder / f"{stat_name}.npy", stat_value)


if __name__ == "__main__":
    
    args = parse_args()
    recordings = None if str(args.recordings).lower() == "none" else args.recordings
    try:
        main(args.repo_root, args.folder_derivatives, recordings=recordings)
        print("\nGrid activation stats computation completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")



    
