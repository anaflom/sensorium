# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Obtain neurons metadata for selected recordings."
    )
    parser.add_argument(
        "--folder-data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Data path.",
    )
    parser.add_argument(
        "--folder-metadata",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "metadata",
        help="Metadata path.",
    )
    parser.add_argument(
        "--recordings",
        type=str,
        nargs="+",
        default='none',
        help="Recording IDs to process (space-separated). Use 'none' to apply it to all recordings.",
    )
    return parser.parse_args()


def main(folder_data, folder_meta, recording=None):

    from ssdatam.dataset import DataSet

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data,
                      folder_metadata=folder_meta,
                      recording=recording,
                      trials_metadata_file_type = "csv",
                      trials_metadata_subfolder = "trials",
                      check=False,
                      )
    
    # load a dataframe with all trials metadata
    _ = dataset.get_trials_metadata()

    # define valid frames for the videos and responses
    dataset.define_valid_frames()

    # define valid responses (non-zero)
    dataset.define_valid_responses()

    # save the updated metadata with valid frames and responses
    dataset.save_trials_metadata()


def cli():
    args = parse_args()
    recordings = None if str(args.recordings).lower() == "none" else args.recordings
    try:
        main(args.folder_data, args.folder_metadata, recording=recordings)
        print("\nDetermination of valid frames and responses completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    cli()
