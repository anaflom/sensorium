# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Obtain neurons metadata for selected recordings."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root path containing the code.",
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


def main(repo_root, folder_data, folder_metadata, recording=None):

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.dataset import DataSet

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_metadata, recording=recording)

    # create a folder for the outputs if it doesn't exists
    dataset.create_folders_metadata(what_global_data=[])

    # generate the metadata for all recordings
    dataset.generates_neurons_metadata(recording=recording)


if __name__ == "__main__":

    args = parse_args()
    recordings = None if str(args.recordings).lower() == "none" else args.recordings
    try:
        main(args.repo_root, args.folder_data, args.folder_metadata, recording=recordings)
        print("\nNeurons metadata generation completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")
