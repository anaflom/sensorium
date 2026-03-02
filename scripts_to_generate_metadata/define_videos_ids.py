# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.dataset import DataSet


def main(folder_data, folder_meta, folder_results, recording=None):

    # parameters
    limit_dissimilarity = (
        5  # set the limit to decide that two videos are the same or different
    )

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(
        folder_data,
        folder_metadata=folder_meta,
        folder_metadata_per_trial=folder_results,
        recording=recording,
    )

    # define IDs
    dataset.define_videos_id(limit_dissimilarity=limit_dissimilarity)


if __name__ == "__main__":

    # data folder
    folder_data = repo_root / "data"
    # metadata folder
    folder_meta = repo_root / "metadata"
    # results folder
    folder_results = repo_root / "intermediate_results"
    # recordings to define videos IDs for (if None, all recordings will be used)
    recordings_to_use = [
    'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    ]

    try:
        main(folder_data, folder_meta, folder_results, recording=recordings_to_use)
        print("\nVideo ID definition completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")
