# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.dataset import DataSet


def main(folder_data, folder_meta, recording=None):

    # the labels to check
    labels = ["NaturalImages", "GaussianDot", "Gabor", "PinkNoise", "RandomDots"]

    # set the limit to decide that two segments are the same or different
    limit_dissimilarity = 20

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta, recording=recording)

    # define segments IDs
    dataset.define_segments_id(labels, limit_dissimilarity=limit_dissimilarity)

    # add the information about segments IDs to the videos metadata
    dataset.add_segments_id_to_video_metadata()


if __name__ == "__main__":

    # path to the folder with the data as downloaded
    folder_data = repo_root / "data"
    # path to the metadata folder
    folder_meta = repo_root / "metadata"
    # recordings to define segments IDs for (if None, all recordings will be used)
    recordings_to_use = [
    'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    ]

    try:
        main(folder_data, folder_meta, recording=recordings_to_use)
        print("\nSegment ID definition completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")
