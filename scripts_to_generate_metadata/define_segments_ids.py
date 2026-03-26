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
        "--limit-dissimilarity",
        type=int,
        default=20,
        help="Limit to decide that two segments are the same or different.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["Gabor", "NaturalImages", "GaussianDot", "PinkNoise", "RandomDots"],
        help="Labels to use for the segments IDs definition (space-separated).",
    )
    return parser.parse_args()


def main(folder_data, folder_meta, limit_dissimilarity=20, labels=["Gabor", "NaturalImages", "GaussianDot", "PinkNoise", "RandomDots"]):

    from ssdatam.dataset import DataSet

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, 
                      folder_metadata=folder_meta,
                      trials_metadata_file_type = "csv",
                      trials_metadata_subfolder = "trials",
                      check=False,
                      )

    # define segments IDs
    dataset.define_segments_id(labels, limit_dissimilarity=limit_dissimilarity)

    # add the information about segments IDs to the videos metadata
    dataset.add_segments_id_to_video_metadata()


def cli():
    args = parse_args()
    try:
        main(
            args.folder_data,
            args.folder_metadata,
            limit_dissimilarity=args.limit_dissimilarity,
            labels=args.labels,
        )
        print("\nSegment ID definition completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    cli()
