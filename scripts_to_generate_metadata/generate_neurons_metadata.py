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

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta)

    # create a folder for the outputs if it doesn't exists
    dataset.create_folders_metadata(what_global_data=[])

    # generate the metadata for all recordings
    dataset.generates_neurons_metadata(recording=recording)


if __name__ == "__main__":

    # path to the folder with the data as downloaded
    folder_data = repo_root / "data"
    # path to the metadata folder
    folder_meta = repo_root / "metadata"
    # recordings to generate metadata for (if None, all recordings will be used)
    recordings_to_process = [
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

    try:
        main(folder_data, folder_meta, recording=recordings_to_process)
        print("\nNeurons metadata generation completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")
