import os
from pathlib import Path

from managedata.videos import VideoSegment
from managedata.videos_duplicates import (compute_dissimilarity_video_list, find_equal_sets_scipy, generate_new_id)
from managedata.handle_dataset import DataSet

def main(folder_data, folder_meta):

    # the labels to check
    labels = ['NaturalImages','GaussianDot','Gabor','PinkNoise','RandomDots']

    # set the limit to decide that two segments are the same or different
    limit_dissimilarity = 20

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta)

    # define segments IDs
    dataset.define_segments_id(labels, limit_dissimilarity=limit_dissimilarity)

    # add the information about segments IDs to the videos metadata
    dataset.add_segments_id_to_video_metadata()


if __name__ == "__main__":

    # path to the folder with the data as downloaded
    folder_data = '/home/anaflo/MDMC/thesis/sensorium/data/'
    # path to the metadata folder
    folder_meta = '/home/anaflo/MDMC/thesis/sensorium/sensorium_metadata/metadata/'


    try:
        main(folder_data, folder_meta)
        print("\nSegment ID definition completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")