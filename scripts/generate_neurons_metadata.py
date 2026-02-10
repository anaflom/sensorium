import numpy as np
import pandas as pd
import os
from managedata.handle_dataset import DataSet


def main(folder_data, folder_meta):

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta)

    # create a folder for the outputs if it doesn't exists
    dataset.create_folders_metadata(what_global_data=[])

    # generate the metadata for all recordings
    trials_to_include = ['train','oracle','live_test_main','live_test_bonus','final_test_main','final_test_bonus']
    dataset.generates_neurons_metadata(trials_to_include=trials_to_include)

    
if __name__ == "__main__":

    # path to the folder with the data as downloaded
    folder_data = '/home/anaflo/MDMC/thesis/sensorium/data/'
    # path to the metadata folder
    folder_meta = '/home/anaflo/MDMC/thesis/sensorium/metadata/'

    try:
        main(folder_data, folder_meta)
        print("\nNeurons metadata generation completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")