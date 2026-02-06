import numpy as np
import pandas as pd
import os
from managedata.handle_dataset import DataSet


def main(folder_data, folder_meta):

    trials_to_include = ['train','oracle','live_test_main','live_test_bonus','final_test_main','final_test_bonus']

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta)

    # compute for all recordings
    for recording in dataset.recording:

        print("\n==========================================")
        print(f"Metadata for neurons for recording {recording}")

        try:
            # load neuron files
            neurons_coord_path = os.path.join(folder_data, recording, "meta", "neurons", "cell_motor_coordinates.npy")
            neurons_ids_path = os.path.join(folder_data, recording, "meta", "neurons", "unit_ids.npy")

            if not os.path.exists(neurons_coord_path) or not os.path.exists(neurons_ids_path):
                print(f"Warning: missing neuron files for {recording}, skipping")
                continue

            neurons_coord = np.load(neurons_coord_path)
            neurons_ids = np.load(neurons_ids_path)

            # compute the stats
            stats = dataset.compute_neurons_stats(recording, trials_to_include=trials_to_include)

            # validate lengths
            n_neurons = len(neurons_ids)
            if stats.shape[0] != n_neurons:
                raise ValueError(f"Mismatch in number of neurons for recording {recording}: "
                                 f"{n_neurons} IDs but stats for {stats.shape[0]} neurons")

            # generate a dataframe with all neurons info
            df_id = pd.DataFrame(neurons_ids, columns=['ID'])
            df_coord = pd.DataFrame(neurons_coord, columns=['coord_x','coord_y','coord_z'])
            meta_neurons = pd.concat([df_id, df_coord, stats], axis=1)

            # save
            folder_recording_meta = os.path.join(folder_meta, recording)
            os.makedirs(folder_recording_meta, exist_ok=True)
            out_path = os.path.join(folder_recording_meta, f"meta-neurons_{recording}.csv")
            meta_neurons.to_csv(out_path, index=False)
            print(f"Saved neurons metadata: {out_path}")

        except Exception as e:
            print(f"Error processing recording {recording}: {e}")
            continue
    
if __name__ == "__main__":

    # path to the folder with the data as downloaded
    folder_data = '/home/anaflo/MDMC/thesis/sensorium/data/'
    # path to the metadata folder
    folder_meta = os.path.join('..','metadata')

    try:
        main(folder_data, folder_meta)
        print("\nNeurons metadata generation completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")