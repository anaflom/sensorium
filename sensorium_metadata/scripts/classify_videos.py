import os
import numpy as np
from tqdm import tqdm

from managedata.handle_dataset import DataSet

def main(folder_data, folder_results):

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=None, folder_intermediateresults=folder_results)
    
    # run for all recordings
    print("Classifying videos...")
    for recording in dataset.recording:
        
        path_to_video_trials = dataset.get_data_list(recording, what_data='videos')
        print('\n============================================================================')
        print(f"Recording {recording} - {len(path_to_video_trials)} video files found")

        # create a folder for the outputs
        path_to_results_metavideos = dataset.create_folder_intermediate_results(recording, what_data='videos')
        
        # load the trials descriptor
        trial_types = dataset.load_trials_descriptor(recording, verbose=False)
        if len(trial_types) != len(path_to_video_trials):
            raise ValueError("The number of trials in the descriptor does not match the number of video files")
        
        # compute for each video (trial)
        for video_trial, trial_type in tqdm(zip(path_to_video_trials, trial_types), 
                                     total=len(path_to_video_trials),
                                     desc=f"Processing {recording}",
                                     disable=False):

            try:
                # initialize class and load video
                video = dataset.load_video_by_trial(recording, os.path.basename(video_trial), verbose=False)

                # run all the classification
                labels, segments = video.run_all()

                first_label_i = labels[0] if labels else None
                n_segments_peaks_i = len(segments[1]["duration"]) if len(segments) > 1 else 0

                # store some other info in the Video object
                video.first_label = first_label_i
                video.trial_type = trial_type
                video.segments_n_peaks = n_segments_peaks_i
                video.segments_bad_n = np.sum(video.segments["bad_properties"])
                video.segments_avg_duration = np.mean(video.segments['duration'])

                # save some metadata for each video to avoid recomputing later
                fields_to_save = ['recording','trial','trial_type','first_label','label',
                                'ID','sampling_freq','valid_frames','n_peaks',
                                'segments_n_peaks','segments_bad_n','segments_avg_duration']
                video.save_metadata(path_to_results_metavideos, 
                                    metadata_for='exemplar',
                                    main_fields = fields_to_save)
                
            except Exception as e:
                print(f"Error processing video {os.path.basename(video_trial)} in {recording}: {e}")
                continue
            

if __name__ == "__main__":

    # base folder
    folder_data = '/home/anaflo/MDMC/thesis/sensorium/data/'
    # results folder
    folder_results = os.path.join('..','intermediate_results')

    try:
        main(folder_data, folder_results)
        print("\nVideo classification completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")