import os
import numpy as np

from managedata.handle_dataset import DataSet
from managedata.videos_duplicates import (compare_with_idvideos, find_equal_sets_scipy)

def main(folder_data, folder_meta, folder_results):

    # parameters
    limit_dissimilarity = 5 # set the limit to decide that two videos are the same or different

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta, folder_intermediateresults=folder_results)
            
    # Load the classification tables for all recordings
    videos_df = dataset.get_trials_intermediate_meta(what_data='videos', set_trials_df=True)
    if 'ID' not in videos_df.columns:
        videos_df["ID"] = None


    for recording in dataset.recording:

        print("\n==========================================")
        print(f"Computing for recording {recording}...")
        
        path_to_data = os.path.join(dataset.folder_data, recording)
        path_to_results_metavideos = os.path.join(dataset.folder_intermediateresults, recording, "videos")

        try:

            vdf_rec = videos_df[(videos_df['recording']==recording)]
            all_labels = list(set(vdf_rec['label'].to_list()))

            for thelabel in all_labels:

                print(f"\nLabel {thelabel}")
                print('-------------------')

                try:

                    # compute the dissimilarity
                    dissimilarity, trials_df = dataset.compute_dissimilarity_videos(recording=recording, label=thelabel, verbose=False)

                    # mask the dissimilarity to find identical videos
                    dissimilarity_masked = dissimilarity<limit_dissimilarity

                    # find the groups of videos
                    list_distint_videos = find_equal_sets_scipy(dissimilarity_masked, elements_names=trials_df['trial'].to_list())


                    # compare each of them with the videos already identified for other recordings
                    new_ids = compare_with_idvideos(thelabel, list_distint_videos, 
                                                    path_to_data, path_to_results_metavideos, dataset.folder_globalmetadata_videos, 
                                                    limit_dissimilarity=limit_dissimilarity)
                    
                    # Validate new_ids
                    if len(new_ids) != len(list_distint_videos):
                        raise ValueError(f"Expected {len(list_distint_videos)} IDs, got {len(new_ids)}")

                    # add the info to the trials table
                    for i, duplicate_trials in enumerate(list_distint_videos):
                        mask = (
                            (videos_df["recording"] == recording) &
                            (videos_df["label"] == thelabel) &
                            videos_df["trial"].isin(duplicate_trials) 
                        )
                        if np.sum(mask) != len(duplicate_trials):
                            raise ValueError(f"Label {thelabel}: Expected {len(duplicate_trials)} trials, found {np.sum(mask)}")
                        videos_df.loc[mask,"ID"] = new_ids[i] 

                except Exception as e:
                    print(f"Error processing label {thelabel} in {recording}: {e}")
                    continue

            # save the trials metadata
            folder_recording_meta = os.path.join(dataset.folder_metadata, recording)
            if not os.path.exists(folder_recording_meta):
                os.makedirs(folder_recording_meta)

            df_meta_trials_rec = videos_df[videos_df['recording']==recording].copy()
            df_meta_trials_rec['valid_trial'] = df_meta_trials_rec['segments_bad_n']==0 
            df_meta_trials_rec = df_meta_trials_rec[['label','ID','trial','trial_type','valid_frames','valid_trial']]
            
            filename = os.path.join(folder_recording_meta,f"meta-trials_{recording}.csv")
            df_meta_trials_rec.to_csv(filename, index=False)
            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Error processing recording {recording}: {e}")
            continue


if __name__ == "__main__":

    # data folder
    folder_data = '/home/anaflo/MDMC/thesis/sensorium/data/'
    # metadata folder
    folder_meta = os.path.join('..','metadata')
    # results folder
    folder_results = os.path.join('..','intermediate_results')

    try:
        main(folder_data, folder_meta, folder_results)
        print("\nVideo classification completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")