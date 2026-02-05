import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import json

from managedata.videos import (Video, VideoID, VideoSegment, VideoSegmentID)
from managedata.responses import Responses
from managedata.behavioral import (Gaze, Pupil, Locomotion)
from managedata.extras import load_metadata_from_id

from managedata.videos_duplicates import (same_segments_edges, compute_dissimilarity_videos)

def combine_data(behavioral_list, weights=None):
    """Combine multiple data objects into one by weighted average.

    Args:
        behavioral_list (list of object belonging to class Behavioral): List of data objects to combine. 
        weights (list of float, optional): Weights for each data array. If None, equal weights are used.

    Returns:
        np.array: Combined data .
    """
    if len(behavioral_list) == 0:
        raise ValueError("behavioral_list must contain at least one object.")
    
    data_list = [b.data for b in behavioral_list]
    data_shape = data_list[0].shape
    for data in data_list:
        if data.shape != data_shape:
            raise ValueError("All data arrays must have the same shape.")
    
    if weights is None:
        weights = np.ones(len(data_list)) / len(data_list)
    else:
        weights = np.asarray(weights)
        if len(weights) != len(data_list):
            raise ValueError("Length of weights must match length of data_list.")
        weights = weights / np.sum(weights)  # Normalize weights

    combined_data = np.zeros(data_shape)
    for data, weight in zip(data_list, weights):
        combined_data += data * weight

    return combined_data

class DataSet():

    def __init__(self,folder_data, folder_metadata, recording=None):
        
        self.folder_data = folder_data
        self.folder_metadata = folder_metadata
        self.folder_globalmetadata_videos = os.path.join(self.folder_metadata,'global_meta','videos')
        self.folder_globalmetadata_segments = os.path.join(self.folder_metadata,'global_meta','segments')
        

        if recording is None :
            recording = [p.name for p in Path(folder_data).iterdir() if p.is_dir()]
        self.recording = recording

        # check metadata exists for all recordings

    
    def load_trials_meta(self):

        if not hasattr(self, 'trials_df'):
            print("Loading trials metadata...")
            recording = self.recording
            trials_df = []
            for rec in recording:
                
                path_to_table = os.path.join(self.folder_metadata, rec)
                file = os.path.join(path_to_table,f"meta-trials_{rec}.csv")

                df = pd.read_csv(file)
                df['trial'] = df['trial'].astype(str)
                df.insert(0, "recording", [rec]*len(df))

                if len(trials_df)==0:
                    trials_df = df.copy()
                else:
                    trials_df = pd.concat([trials_df, df], axis=0)
            
            if len(trials_df)==0:
                raise ValueError(f"No trials metadata found in {self.folder_metadata}")
            else:
                self.trials_df = trials_df.reset_index(drop=True)

          

    def filter_trials(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None):

        conditions_val = [recording, label, trial_type, ID, trial, valid_trial]
        conditions_key = ['recording', 'label', 'trial_type','ID', 'trial', 'valid_trial']
        
        # load a table with trials metadata for that recording
        self.load_trials_meta()
        all_trials_df = self.trials_df.copy()
        
        # find trials agreeing on that conditions
        mask = np.full(len(all_trials_df), True)
        for key, val in zip(conditions_key, conditions_val):
            if val is not None:
                if isinstance(val, list):
                    maski = all_trials_df[key].isin(val)
                else:
                    maski = all_trials_df[key]==val
                mask = np.logical_and(mask, maski)
            
        return all_trials_df.loc[mask]
    

    def count_videos_across(self, subset):

        all_conditions = {'recording', 'label', 'trial_type','ID'}
        if not (set(subset)<=all_conditions):
            raise ValueError(f"The subset must be included in {all_conditions}")

        # load a table with all trials metadata
        self.load_trials_meta()

        # count
        counts = self.trials_df.value_counts(subset=subset)
        counts_df = counts.reset_index()
        counts_df.columns = subset+["count"]

        return counts_df
    

    def load_video_by_id(self, id):
        
        files = list(Path(self.folder_globalmetadata_videos).glob(f"*{id}.json"))
        if len(files)!=1:
            raise ValueError(f"{len(files)} files were found with the pattern *{id}.json in {self.folder_globalmetadata_videos}, but 1 was expected")
        
        return VideoID(self.folder_data, self.folder_globalmetadata_videos, files[0].stem.split('-')[1])


    def load_segment_by_id(self, id):
        
        files = list(Path(self.folder_globalmetadata_segments).glob(f"*{id}.json"))
        if len(files)!=1:
            raise ValueError(f"{len(files)} files were found with the pattern *{id}.json in {self.folder_globalmetadata_segments}, but 1 was expected")
        
        return VideoSegmentID(self.folder_data, self.folder_globalmetadata_videos, self.folder_globalmetadata_segments, files[0].stem.split('-')[1])
    
 
    def load_video_by_trial(self, recording, trial):

        # load the data
        recording_folder = os.path.join(self.folder_data, recording)
        video = Video(recording_folder, trial)

        # find the video ID
        trials_meta = self.filter_trials(recording=recording, trial=trial)
        if len(trials_meta)==1:
            video.ID = trials_meta["ID"].iloc[0]
            video.label = trials_meta['label'].iloc[0] 

            # load the metadata 
            video.load_metadata_from_id(self.folder_globalmetadata_videos)
        else:
            raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")

        return video
    

    def load_response_by_trial(self, recording, trial):

        # load the data
        recording_folder = os.path.join(self.folder_data, recording)
        response = Responses(recording_folder, trial)

        # find the video ID
        trials_meta = self.filter_trials(recording=recording, trial=trial)
        if len(trials_meta)==1:
            response.ID = trials_meta["ID"].iloc[0]
            response.label = trials_meta['label'].iloc[0] 

            # load the metadata 
            response.load_metadata_videoid(self.folder_globalmetadata_videos)
        else:
            raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")

        # load neurons metadata
        response.load_metadata_neurons(self.folder_metadata)

        return response
    

    def load_behavior_by_trial(self, recording, trial, behavior_type='pupil'):

        # load the data
        recording_folder = os.path.join(self.folder_data, recording)
        if behavior_type.lower()=='pupil':
            behavior = Pupil(recording_folder, trial)
        elif behavior_type.lower()=='gaze':
            behavior = Gaze(recording_folder, trial)
        elif behavior_type.lower()=='locomotion':
            behavior = Locomotion(recording_folder, trial)
        else:
            raise ValueError(f"behavior_type must be 'pupil', 'gaze', or 'locomotion', got {behavior_type}")
        # find the video ID
        trials_meta = self.filter_trials(recording=recording, trial=trial)
        if len(trials_meta)==1:
            behavior.ID = trials_meta["ID"].iloc[0]
            behavior.label = trials_meta['label'].iloc[0] 
            # load the metadata 
            behavior.load_metadata_videoid(self.folder_globalmetadata_videos)
        else:
            raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")

        return behavior
    

    def load_responses_by(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None):

        trials_df = self.filter_trials(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial)
        responses = []
        for index, row in trials_df.iterrows():
            resp = self.load_response_by_trial(recording=row['recording'], trial=row['trial'])
            responses.append(resp)

        return responses, trials_df
        

    def load_videos_by(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None):

        trials_df = self.filter_trials(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial)
        videos = []
        for index, row in trials_df.iterrows():
            vi = self.load_video_by_trial(recording=row['recording'], trial=row['trial'])
            videos.append(vi)

        return videos, trials_df
    

    def load_behavior_by(self, behavior_type, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None):

        trials_df = self.filter_trials(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial)
        behavior = []
        for index, row in trials_df.iterrows():
            beh = self.load_behavior_by_trial(recording=row['recording'], trial=row['trial'], behavior_type=behavior_type)
            behavior.append(beh)

        return behavior, trials_df
    

    def compute_dissimilarity_videos(self, video_i, video_j, dissimilarity_measure='mse'):
        return compute_dissimilarity_videos(video_i, video_j, dissimilarity_measure=dissimilarity_measure)


    def compute_dissimilarity_video_trials(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None, dissimilarity_measure='mse', check_edges_first=True):

        videos, trials_df = self.load_videos_by(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial)

        n_trials = len(videos)
        dissimilarity = np.full((n_trials,n_trials), np.nan)

        for i, video_i in enumerate(tqdm(videos)):
            
            for j in np.arange(i,n_trials):
                
                video_j = videos[j]

                # check if the segments in which the video is splited are the same
                if check_edges_first:
                    do_comparison = same_segments_edges(video_i, video_j, frames_tolerance=2)
                else:
                    do_comparison = True
                    
                # if same segments, compare
                if do_comparison:
                    dissimilarity[i,j] = compute_dissimilarity_videos(video_i, video_j, dissimilarity_measure=dissimilarity_measure)

        # fill the lower triangle
        i_lower = np.tril_indices(dissimilarity.shape[0], -1)
        dissimilarity[i_lower] = dissimilarity.T[i_lower]

        return  dissimilarity, trials_df
    

    def find_segment(self, segment_id):
        
        # load metadata
        metadata_segment = load_metadata_from_id(segment_id, self.folder_globalmetadata_segments)

        # get the duplicates of the segment
        duplicates_segment = metadata_segment.get("duplicates", {})
        if not duplicates_segment:
            raise ValueError("No duplicates found in segment metadata")
        
        # loop over each video id containig the segment
        trials = []
        recording  = []
        video_label = []
        video_id = []
        segment_label = []
        segment_index = []
        frame_start = []
        frame_end = []
        for v_id, s_duplicates_val in duplicates_segment.items():
            # load metadata video id
            metadata_video = load_metadata_from_id(v_id, self.folder_globalmetadata_videos)
            duplicates_video = metadata_video.get("duplicates", {})
            if not duplicates_video:
                raise ValueError(f"No duplicates found in video metadata for {v_id}")
            for segm_idx in s_duplicates_val["segment_index"]:
                for rec, v_duplicates_val in duplicates_video.items():
                    trl = list(v_duplicates_val['trials'])
                    trials = trials + trl
                    recording = recording + [rec for i in range(len(trl))]
                    video_label = video_label + [metadata_video['label'] for i in range(len(trl))]
                    video_id = video_id + [v_id for i in range(len(trl))]
                    segment_label = segment_label + [metadata_segment['label'] for i in range(len(trl))]
                    segment_index = segment_index + [segm_idx for i in range(len(trl))]
                    frame_start = frame_start + [metadata_video['segments']['frame_start'][segm_idx] for i in range(len(trl))]
                    frame_end = frame_end + [metadata_video['segments']['frame_end'][segm_idx] for i in range(len(trl))]

        return pd.DataFrame({'segment_ID': [segment_id for i in range(len(trials))],
                           'segment_label': segment_label,
                           'video_ID': video_id,
                           'video_label': video_label,
                           'recording': recording,
                           'trial': trials,
                           'segment_index': segment_index,
                           'frame_start': frame_start,
                           'frame_end': frame_end,
                           })
        
        
    def load_segments_meta(self):

        # load the segments metadata
        if not hasattr(self, 'segments_df'):
            print("Loading segments metadata...")
            files = list(Path(self.folder_globalmetadata_segments).glob("*.json"))
            if len(files)==0:
                raise ValueError(f"No json files found in {self.folder_globalmetadata_segments}")
            for i, fff in enumerate(files):
                df = self.find_segment(Path(fff).stem.split('-')[1])
                if i==0:
                    segments_df = df.copy()
                else:
                    segments_df = pd.concat([segments_df, df])
            # store as an attribute
            self.segments_df = segments_df.reset_index(drop=True)

    
    def filter_segments(self, recording=None, video_label=None, segment_label=None, trial=None, video_ID=None, segment_ID=None ):

        conditions_val = [recording, video_label, segment_label, trial, video_ID, segment_ID]
        conditions_key = ['recording', 'video_label', 'segment_label', 'trial','video_ID', 'segment_ID']
        
        # load a table with trials metadata for that recording
        self.load_segments_meta()
        all_segments_df = self.segments_df.copy()
        
        # find trials agreeing on that conditions
        mask = np.full(len(all_segments_df), True)
        for key, val in zip(conditions_key, conditions_val):
            if val is not None:
                if isinstance(val, list):
                    maski = all_segments_df[key].isin(val)
                else:
                    maski = all_segments_df[key]==val
                mask = np.logical_and(mask, maski)
            
        return all_segments_df.loc[mask]
         

    def count_segments_across(self, subset):

        
        all_conditions = {'recording', 'video_label', 'segment_label', 'video_ID','segment_ID','segment_index'}
        if not (set(subset)<=all_conditions):
            raise ValueError(f"The subset must be included in {all_conditions}")

        # load a table with all trials metadata
        self.load_segments_meta()

        # count
        counts = self.segments_df.value_counts(subset=subset)
        counts_df = counts.reset_index()
        counts_df.columns = subset+["count"]

        return counts_df
    

    

 
    
