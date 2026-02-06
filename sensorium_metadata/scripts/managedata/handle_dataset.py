from tabnanny import verbose
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import json

from managedata.videos import (Video, VideoID, VideoSegment, VideoSegmentID)
from managedata.responses import Responses
from managedata.behavioral import (Gaze, Pupil, Locomotion)
from managedata.data_loading import (load_metadata_from_id,
                               load_trials_descriptor)

from managedata.videos_duplicates import (same_segments_edges, compute_dissimilarity_video_list)

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

    def __init__(self, folder_data, folder_metadata=None, 
                 folder_intermediateresults=None, recording=None):
        
        self.folder_data = folder_data
        self.folder_metadata = folder_metadata
        self.folder_intermediateresults = folder_intermediateresults
        if self.folder_metadata is not None:
            self.folder_globalmetadata_videos = os.path.join(self.folder_metadata,'global_meta','videos')
            self.folder_globalmetadata_segments = os.path.join(self.folder_metadata,'global_meta','segments')
            if not os.path.exists(self.folder_globalmetadata_videos):
                os.makedirs(self.folder_globalmetadata_videos)
            if not os.path.exists(self.folder_globalmetadata_segments):
                os.makedirs(self.folder_globalmetadata_segments)
        else:
            self.folder_globalmetadata_videos = None
            self.folder_globalmetadata_segments = None

        if recording is None :
            recording = [p.name for p in Path(folder_data).iterdir() if p.is_dir()]
        self.recording = recording

        # check metadata exists for all recordings

    def load_trials_descriptor(self, recording, verbose=False):        
        return load_trials_descriptor(os.path.join(self.folder_data, recording), verbose=verbose)
    

    def get_data_list(self, recording, what_data='videos'):
        path_to_data = os.path.join(self.folder_data, recording,'data',what_data)        
        return list(Path(path_to_data).glob("*.npy"))
    

    def create_folder_intermediate_results(self, recording, what_data='videos'):
        if self.folder_intermediateresults is None:
            raise ValueError("folder_intermediateresults is None, cannot create folder for intermediate results")
        
        path_to_results = os.path.join(self.folder_intermediateresults, recording)
        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)
        path_to_results_whatdata = os.path.join(path_to_results, what_data)
        if not os.path.exists(path_to_results_whatdata):
            os.makedirs(path_to_results_whatdata)

        return path_to_results_whatdata
    

    def get_trials_intermediate_meta(self, what_data='videos', set_trials_df=False):

        if self.folder_intermediateresults is None:
            raise ValueError("folder_intermediateresults is None, cannot load trials metadata")

        print("Loading trials intermediate metadata...")
        all_rows = []

        for rec in self.recording:
            path_to_results = os.path.join(self.folder_intermediateresults, rec, what_data)

            if not os.path.exists(path_to_results):
                print(f"Warning: Path does not exist: {path_to_results}")
                continue

            files = list(Path(path_to_results).glob("*.json"))  

            if len(files) == 0:
                print(f"Warning: No JSON files found in {path_to_results}")
                continue

            for fff in files:
                try:
                    with open(fff, 'r') as f:
                        metadata = json.load(f)
                    
                    # Remove 'segments' key if present
                    metadata_filtered = {k: v for k, v in metadata.items() if k != 'segments'}
                    metadata_filtered['recording'] = rec
                    all_rows.append(metadata_filtered)
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {fff}: {e}")
                    continue
        
        if len(all_rows) == 0:
            print("Warning: No metadata rows collected")
            return pd.DataFrame()

        trials_df = pd.DataFrame(all_rows)
        if 'trial' in trials_df.columns:
            trials_df['trial'] = trials_df['trial'].astype(str)

        if set_trials_df:
            self.trials_df = trials_df.reset_index(drop=True)

        return trials_df.reset_index(drop=True)
            


    def get_trials_meta(self, reload=False, set_trials_df=True):

        if self.folder_metadata is None:
            raise ValueError("folder_metadata is None, cannot load trials metadata")

        if not hasattr(self, 'trials_df') or reload:
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
        
        if set_trials_df:
            self.trials_df = trials_df.reset_index(drop=True)

        return self.trials_df.reset_index(drop=True)
          

    def filter_trials(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None):

        conditions_val = [recording, label, trial_type, ID, trial, valid_trial]
        conditions_key = ['recording', 'label', 'trial_type','ID', 'trial', 'valid_trial']
        
        # get a table with trials metadata if not loaded yet
        if not hasattr(self, 'trials_df'):
            raise ValueError("trials_df is not loaded, please run get_trials_meta() or get_trials_intermediate_meta() first")
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

        # get a table with trials metadata if not loaded yet
        if not hasattr(self, 'trials_df'):
            raise ValueError("trials_df is not loaded, please run get_trials_meta() or get_trials_intermediate_meta() first")
        all_trials_df = self.trials_df.copy()
        
        # count
        counts = all_trials_df.value_counts(subset=subset)
        counts_df = counts.reset_index()
        counts_df.columns = subset+["count"]

        return counts_df
    

    def load_video_by_id(self, id):
        
        if self.folder_globalmetadata_videos is None:
            raise ValueError("folder_metadata is None, cannot load video by id")
        
        files = list(Path(self.folder_globalmetadata_videos).glob(f"*{id}.json"))
        if len(files)!=1:
            raise ValueError(f"{len(files)} files were found with the pattern *{id}.json in {self.folder_globalmetadata_videos}, but 1 was expected")
        
        return VideoID(self.folder_data, self.folder_globalmetadata_videos, files[0].stem.split('-')[1])


    def load_segment_by_id(self, id):
        
        if self.folder_globalmetadata_segments is None or self.folder_globalmetadata_videos is None:
            raise ValueError("folder_metadata is None, cannot load segment by id")
        
        files = list(Path(self.folder_globalmetadata_segments).glob(f"*{id}.json"))
        if len(files)!=1:
            raise ValueError(f"{len(files)} files were found with the pattern *{id}.json in {self.folder_globalmetadata_segments}, but 1 was expected")
        
        return VideoSegmentID(self.folder_data, self.folder_globalmetadata_videos, self.folder_globalmetadata_segments, files[0].stem.split('-')[1])
    
 
    def load_video_by_trial(self, recording, trial, verbose=True):

        # load the data
        recording_folder = os.path.join(self.folder_data, recording)
        video = Video(recording_folder, trial)

        # lookup trial metadata
        try:
            trials_meta = self.filter_trials(recording=recording, trial=trial)
            if len(trials_meta) != 1:
                raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")
        except Exception as e:
            if verbose:
                print(f"Could not load metadata for recording {recording}, trial {trial}: {e}")
            return video

        video.ID = trials_meta["ID"].iloc[0]
        video.label = trials_meta["label"].iloc[0]

        # try loading metadata from global metadata folder (if configured)
        if self.folder_globalmetadata_videos is not None:
            try:
                video.load_metadata_from_id(self.folder_globalmetadata_videos)
                if verbose:
                    print("Metadata loaded from ID")
                return video
            except Exception as e:
                if verbose:
                    print(f"load_metadata_from_id failed: {e}")

        # fallback: try loading metadata from intermediate results (if configured)
        if self.folder_intermediateresults is not None:
            path_to_metadata_file = os.path.join(self.folder_intermediateresults, recording, "videos", f"{trial}.json")
            if os.path.exists(path_to_metadata_file):
                try:
                    video.load_metadata(path_to_metadata_file)
                    if verbose:
                        print("Metadata loaded from intermediate results")
                except Exception as e:
                    if verbose:
                        print(f"Could not load metadata from intermediate results: {e}")
            else:
                if verbose:
                    print(f"Metadata file not found: {path_to_metadata_file}")

        return video
    

    def load_response_by_trial(self, recording, trial, verbose=True):

        # load the data
        recording_folder = os.path.join(self.folder_data, recording)
        response = Responses(recording_folder, trial)

        # find the video ID
        try:
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

        except Exception as e:
            if verbose:
                print(f"Could not load metadata for recording {recording}, trial {trial}: {e}")

        return response
    

    def load_behavior_by_trial(self, recording, trial, behavior_type='pupil', verbose=True):

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
        try:
            trials_meta = self.filter_trials(recording=recording, trial=trial)
            if len(trials_meta)==1:
                behavior.ID = trials_meta["ID"].iloc[0]
                behavior.label = trials_meta['label'].iloc[0] 
                # load the metadata 
                behavior.load_metadata_videoid(self.folder_globalmetadata_videos)
            else:
                raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")
        except Exception as e:
            if verbose:
                print(f"Could not load metadata for recording {recording}, trial {trial}: {e}")

        return behavior
    

    def load_responses_by(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None, verbose=True):

        trials_df = self.filter_trials(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial)
        responses = []
        for index, row in trials_df.iterrows():
            resp = self.load_response_by_trial(recording=row['recording'], trial=row['trial'], verbose=verbose)
            responses.append(resp)

        return responses, trials_df
        

    def load_videos_by(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None, verbose=True):

        trials_df = self.filter_trials(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial)
        videos = []
        for index, row in trials_df.iterrows():
            vi = self.load_video_by_trial(recording=row['recording'], trial=row['trial'], verbose=verbose)
            videos.append(vi)

        return videos, trials_df
    

    def load_behavior_by(self, behavior_type, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None, verbose=True):

        trials_df = self.filter_trials(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial)
        behavior = []
        for index, row in trials_df.iterrows():
            beh = self.load_behavior_by_trial(recording=row['recording'], trial=row['trial'], behavior_type=behavior_type, verbose=verbose)
            behavior.append(beh)

        return behavior, trials_df
    

    def compute_dissimilarity_videos(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None, dissimilarity_measure='mse', check_edges_first=True, verbose=True):
        videos, trials_df = self.load_videos_by(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial, verbose=verbose)
        dissimilarity = compute_dissimilarity_video_list(videos, dissimilarity_measure=dissimilarity_measure, check_edges_first=check_edges_first)
        return  dissimilarity, trials_df   


    def find_segment(self, segment_id):

        if self.folder_globalmetadata_segments is None or self.folder_globalmetadata_videos is None:
            raise ValueError("folder_metadata is None, cannot load segment by id")
        
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
        
        
    def get_segments_meta(self, reload=False):

        if self.folder_globalmetadata_segments is None or self.folder_globalmetadata_videos is None:
            raise ValueError("folder_metadata is None, cannot load segment by id")
        
        # load the segments metadata
        if not hasattr(self, 'segments_df') or reload:
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
        
        # load a table with segments metadata if not loaded yet
        if not hasattr(self, 'segments_df'):
            raise ValueError("segments_df is not loaded, please run get_segments_meta() first")
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

        # load a table with segments metadata if not loaded yet
        if not hasattr(self, 'segments_df'):
            raise ValueError("segments_df is not loaded, please run get_segments_meta() first")
        all_segments_df = self.segments_df.copy()

        # count
        counts = all_segments_df.value_counts(subset=subset)
        counts_df = counts.reset_index()
        counts_df.columns = subset+["count"]

        return counts_df
    
    def load_all_data(self, recording, what_data):
        """
        Load all data files from a recording directory.
        
        Args:
            recording (str): Recording name (e.g., 'dynamic29156-11-10-Video-...')
            what_data (str): Data type folder (e.g., 'responses', 'videos')
            
        Returns:
            np.ndarray: Stacked array of all loaded data files
        """

        path_to_data = os.path.join(self.folder_data, recording, "data", what_data)
        if not os.path.exists(path_to_data):
            raise ValueError(f"Path does not exist: {path_to_data}")
        
        # Get sorted list of .npy files only
        data_files = sorted([f for f in os.listdir(path_to_data) if f.endswith('.npy')])
        
        if len(data_files) == 0:
            raise ValueError(f"No .npy files found in {path_to_data}")
        
        data_all = []
        for file in data_files:
            try:
                filepath = os.path.join(path_to_data, file)
                data = np.load(filepath)
                data_all.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
                continue
        
        if len(data_all) == 0:
            raise ValueError(f"No data successfully loaded from {path_to_data}")
                    
        return np.array(data_all)
    

    def compute_neurons_stats(self, recording, trials_to_include=None):
        '''
        Computes descriptive statistics for each neuron activation pattern across all
        trials belonging to certain conditions.

        Args:
            recording (str): Recording folder name
            trials_to_include (list, optional): List of trial type values to include.
                                              If None, all trials are included
        
        Returns:
            pd.DataFrame: Statistics (mean, std, median, min, max) for each neuron
        '''
        # load all responses
        print("Loading all responses...")
        resp_all = self.load_all_data(recording, what_data='responses')
        
        # Validate response array shape
        if resp_all.ndim != 3:
            raise ValueError(f"Expected 3D response array (trials, neurons, timepoints), got shape {resp_all.shape}")
        
        # compute responses statistics per neuron
        stats = {}
        stats['mean'] = np.full(resp_all.shape[1], np.nan)
        stats['std'] = np.full(resp_all.shape[1], np.nan)
        stats['median'] = np.full(resp_all.shape[1], np.nan)
        stats['min'] = np.full(resp_all.shape[1], np.nan)
        stats['max'] = np.full(resp_all.shape[1], np.nan)

        if trials_to_include!=None:

            # load the trials descriptor
            trials = self.load_trials_descriptor(recording, verbose=False)

            # Validate trials_to_include values exist
            trials_list = list(trials) if not isinstance(trials, list) else trials
            invalid_trials = set(trials_to_include) - set(trials_list)
            if invalid_trials:
                print(f"Warning: trial types {invalid_trials} not found in descriptor")
            
            # select the trials
            idx_trials_stats = np.zeros(resp_all.shape[0], dtype=bool)
            for c in trials_to_include:
                idx_trials_stats = np.logical_or(np.array(trials_list) == c, idx_trials_stats)

            included = sorted(set(np.array(trials_list)[idx_trials_stats]))
            excluded = sorted(set(trials_list) - set(included))
            print("Included trial types: " + ", ".join(f'"{x}"' for x in included))
            if excluded:
                print("Excluded trial types: " + ", ".join(f'"{x}"' for x in excluded))

        else:
            idx_trials_stats = np.ones(resp_all.shape[0], dtype=bool)

        n_included = np.sum(idx_trials_stats)
        if n_included == 0:
            raise ValueError("No trials match the specified criteria")
        
        print(f"Computing neurons stats over {np.sum(idx_trials_stats)} out of {resp_all.shape[0]} total trials")
        print(f"Computing for {resp_all.shape[1]} neurons...")

        for ni in tqdm(range(resp_all.shape[1]), total=resp_all.shape[1], desc="Computing neuron stats", disable=False):
            stats['mean'][ni] = np.nanmean(resp_all[idx_trials_stats,ni,:])
            stats['std'][ni] = np.nanstd(resp_all[idx_trials_stats,ni,:])
            stats['median'][ni] = np.nanmedian(resp_all[idx_trials_stats,ni,:])
            stats['min'][ni] = np.nanmin(resp_all[idx_trials_stats,ni,:])
            stats['max'][ni] = np.nanmax(resp_all[idx_trials_stats,ni,:])

        return pd.DataFrame.from_dict(stats)
    

    

 
    
