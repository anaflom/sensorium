from tabnanny import verbose
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import json
import warnings


from utils.metadata import ( validate_global_trials_metadata, 
                                 validate_global_neurons_metadata, 
                                 validate_metadata_video_json, 
                                 validate_metadata_per_trial_json, 
                                 validate_metadata_segment_json,
                                 validate_metadata_video_dict,
                                 check_metadata_integrity,
                                 check_metadata_per_trial_integrity,
                                 )

from utils.videos import (Video, VideoID, VideoSegment, VideoSegmentID)
from utils.responses import Responses
from utils.behavioral import (Gaze, Pupil, Locomotion)
from utils.data_handling import (load_all_data, 
                                     load_metadata_from_id,
                                     load_trials_descriptor,
                                     save_json,
                                     check_data_integrity)

from utils.videos_duplicates import (same_segments_edges, 
                                          compute_dissimilarity_video_list, 
                                          compare_with_idvideos, 
                                          find_equal_sets_scipy,
                                          generate_new_id)


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


def print_title(s, verbose):
    print(f"\n{s:-<100}") if verbose else None


class DataSet():

    def __init__(self, folder_data, folder_metadata=None, 
                 folder_metadata_per_trial=None, recording=None, verbose=True):
        
        print_title('Initializing DataSet ', verbose)

        # set data folders and check they exist
        self.folder_data = folder_data
        if not os.path.exists(folder_data):
            raise ValueError(f"Path does not exist: {folder_data}")
        if recording is None :
            recording = [p.name for p in Path(folder_data).iterdir() if p.is_dir()]
        self.recording = recording

        # Check the data and store some info about it
        self.check_data(verbose=verbose)

        # set metadata folders 
        self.folder_metadata = folder_metadata
        self.folder_metadata_per_trial = folder_metadata_per_trial
        
        # check the metadata folders and files and their consistency with the data files
        self.check_metadata(verbose=verbose)
        self.check_metadata_per_trial(verbose=verbose)


    def __str__(self):
        s = ''
        s= s + f"The dataset contains {len(self.recording)} recordings:\n"
        for rec in self.recording:
            s = s + f"  - {rec} with {self.info[rec]['n_neurons']} neurons recorded, {self.info[rec]['n_trials']} trials, and {self.info[rec]['samples_per_trial']} samples per trial\n"
        if hasattr(self, 'good_metadata') and self.good_metadata:
            s = s + "The recoding has consistent metadata\n"
        if hasattr(self, 'good_metadata_per_trial') and self.good_metadata_per_trial:
            s = s + "The recoding has consistent metadata per trial\n"
        
        return s
        

    def check_data(self, verbose=True):

        print_title('Checking the data ', verbose)

        self.info = {}
        good_data_per_recording = {}
        is_valid = True
        for rec in self.recording:

            # check the data folder
            path_to_data = os.path.join(self.folder_data, rec)
            all_fine, info_data_rec = check_data_integrity(path_to_data, verbose=verbose)
            self.info[rec] = info_data_rec
                
            # stroe the infromation about data quality
            good_data_per_recording[rec] = all_fine

            # print some finel info and store some information
            if all_fine:
                print(f"- All data files seem consistent across trials and data types for recording {rec}.") if verbose else None

            # update the varaible holding wheter data is ok across all recordings
            is_valid = is_valid and all_fine

        # stroe the infromation about data quality   
        self._good_data_per_recording = good_data_per_recording
        self._good_data = is_valid

        if is_valid:
            print(" > VALID data for all recordings in the dataset") if verbose else None
        else:
            print(" > INVALID data") if verbose else None
          


    def check_metadata(self, verbose=True):
        # check that the metadata folder has the corrrect structure and that the files are consistent with the data files 
        
        print_title('Checking metadata ', verbose)

        if not hasattr(self, 'info'):
            raise ValueError("Data must be checked with check_data() before checking metadata")

        if self.folder_metadata is None:
            self.folder_globalmetadata_videos = None
            self.folder_globalmetadata_segments = None
            is_valid = False
            results = None
 
        elif not Path(self.folder_metadata).exists():
            warnings.warn("The metadata folder was set but it does not exist, you can create it with create_folders_metadata()")
            self.folder_globalmetadata_videos = None
            self.folder_globalmetadata_segments = None
            is_valid = False
            results = None

        else:
            self.folder_globalmetadata_videos = Path(self.folder_metadata) / 'global_meta' / 'videos'
            self.folder_globalmetadata_segments = Path(self.folder_metadata) / 'global_meta' / 'segments'
            
            is_valid, results = check_metadata_integrity(self.folder_metadata, 
                                                         self.recording, 
                                                         self.folder_globalmetadata_videos, 
                                                         self.folder_globalmetadata_segments, 
                                                         self.info, 
                                                         verbose=True)
             
        self._good_metadata = is_valid
        if results is not None:
            self._good_metadata_per_recording = results['good_metadata_per_recording']
            self._good_global_meta_videos = results['good_global_meta_videos']
            self._good_global_meta_segments = results['good_global_meta_segments']
        else:
            self._good_metadata_per_recording = {rec:False for rec in self.recording}
            self._good_global_meta_videos = False
            self._good_global_meta_segments = False

        if is_valid:
            print(" > VALID metadata for all recordings in the dataset") if verbose else None
        else:
            print(" > INVALID metadata") if verbose else None
    

    def check_metadata_per_trial(self, verbose=True):
        # check that the metadata folder has the corrrect structure and that the files are consistent with the data files 
        
        print_title('Checking metadata per trial ', verbose)

        if not hasattr(self, 'info'):
            raise ValueError("Data must be checked with check_data() before checking metadata")

        is_valid, per_recording = check_metadata_per_trial_integrity(self.folder_metadata_per_trial, self.recording, self.info, verbose=verbose)

        self._good_metadata_per_trial = is_valid
        self._good_metadata_per_trial_per_recording = per_recording

        if is_valid:
            print(" > VALID metadata per trials for all recordings in the dataset") if verbose else None
        else:
            print(" > INVALID metadata per trials") if verbose else None
               

    def get_data_list(self, recording, what_data='videos'):
        path_to_data = os.path.join(self.folder_data, recording,'data',what_data)        
        return list(Path(path_to_data).glob("*.npy"))
    

    def create_folders_metadata_per_trial(self, recording=None, what_data='videos', verbose=True):
        if self.folder_metadata_per_trial is None:
            raise ValueError("folder_metadata_per_trial is None, cannot create folder for metadata per trial")
        
        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]
        
        if isinstance(what_data, str):
            what_data = [what_data]

        print_title('Creating metadata per trials folders if necessary ', verbose)

        for rec in recording:
            path_to_meta = Path(self.folder_metadata_per_trial) / rec
            created = not path_to_meta.exists()
            path_to_meta.mkdir(parents=True, exist_ok=True)
            if created:
                print(f"- Metadata per trial folder for recording {rec} was created in {path_to_meta}") if verbose else None
            
            for w in what_data:
                path_to_meta_whatdata = path_to_meta / w
                created = not path_to_meta_whatdata.exists()
                path_to_meta_whatdata.mkdir(parents=True, exist_ok=True)
                if created:
                    print(f"- Metadata per trial folder for recording {rec} for {w} data was created in {path_to_meta_whatdata}") if verbose else None
            

    def create_folders_metadata(self, recording=None, what_global_data=['videos','segments'], verbose=True):
        
        if self.folder_metadata is None:
            raise ValueError("folder_metadata is None, cannot create folder for metadata")
        
        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]

        if isinstance(what_global_data, str):
            what_global_data = [what_global_data]

        print_title('Creating metadata folders if necessary ', verbose)
        for rec in recording:
            path_to_meta = Path(self.folder_metadata) / rec
            created = not path_to_meta.exists()
            path_to_meta.mkdir(parents=True, exist_ok=True)
            if created:
                print(f"- Metadata folder for recording {rec} was created in {path_to_meta}") if verbose else None
            
        for w in what_global_data:
            path_to_meta_global = Path(self.folder_metadata) / 'global_meta' / w
            created = not path_to_meta_global.exists()
            path_to_meta_global.mkdir(parents=True, exist_ok=True)
            if created:
                print(f"- Metadata folder for global metadata {w} was created in {path_to_meta_global}") if verbose else None
        if 'videos' in what_global_data:
            self.folder_globalmetadata_videos = os.path.join(self.folder_metadata, 'global_meta','videos')
        if 'segments' in what_global_data:
            self.folder_globalmetadata_segments = os.path.join(self.folder_metadata, 'global_meta','segments')

    

    def get_trials_metadata_per_trials(self, what_data='videos', set_trials_df=True, verbose=True):

        if self.folder_metadata_per_trial is None:
            raise ValueError("folder_metadata_per_trial is None, cannot load trials metadata")

        s = f"Loading trials from metadata {self.folder_metadata_per_trial} for {what_data} "
        print_title(s, verbose)
        all_rows = []

        for rec in self.recording:
            path_to_results = os.path.join(self.folder_metadata_per_trial, rec, what_data)

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
                    
                    # Remove some fields that are not needed and that could be too heavy to store in the table, and add the recording name, then store the row
                    # metadata_filtered = {k: v for k, v in metadata.items() if k != 'segments'}
                    metadata_filtered = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, type(None)))}
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

        trials_df = trials_df.reset_index(drop=True)

        if set_trials_df:
            self.trials_df = trials_df

        return trials_df
            


    def get_trials_metadata(self, set_trials_df=True, verbose=True):

        if self.folder_metadata is None:
            raise ValueError("folder_metadata is None, cannot load trials metadata")
        
        print_title('Loading trials metadata ', verbose)
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
        
        trials_df = trials_df.reset_index(drop=True)
        
        if set_trials_df:
            self.trials_df = trials_df

        return trials_df
          

    def filter_trials(self, recording=None, label=None, trial_type=None, ID=None, trial=None, valid_trial=None):

        conditions_val = [recording, label, trial_type, ID, trial, valid_trial]
        conditions_key = ['recording', 'label', 'trial_type','ID', 'trial', 'valid_trial']
        
        # get a table with trials metadata if not loaded yet
        if not hasattr(self, 'trials_df'):
            raise ValueError("trials_df is not loaded, please run get_trials_metadata() or get_trials_metadata_per_trials() first")
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
    
    
    def get_indexes_of_trials(self, recording, label=None, trial_type=None, ID=None, trial=None, valid_trial=None):
        
        if not isinstance(recording, str) and recording in self.recording:
            raise ValueError(f"'recording' must be a string indicating a recording of the dataset, {recording} is not valid")
        
        # get a data frame with the filtered trials
        filtered_trials_df = self.filter_trials(recording=recording, label=label, trial_type=trial_type, ID=ID, trial=trial, valid_trial=valid_trial)
        the_trials = sorted(filtered_trials_df['trial'].to_list())

        # get the indexes
        trial_indexes = [self.info[recording]['trials'].index(t) for t in the_trials]

        return trial_indexes



    def count_videos_across(self, subset):

        all_conditions = {'recording', 'label', 'trial_type','ID'}
        if not (set(subset)<=all_conditions):
            raise ValueError(f"The subset must be included in {all_conditions}")

        # get a table with trials metadata if not loaded yet
        if not hasattr(self, 'trials_df'):
            raise ValueError("trials_df is not loaded, please run get_trials_metadata() or get_trials_metadata_per_trials() first")
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
    
 
    def load_video_by_trial(self, recording, trial, verbose=True, try_global_first=True):

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
        if self.folder_globalmetadata_videos is not None and try_global_first:
            try:
                video.load_metadata_from_id(self.folder_globalmetadata_videos)
                if verbose:
                    print("Metadata loaded from ID")
                return video
            except Exception as e:
                if verbose:
                    print(f"load_metadata_from_id failed: {e}")

        # fallback: try loading metadata from metadata per trials (if configured)
        if self.folder_metadata_per_trial is not None:
            path_to_metadata_file = os.path.join(self.folder_metadata_per_trial, recording, "videos", f"{trial}.json")
            if os.path.exists(path_to_metadata_file):
                try:
                    video.load_metadata(path_to_metadata_file)
                    if verbose:
                        print("Metadata loaded from metadata per trial")
                except Exception as e:
                    if verbose:
                        print(f"Could not load metadata from metadata per trial: {e}")
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
            raise ValueError("folder_metadata is None, cannot find segments by id")
        
        # load metadata
        metadata_segment, _ = load_metadata_from_id(segment_id, self.folder_globalmetadata_segments)

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
            metadata_video, _ = load_metadata_from_id(v_id, self.folder_globalmetadata_videos)
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
        
        
    def get_segments_meta(self, set_segments_df=True, verbose=True):

        if self.folder_globalmetadata_segments is None or self.folder_globalmetadata_videos is None:
            raise ValueError("folder_metadata is None, cannot load segment by id")
        
        # load the segments metadata
        print_title('Loading segments metadata ', verbose)
        files = list(Path(self.folder_globalmetadata_segments).glob("*.json"))
        if len(files)==0:
            raise ValueError(f"No json files found in {self.folder_globalmetadata_segments}")
        for i, fff in enumerate(files):
            df = self.find_segment(Path(fff).stem.split('-')[1])
            if i==0:
                segments_df = df.copy()
            else:
                segments_df = pd.concat([segments_df, df])

        segments_df = segments_df.reset_index(drop=True)

        # store as an attribute
        if set_segments_df:
            self.segments_df = segments_df

        return segments_df

    
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
    
    def load_all_data(self, recording, what_data, data_slice=None):
        """
        Load all data files from a recording directory.
        
        Args:
            recording (str): Recording name (e.g., 'dynamic29156-11-10-Video-...')
            what_data (str): Data type folder (e.g., 'responses', 'videos')
            
        Returns:
            np.ndarray: Stacked array of all loaded data files
        """
                    
        return load_all_data(os.path.join(self.folder_data, recording), what_data, data_slice=data_slice)
    

    def compute_neurons_stats(self, recording, idx_trials_stats=None):
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

        # get the number of neurons
        n_neurons = self.info[recording]['n_neurons']
        n_trials = self.info[recording]['n_trials']

        # initialize
        stats = {}
        stats['mean_activation'] = np.full(n_neurons, np.nan)
        stats['std_activation'] = np.full(n_neurons, np.nan)
        stats['median_activation'] = np.full(n_neurons, np.nan)
        stats['min_activation'] = np.full(n_neurons, np.nan)
        stats['max_activation'] = np.full(n_neurons, np.nan)

        if idx_trials_stats is None:
            idx_trials_stats = np.ones(n_trials, dtype=bool)

        n_included = np.sum(idx_trials_stats)
        if n_included == 0:
            raise ValueError("No trials included")
        
        print(f"Computing neurons stats for {n_neurons} neurons over {np.sum(idx_trials_stats)} out of {n_trials} total trials")

        for ni in tqdm(range(n_neurons), total=n_neurons, desc="Computing neuron stats", disable=False):
            
            # load all responses
            resp_all = self.load_all_data(recording, what_data='responses', data_slice=(ni, slice(None)))
        
            # compute
            stats['mean_activation'][ni] = np.nanmean(resp_all[idx_trials_stats,:])
            stats['std_activation'][ni] = np.nanstd(resp_all[idx_trials_stats,:])
            stats['median_activation'][ni] = np.nanmedian(resp_all[idx_trials_stats,:])
            stats['min_activation'][ni] = np.nanmin(resp_all[idx_trials_stats,:])
            stats['max_activation'][ni] = np.nanmax(resp_all[idx_trials_stats,:])

        return pd.DataFrame.from_dict(stats)
    

    def generates_neurons_metadata(self, recording=None, idx_trials_stats=None, verbose=True):

        if recording is None:
            recording = self.recording
        
        if isinstance(recording,str):
            recording = [recording]

        # create a folder for the outputs if it doesn't exists
        self.create_folders_metadata(what_global_data=[])

        # compute for all recordings
        print_title('Computing metadata for neurons ', verbose)
        for rec in recording:

            print(f"\nMetadata for recording {rec}") if verbose else None

            try:
                
                # compute the stats
                stats = self.compute_neurons_stats(rec, idx_trials_stats=idx_trials_stats)

                # get neurons metadats
                neurons_coord = self.info[rec]['neurons']['coord']
                if neurons_coord:
                    df_coord = pd.DataFrame(neurons_coord, columns=['coord_x','coord_y','coord_z'])
                else:
                    warnings.warn(f"Neurons coordinates were not defined for recording {rec}")
                    df_coord = pd.DataFrame(np.full((self.info[rec]['n_neurons'], 3), None), columns=['coord_x', 'coord_y', 'coord_z'])

                neurons_ids = self.info[rec]['neurons']['IDs']
                if neurons_ids:
                    df_id = pd.DataFrame(neurons_ids, columns=['ID'])
                else:
                    warnings.warn(f"Neurons IDs were not defined for recording {rec}")
                    df_id = pd.DataFrame(np.full((self.info[rec]['n_neurons'], 1), None), columns=['ID'])

                # generate a dataframe with all neurons info
                meta_neurons = pd.concat([df_id, df_coord, stats], axis=1)

                # save
                folder_recording_meta = Path(self.folder_meta) / rec
                out_path = os.path.join(folder_recording_meta, f"meta-neurons_{rec}.csv")
                meta_neurons.to_csv(out_path, index=False)
                print(f"Saved neurons metadata: {out_path}") if verbose else None

            except Exception as e:
                print(f"Error processing recording {rec}: {e}")
                continue


    def clasiffy_videos(self, recording=None, verbose=True):

        if recording is None:
            recording = self.recording
        
        if isinstance(recording,str):
            recording = [recording]

        # create a folder for the outputs if it doesn't exists
        self.create_folders_metadata_per_trial(what_data='videos')

        print_title('Classifying videos ', verbose)
        for rec in recording:
            
            path_to_video_trials = self.get_data_list(rec, what_data='videos')
            print(f"\nRecording {rec} - {len(path_to_video_trials)} video files found") if verbose else None

            # folder for the outputs
            path_to_results_metavideos = os.path.join(self.folder_metadata_per_trial, rec, 'videos')
            
            # load the trials descriptor
            trial_types = self.info[rec]['trial_type']
            if len(trial_types) != len(path_to_video_trials):
                raise ValueError("The number of trials in the descriptor does not match the number of video files")
            
            # compute for each video (trial)
            for video_trial, trial_type in tqdm(zip(path_to_video_trials, trial_types), 
                                        total=len(path_to_video_trials),
                                        desc=f"Processing {rec}",
                                        disable=False):

                try:
                    # initialize class and load video
                    video = self.load_video_by_trial(rec, os.path.basename(video_trial), verbose=False)

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
                                    'ID','sampling_freq','valid_frames','peaks','n_peaks',
                                    'segments_n_peaks','segments_bad_n','segments_avg_duration']
                    video.save_metadata(path_to_results_metavideos, 
                                        metadata_for='exemplar',
                                        main_fields = fields_to_save)
                    
                except Exception as e:
                    print(f"Error processing video {os.path.basename(video_trial)} in {rec}: {e}")
                    continue
                
    
    def define_videos_id(self, recording=None, limit_dissimilarity=5, verbose=True):

        if recording is None:
            recording = self.recording
        
        if isinstance(recording,str):
            recording = [recording]

        # create a folder for the outputs if it doesn't exists
        self.create_folders_metadata(what_global_data=['videos'])

        # Load the classification tables for all recordings
        videos_df = self.get_trials_metadata_per_trials(what_data='videos', set_trials_df=True)
        if 'ID' not in videos_df.columns:
            videos_df["ID"] = None

        print_title('Defining videos IDs ', verbose)
        for rec in recording:

            print(f"\nComputing for recording {rec}...") if verbose else None
            
            path_to_data = os.path.join(self.folder_data, rec)
            path_to_results_metavideos = os.path.join(self.folder_metadata_per_trial, rec, "videos")

            try:

                vdf_rec = videos_df[(videos_df['recording']==rec)]
                all_labels = list(set(vdf_rec['label'].to_list()))

                for thelabel in all_labels:

                    print(f">>> Label {thelabel}") if verbose else None

                    try:

                        # compute the dissimilarity
                        dissimilarity, trials_df = self.compute_dissimilarity_videos(recording=rec, label=thelabel, verbose=False)

                        # mask the dissimilarity to find identical videos
                        dissimilarity_masked = dissimilarity<limit_dissimilarity

                        # find the groups of videos
                        list_distint_videos = find_equal_sets_scipy(dissimilarity_masked, elements_names=trials_df['trial'].to_list())

                        # compare each of them with the videos already identified for other recordings
                        new_ids = compare_with_idvideos(thelabel, list_distint_videos, 
                                                        path_to_data, path_to_results_metavideos, self.folder_globalmetadata_videos, 
                                                        limit_dissimilarity=limit_dissimilarity)
                        
                        # Validate new_ids
                        if len(new_ids) != len(list_distint_videos):
                            raise ValueError(f"Expected {len(list_distint_videos)} IDs, got {len(new_ids)}")

                        # add the info to the trials table
                        for i, duplicate_trials in enumerate(list_distint_videos):
                            mask = (
                                (videos_df["recording"] == rec) &
                                (videos_df["label"] == thelabel) &
                                videos_df["trial"].isin(duplicate_trials) 
                            )
                            if np.sum(mask) != len(duplicate_trials):
                                raise ValueError(f"Label {thelabel}: Expected {len(duplicate_trials)} trials, found {np.sum(mask)}")
                            videos_df.loc[mask,"ID"] = new_ids[i] 

                    except Exception as e:
                        print(f"Error processing label {thelabel} in {rec}: {e}")
                        continue

                # save the trials metadata
                df_meta_trials_rec = videos_df[videos_df['recording']==rec].copy()
                df_meta_trials_rec['valid_trial'] = df_meta_trials_rec['segments_bad_n']==0 
                df_meta_trials_rec = df_meta_trials_rec[['label','ID','trial','trial_type','valid_frames','valid_trial']]
                
                folder_recording_meta = os.path.join(self.folder_metadata, rec)
                filename = os.path.join(folder_recording_meta,f"meta-trials_{rec}.csv")
                df_meta_trials_rec.to_csv(filename, index=False)
                print(f"Saved: {filename}")

            except Exception as e:
                print(f"Error processing recording {rec}: {e}")
                continue
 
    
    def define_segments_id(self, labels, recording=None, limit_dissimilarity=20, verbose=True):

        if recording is None:
            recording = self.recording
        
        if isinstance(recording,str):
            recording = [recording]

        if isinstance(labels,str):
            labels = [labels]

        # create a folder for the outputs if it doesn't exists
        self.create_folders_metadata(what_global_data=['segments'])

        # validate required folders
        if self.folder_globalmetadata_videos is None or self.folder_globalmetadata_segments is None:
            raise ValueError("folder_globalmetadata_videos and folder_globalmetadata_segments must be set")

        print_title('Finding idenitcal segments ', verbose)
        all_used_ids = []

        for lab in labels:

            print(f">>> Label {lab}") if verbose else None
            
            all_segments = []
            folder = Path(self.folder_globalmetadata_videos)
            json_files = list(folder.glob(f"{lab}*.json"))
            print(f"- {len(json_files)} distint videos found") if verbose else None

            if len(json_files) == 0:
                print(f"Warning: No videos found for label {lab}") if verbose else None
                continue

            # load all segments
            for file_videoID in json_files:
                try:
                    video_id = Path(file_videoID).stem.split('-')[1]
                    video = self.load_video_by_id(video_id)

                    if not hasattr(video, 'segments') or 'frame_start' not in video.segments:
                            print(f"Warning: Video {video_id} has no valid segments") if verbose else None
                            continue

                    for seg_idx in range(len(video.segments['frame_start'])):
                        try: 
                            segment = VideoSegment(video, seg_idx)
                            segment.label_from_parentvideo()
                            all_segments.append(segment)
                        except Exception as e:
                            print(f"Warning: Could not load segment {seg_idx} from video {video_id}: {e}") if verbose else None
                            continue

                except Exception as e:
                    print(f"Warning: Could not load video {video_id}: {e}")
                    continue

            print(f"- {len(all_segments)} segments were found and loaded") if verbose else None

            if len(all_segments) == 0:
                print(f"Warning: No segments found for label {lab}") if verbose else None
                continue

            # compute dissimilarity
            try:
                print('Computing dissimilarity between segments...') if verbose else None
                dissimilarity = compute_dissimilarity_video_list(all_segments, dissimilarity_measure='mse', check_edges_first=False)
            except Exception as e:
                print(f"Error computing dissimilarity for label {lab}: {e}") if verbose else None
                continue

            # extract sets of identical segments
            mask = dissimilarity<=limit_dissimilarity
            list_identical = find_equal_sets_scipy(mask)
            print(f"- {len(list_identical)} different segments were found") if verbose else None

            # loop over identical segments and save metadata 
            print("Saving metadata...") if verbose else None
            for setiden in list_identical:
                try:
                    # generate a new id
                    the_id = generate_new_id(all_used_ids, prefix='s')
                    all_used_ids.append(the_id)

                    # generate a SegmentID object from the exemplar segment and add the duplicates
                    segment_i_id = all_segments[next(iter(setiden))].copy(deep=True)
                    segment_i_id.ID = the_id
                    for k in setiden:
                        segment_i_id.add_duplicates( all_segments[k].parentvideo['ID'], all_segments[k].parentvideo['segment_index'])

                    # save a json file with the video metadata
                    segment_i_id.save_metadata(self.folder_globalmetadata_segments)
                
                except Exception as e:
                    print(f"Error processing segment set: {e}") if verbose else None
                    continue


    def add_segments_id_to_video_metadata(self, verbose=True):

        segments_df = self.get_segments_meta()
        segm = segments_df[['segment_ID','segment_label','video_ID', 'video_label','segment_index']].drop_duplicates()

        print_title('Adding segments IDs info to the videos matadata ', verbose)
        for index, row in segm.iterrows():

            try: 

                # load the video ID metadata
                metadata_video, file_path = load_metadata_from_id(row['video_ID'], self.folder_globalmetadata_videos)
                                    
                # create the segment_ID key if not present
                if not 'segment_ID' in metadata_video['segments']:
                    metadata_video['segments']['segment_ID'] = ["" for _ in range(len(metadata_video['segments']['frame_start']))]

                # add the segment ID
                metadata_video['segments']['segment_ID'][row['segment_index']] = row['segment_ID']

                # validate the video metadata and save
                is_valid = validate_metadata_video_dict(metadata_video)

                if is_valid:
                    save_json(metadata_video, file_path)
                else:
                    warnings.warn(f"segment ID info could no be added to video {row['video_ID']} fro segment {row['segment_index']}")
            
            except Exception as e:
                print(f"Error {e} processing segment: {row['segment_ID']} - video: {row['video_ID']} - segment index {row['segment_index']}") if verbose else None
                continue
            