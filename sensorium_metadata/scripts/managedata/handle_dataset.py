from tabnanny import verbose
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import json
import warnings


from managedata.metadata import ( validate_global_trials_metadata, 
                                 validate_global_neurons_metadata, 
                                 validate_metadata_video_json, 
                                 validate_metadata_per_trial_json, 
                                 validate_metadata_segment_json)

from managedata.videos import (Video, VideoID, VideoSegment, VideoSegmentID)
from managedata.responses import Responses
from managedata.behavioral import (Gaze, Pupil, Locomotion)
from managedata.data_loading import (load_all_data, 
                                     load_metadata_from_id,
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
                 folder_metadata_per_trial=None, recording=None, verbose=True):
        
        print("Initializing DataSet...") if verbose else None

        # set data folders and check they exist
        self.folder_data = folder_data
        if not os.path.exists(folder_data):
            raise ValueError(f"Path does not exist: {folder_data}")
        if recording is None :
            recording = [p.name for p in Path(folder_data).iterdir() if p.is_dir()]
        self.recording = recording

        # Check the data and store some info about it
        self.check_data(verbose=verbose)

        # set metadata folders and check they exist (if not None)
        self.folder_metadata = folder_metadata
        if folder_metadata is not None and not os.path.exists(folder_metadata):
            raise ValueError(f"Path does not exist: {folder_metadata}")
        
        self.folder_metadata_per_trial = folder_metadata_per_trial
        if folder_metadata_per_trial is not None and not os.path.exists(folder_metadata_per_trial):
            raise ValueError(f"Path does not exist: {folder_metadata_per_trial}")
        
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


        print("Checking data per trial ------------------------------------------") if verbose else None
        
        info = {}
        is_valid = True
        for rec in self.recording:

            all_fine = True

            path_to_data = os.path.join(self.folder_data, rec, "data")
            if not os.path.exists(path_to_data):
                warnings.warn(f"Warning: Path does not exist: {path_to_data}")
                all_fine = False
                continue

            n_trials = {}
            data_shape = {}
            samples_per_trial = {}
            the_trials = {}
            n_neurons = []
            for what_data in ['responses', 'videos', 'behavior','pupil_center']:
                path_to_whatdata = os.path.join(path_to_data, what_data)
                if not os.path.exists(path_to_whatdata):
                    warnings.warn(f"Warning: Path does not exist: {path_to_whatdata}")
                    all_fine = False
                    samples_per_trial[what_data] = None
                    n_trials[what_data] = 0
                    continue
                files = list(Path(path_to_whatdata).glob("*.npy"))
                if len(files) == 0:
                    warnings.warn(f"Warning: No .npy files found in {path_to_whatdata}")
                    all_fine = False
                    samples_per_trial[what_data] = None
                    n_trials[what_data] = 0
                    continue

                n_trials[what_data] = len(files)
                the_trials[what_data] = set([Path(f).stem for f in files])
                for fff in files:
                    try:
                        data = np.load(fff, mmap_mode='r')
                        if what_data not in data_shape:
                            data_shape[what_data] = data.shape
                            samples_per_trial[what_data] = data.shape[-1]
                        else:
                            if data.shape != data_shape[what_data]:
                                warnings.warn(f"Warning: Different data shapes across {what_data} files in {rec}: {data.shape} vs {data_shape[what_data]}")
                                all_fine = False
                            if data.shape[-1] != samples_per_trial[what_data]:
                                warnings.warn(f"Warning: Different number of samples per trial across {what_data} files in {rec}: {data.shape[-1]} vs {samples_per_trial[what_data]}")
                                all_fine = False
                        if what_data=='responses':
                            n_neurons.append(data.shape[0])
                    except Exception as e:
                        warnings.warn(f"Warning: Could not load {fff}: {e}")
                        all_fine = False

            if len(set(n_trials.values()))>1:
                warnings.warn(f"Warning: Different number of trials across data types in {rec}: {n_trials}")
                all_fine = False
                n_trials = None
            else:
                n_trials = set(n_trials.values()).pop()
            
            if not all(s == the_trials['responses'] for s in the_trials.values()):
                warnings.warn(f"Warning: Different trial files across data types in {rec}")
                all_fine = False
            else:
                the_trials = the_trials['responses']

            if len(set(samples_per_trial.values()))>1:
                warnings.warn(f"Warning: Different number of samples per trial across data types in {rec}: {samples_per_trial}")
                all_fine = False
                samples_per_trial = None
            else:
                samples_per_trial = set(samples_per_trial.values()).pop()

            unique_n_neurons = set(n_neurons)
            if len(unique_n_neurons)==0:
                all_fine = False
                n_neurons = None
            elif len(unique_n_neurons)>1:
                    warnings.warn(f"Warning: Different number of neurons across response files in {rec}: {unique_n_neurons}")
                    all_fine = False
                    n_neurons = None
            else:
                n_neurons = unique_n_neurons.pop()
            

            if all_fine:
                print(f"- All data files seem consistent across trials and data types for recording {rec}.") if verbose else None

            is_valid = is_valid and all_fine

            info[rec] = {
                'n_trials': n_trials,
                'trials': the_trials,
                'samples_per_trial': samples_per_trial,
                'n_neurons': n_neurons,
            }

        self.info = info
        self.good_data = is_valid


    def check_metadata(self, verbose=True):
        # check that the metadata folder has the corrrect structure and that the files are consistent with the data files 
        
        print("Checking metadata ------------------------------------------") if verbose else None

        if not hasattr(self, 'info'):
            raise ValueError("Data must be checked with check_data() before checking metadata")

        if self.folder_metadata is not None:

            is_valid = True

            self.folder_globalmetadata_videos = os.path.join(self.folder_metadata,'global_meta','videos')
            self.folder_globalmetadata_segments = os.path.join(self.folder_metadata,'global_meta','segments')
            if not os.path.exists(self.folder_globalmetadata_videos):
                os.makedirs(self.folder_globalmetadata_videos)
                warnings.warn(f"Warning: folder_globalmetadata_videos did not exist, created it: {self.folder_globalmetadata_videos}")
            if not os.path.exists(self.folder_globalmetadata_segments):
                os.makedirs(self.folder_globalmetadata_segments)
                warnings.warn(f"Warning: folder_globalmetadata_segments did not exist, created it: {self.folder_globalmetadata_segments}")

            # check all the video metadata files
            files = list(Path(self.folder_globalmetadata_videos).glob(f"*.json"))
            good_files = np.full(len(files), False)
            for i, fff in enumerate(files):
                try:
                    _, good_files[i] = validate_metadata_video_json(fff)
                except Exception as e:
                    warnings.warn(f"Warning: Could not validate metadata video json file {fff}: {e}")
            print(f"{good_files.sum()} out of {len(files)} metadata video json files are valid in {self.folder_globalmetadata_videos}") if verbose else None
            is_valid = is_valid and good_files.sum()==len(files)

            # check all the segment metadata files
            files = list(Path(self.folder_globalmetadata_segments).glob(f"*.json"))
            good_files = np.full(len(files), False)
            for i, fff in enumerate(files):
                try:
                    _, good_files[i] = validate_metadata_segment_json(fff)
                except Exception as e:
                    warnings.warn(f"Warning: Could not validate metadata segment json file {fff}: {e}")
            print(f"{good_files.sum()} out of {len(files)} metadata segment json files are valid in {self.folder_globalmetadata_segments}") if verbose else None
            is_valid = is_valid and good_files.sum()==len(files)

            # check the metadata for each recording and its consistency with the data files
            for rec in self.recording:
                all_fine = True

                # check if a folder for the recording exists in the metadata folder
                path_to_table = os.path.join(self.folder_metadata, rec)
                if not os.path.exists(path_to_table):
                    warnings.warn(f"Warning: Path does not exist: {path_to_table}")
                    all_fine = False
                    continue
                
                # check the trials metadata file
                file = os.path.join(path_to_table,f"meta-trials_{rec}.csv")
                if not os.path.exists(file):
                    warnings.warn(f"Warning: File does not exist: {file}")
                    all_fine = False
                    continue
                try:
                    df, is_ok = validate_global_trials_metadata(file)
                    all_fine = all_fine and is_ok

                    # check that the trials in the metadata file are consistent with the trials in the data files
                    if 'trial' in df.columns:
                        if not df['trial'].isin(self.info[rec]['trials']).all():
                            triasl_diff = set(df['trial'].values) - self.info[rec]['trials']
                            all_fine = False
                            warnings.warn(f"Warning: {len(triasl_diff)} trials in {file} are not found in data files for recording {rec}: {triasl_diff}")

                    # check that the IDs in the metadata file exist in the IDs in the global metadata videos folder (if configured)
                    if 'ID' in df.columns:
                        the_ids = set(df['ID'].values)
                        for id in the_ids:
                            files = list(Path(self.folder_globalmetadata_videos).glob(f"*{id}.json"))
                            if len(files)==0:
                                all_fine = False
                                warnings.warn(f"Warning: No metadata file found for ID {id} in folder_globalmetadata_videos")
                            elif len(files)>1:
                                all_fine = False
                                warnings.warn(f"Warning: Multiple metadata files found for ID {id} in folder_globalmetadata_videos: {[f.name for f in files]}")

                except Exception as e:
                    warnings.warn(f"Warning: Could not load {file}: {e}")
                    all_fine = False
                    continue

                # check the neurons metadata file
                file = os.path.join(path_to_table,f"meta-neurons_{rec}.csv")
                df, is_ok = validate_global_neurons_metadata(file)
                all_fine = all_fine and is_ok
                if len(df)!=self.info[rec]['n_neurons']:
                    all_fine = False
                    warnings.warn(f"Warning: Number of neurons in {file} does not match the number of neurons in the data files for recording {rec}: {len(df)} vs {self.info[rec]['n_neurons']}")

                # print if all fine for the recording
                if all_fine:
                    print(f"- Metadata seems ok for recording {rec}.") if verbose else None

                is_valid = is_valid and all_fine
            
            
        else:
            self.folder_globalmetadata_videos = None
            self.folder_globalmetadata_segments = None
            is_valid = False

        self.good_metadata = is_valid
    

    def check_metadata_per_trial(self, verbose=True):
        # check that the metadata folder has the corrrect structure and that the files are consistent with the data files 
        
        print("Checking metadata per trial ------------------------------------------") if verbose else None

        if not hasattr(self, 'info'):
            raise ValueError("Data must be checked with check_data() before checking metadata")

        if self.folder_metadata_per_trial is not None:
            is_valid = True

            # check the metadata for each recording and its consistency with the data files
            for rec in self.recording:
                all_fine = True

                # check if a folder for the recording exists in the metadata folder
                path_to_table = os.path.join(self.folder_metadata_per_trial, rec, 'videos')
                if not os.path.exists(path_to_table):
                    warnings.warn(f"Warning: Path does not exist: {path_to_table}")
                    all_fine = False
                    continue

                # check all the video metadata files
                files = list(Path(self.folder_globalmetadata_videos).glob(f"*.json"))
                good_files = np.full(len(files), False)
                trials = []
                for i, fff in enumerate(files):
                    try:
                        metadata, good_files[i] = validate_metadata_per_trial_json(fff)
                        trials.append(metadata['trial'])
                    except Exception as e:
                        all_fine = False
                        warnings.warn(f"Warning: Could not validate metadata per trial json file {fff}: {e}")
                print(f"- Found {len(files)} trial metadata files for recording {rec}, {good_files.sum()} are valid")

                if set(trials) != self.info[rec]['trials']:
                    trials_diff = set(trials) - self.info[rec]['trials']
                    all_fine = False
                    warnings.warn(f"Warning: Trials in metadata per trial json files do not match the trials in the data files for recording {rec}: {trials_diff}")
                
                # print if all fine for the recording
                if all_fine:
                    print(f"- Metadata per trials seems ok for recording {rec}.") if verbose else None

                is_valid = is_valid and all_fine

        else:
            is_valid = False

        self.good_metadata_per_trial = is_valid
                
                

    def load_trials_descriptor(self, recording, verbose=False):        
        return load_trials_descriptor(os.path.join(self.folder_data, recording), verbose=verbose)
    

    def get_data_list(self, recording, what_data='videos'):
        path_to_data = os.path.join(self.folder_data, recording,'data',what_data)        
        return list(Path(path_to_data).glob("*.npy"))
    

    def create_folder_metadata_per_trial(self, recording, what_data='videos'):
        if self.folder_metadata_per_trial is None:
            raise ValueError("create_folder_metadata_per_trial is None, cannot create folder for metadata per trial")
        
        path_to_meta = os.path.join(self.folder_metadata_per_trial, recording)
        if not os.path.exists(path_to_meta):
            os.makedirs(path_to_meta)
        path_to_meta_whatdata = os.path.join(path_to_meta, what_data)
        if not os.path.exists(path_to_meta_whatdata):
            os.makedirs(path_to_meta_whatdata)

        return path_to_meta_whatdata
    

    def get_trials_metadata_per_trials(self, what_data='videos', set_trials_df=False):

        if self.folder_metadata_per_trial is None:
            raise ValueError("folder_metadata_per_trial is None, cannot load trials metadata")

        print(f"Loading trials from metadata {self.folder_metadata_per_trial} for {what_data}...")
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
            


    def get_trials_metadata(self, reload=False, set_trials_df=True):

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
        
        return self.trials_df.reset_index(drop=True)
          

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

        if trials_to_include!=None:

            # load the trials descriptor
            trials = self.load_trials_descriptor(recording, verbose=False)

            # Validate trials_to_include values exist
            trials_list = list(trials) if not isinstance(trials, list) else trials
            invalid_trials = set(trials_to_include) - set(trials_list)
            if invalid_trials:
                print(f"Warning: trial types {invalid_trials} not found in descriptor")
            
            # select the trials
            idx_trials_stats = np.zeros(n_trials, dtype=bool)
            for c in trials_to_include:
                idx_trials_stats = np.logical_or(np.array(trials_list) == c, idx_trials_stats)

            included = sorted(set(np.array(trials_list)[idx_trials_stats]))
            excluded = sorted(set(trials_list) - set(included))
            print("Included trial types: " + ", ".join(f'"{x}"' for x in included))
            if excluded:
                print("Excluded trial types: " + ", ".join(f'"{x}"' for x in excluded))

        else:
            idx_trials_stats = np.ones(n_trials, dtype=bool)

        n_included = np.sum(idx_trials_stats)
        if n_included == 0:
            raise ValueError("No trials match the specified criteria")
        
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
    

    

 
    
