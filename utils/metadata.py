from pathlib import Path
import json
import os
import pandas as pd
import numpy as np


def _validate_csv_metadata(filepath, mandatory_columns=None, optional_columns=None, verbose=True):
    """
    Validate the structure of global metadata for a trial.
    
    Args:
        filepath (str or Path): Path to the CSV metadata file.

    Returns:
        dataframe: The loaded metadata dataframe if valid.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid CSV or is missing mandatory columns.
    
    """
    filepath = Path(filepath)
    is_valid = True
    
    # Check if file exists
    if not filepath.exists():
        is_valid = False
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    # Try to load CSV and validate columns
    try:
        df = pd.read_csv(filepath)
        if 'trial' in df.columns:
            df['trial'] = df['trial'].astype(str)
    except Exception as e:
        is_valid = False
        raise ValueError(f"Error reading {filepath}: {e}")
    
    # Check for mandatory columns and warn about unexpected columns
    if mandatory_columns is None:
        mandatory_columns = set()
    if optional_columns is None:
        optional_columns = set()
    for col in mandatory_columns:
        if col not in df.columns:
            is_valid = False
            print(f"Warning: '{col}' column not found in {filepath}") if verbose else None

    # # Warn about unexpected columns
    # unexpected_columns = set(df.columns) - mandatory_columns - optional_columns
    # if unexpected_columns:
    #     print(f"Warning: Unexpected columns in {filepath}: {unexpected_columns}") if verbose else None

    return df, is_valid


def validate_global_trials_metadata(filepath):
    """
    Validate the structure of global metadata for a trial.
    
    Args:
        filepath (str or Path): Path to the CSV metadata file.

    Returns:
        dataframe: The loaded metadata dataframe if valid.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid CSV or is missing mandatory columns.
    
    """
    return _validate_csv_metadata(filepath, mandatory_columns={'trial', 'ID', 'label', 'trial_type', 'valid_trial'}, optional_columns={'valid_frames'})
        

def validate_global_neurons_metadata(filepath):
    """
    Validate the structure of global metadata for a trial.
    
    Args:
        filepath (str or Path): Path to the CSV metadata file.

    Returns:
        dataframe: The loaded metadata dataframe if valid.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid CSV or is missing mandatory columns.
    
    """
    mandatory_columns={ 'ID', 'coord_x', 'coord_y', 'coord_z'}
    optional_columns={'mean_activation', 'std_activation', 'median_activation', 'min_activation', 'max_activation'}
    return _validate_csv_metadata(filepath, mandatory_columns=mandatory_columns, optional_columns=optional_columns)
        

def validate_metadata_video_dict(metadata):
    """
    Check if a dictionary with metadata has all mandatory fields.
    
    Args:
        metadata (dict): Dictionary with the metadata.
        
    Returns:
        is_valid: if valid.
        
    """
    is_valid = True
    
    # Check for mandatory fields
    mandatory_fields = {'label', 'ID', 'valid_frames', 'sampling_freq'}
    optional_fields = {'duplicates', 'segments'}
    
    missing_fields = mandatory_fields - set(metadata.keys())
    if missing_fields:
        is_valid = False
        raise ValueError(f"Missing mandatory fields: {missing_fields}")
    
    # # Warn about unexpected fields
    # all_allowed_fields = mandatory_fields | optional_fields
    # unexpected_fields = set(metadata.keys()) - all_allowed_fields
    # if unexpected_fields:
    #     print(f"Warning: Unexpected fields : {unexpected_fields}") if verbose else None

    # Check the structure of 'segments' if it exists
    if 'segments' in metadata:
        valid_segment = _validate_metadata_videos_segment_field(metadata['segments'])
        is_valid = is_valid and valid_segment

    # Check the structure of 'duplicates' if it exists
    if 'duplicates' in metadata:
        valid_duplicates = _validate_metadata_videos_duplicates_field(metadata['duplicates'])
        is_valid = is_valid and valid_duplicates
            
    return is_valid


def validate_metadata_video_json(filepath):
    """
    Check if a JSON metadata file exists and has all mandatory fields.
    
    Args:
        filepath (str or Path): Path to the JSON metadata file.
        
    Returns:
        dict: The loaded metadata dictionary if valid.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is missing mandatory fields.
    """
    filepath = Path(filepath)
    is_valid = True
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    # Try to load JSON
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        is_valid = validate_metadata_video_dict(metadata)
    except json.JSONDecodeError as e:
        is_valid = False
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except Exception as e:
        is_valid = False
        raise ValueError(f"Error reading {filepath}: {e}")
                
    return metadata, is_valid


def validate_metadata_per_trial_dict(metadata):
    """
    Check if a dictionary with metadata has all mandatory fields.
    
    Args:
        metadata (dict): Dictionary with the metadata.
        
    Returns:
        is_valid: if valid.
        
    """
    is_valid = True
    
    # Check for mandatory fields
    mandatory_fields = {'trial', 'label', 'ID', 'valid_frames', 'sampling_freq'}
    optional_fields = {'recording', 'trial_type', 
                       'first_label', 'n_peaks', 'segments_n_peaks', 'segments_bad_n', 'segments_avg_duration', 
                       'duplicates', 'segments'}
    
    missing_fields = mandatory_fields - set(metadata.keys())
    if missing_fields:
        is_valid = False
        raise ValueError(f"Missing mandatory fields: {missing_fields}")
    
    # # Warn about unexpected fields
    # all_allowed_fields = mandatory_fields | optional_fields
    # unexpected_fields = set(metadata.keys()) - all_allowed_fields
    # if unexpected_fields:
    #     print(f"Warning: Unexpected fields: {unexpected_fields}") if verbose else None

    # Check the structure of 'segments' if it exists
    if 'segments' in metadata:
        valid_segments = _validate_metadata_videos_segment_field(metadata['segments'])
        is_valid = is_valid and valid_segments
                      
    return is_valid


def validate_metadata_per_trial_json(filepath):
    """
    Check if a JSON metadata file exists and has all mandatory fields.
    
    Args:
        filepath (str or Path): Path to the JSON metadata file.
        
    Returns:
        dict: The loaded metadata dictionary if valid.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is missing mandatory fields.
    """
    filepath = Path(filepath)
    is_valid = True
    
    # Check if file exists
    if not filepath.exists():
        is_valid = False
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    # Try to load JSON
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        is_valid = validate_metadata_per_trial_dict(metadata)
    except json.JSONDecodeError as e:
        is_valid = False
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except Exception as e:
        is_valid = False
        raise ValueError(f"Error reading {filepath}: {e}")
                          
    return metadata, is_valid


def _validate_metadata_videos_segment_field(segments, from_filepath=None, verbose=True):
    is_valid = True
    if not isinstance(segments, dict):
        is_valid = False
        raise ValueError(f"'segments' field must be a dictionary in {from_filepath}")
    if 'frame_start' not in segments or 'frame_end' not in segments:
        is_valid = False
        raise ValueError(f"'segments' must contain 'frame_start' and 'frame_end' keys in {from_filepath}")
    for seg in segments.values():
        if not isinstance(seg, list):
            is_valid = False
            raise ValueError(f"Each entry in 'segments' must be a list in {from_filepath}")
    n = [len(v) for v in segments.values()]
    if len(set(n)) > 1:
        is_valid = False
        print(f"Warning: Different number of segments across entries in 'segments' in {from_filepath}: {n}") if verbose else None

    return is_valid


def _validate_metadata_videos_duplicates_field(duplicates, from_filepath=None):
    is_valid = True
    if not isinstance(duplicates, dict):
        is_valid = False
        raise ValueError(f"'duplicates' field must be a dictionary in {from_filepath}")
    for key, value in duplicates.items():
        if not isinstance(value, dict):
            is_valid = False
            raise ValueError(f"Each entry in 'duplicates' must be a dictionary in {from_filepath}")
        if 'trials' not in value:
                is_valid = False
                raise ValueError(f"Each entry in 'duplicates' must contain 'trials' key in {from_filepath}, missing in {key}")
        if not isinstance(value['trials'], list):
                is_valid = False
                raise ValueError(f"'trials' in 'duplicates' {key} must be a list in {from_filepath}")
        
    return is_valid


def validate_metadata_segment_json(filepath, verbose=True):
    """
    Check if a JSON metadata file exists and has all mandatory fields.
    
    Args:
        filepath (str or Path): Path to the JSON metadata file.
        
    Returns:
        dict: The loaded metadata dictionary if valid.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is missing mandatory fields.
    """
    filepath = Path(filepath)
    is_valid = True
    
    # Check if file exists
    if not filepath.exists():
        is_valid = False
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    # Try to load JSON
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        is_valid = False
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except Exception as e:
        is_valid = False
        raise ValueError(f"Error reading {filepath}: {e}")
    
    # Check for mandatory fields
    mandatory_fields = {'label', 'ID', 'valid_frames', 'sampling_freq'}
    optional_fields = {'duplicates'}
    
    missing_fields = mandatory_fields - set(metadata.keys())
    if missing_fields:
        is_valid = False
        raise ValueError(f"Missing mandatory fields in {filepath}: {missing_fields}")
    
    # # Warn about unexpected fields
    # all_allowed_fields = mandatory_fields | optional_fields
    # unexpected_fields = set(metadata.keys()) - all_allowed_fields
    # if unexpected_fields:
    #     print(f"Warning: Unexpected fields in {filepath}: {unexpected_fields}") if verbose else None
    
    # Check the structure of 'duplicates' if it exists
    if 'duplicates' in metadata:
        valid_duplicates = _validate_metadata_segments_duplicates_field(metadata['duplicates'], from_filepath=filepath)
        is_valid = is_valid and valid_duplicates
            
    return metadata, is_valid
                
def _validate_metadata_segments_duplicates_field(duplicates, from_filepath=None):
    is_valid = True
    if not isinstance(duplicates, dict):
        is_valid = False
        raise ValueError(f"'duplicates' field must be a dictionary in {from_filepath}")
    for key, value in duplicates.items():
        if not isinstance(value, dict):
            is_valid = False
            raise ValueError(f"Each entry in 'duplicates' must be a dictionary in {from_filepath}")
        if 'segment_index' not in value:
                is_valid = False
                raise ValueError(f"Each entry in 'duplicates' must contain 'segment_index' key in {from_filepath}, missing in {key}")
        if not isinstance(value['segment_index'], list):
                is_valid = False
                raise ValueError(f"'segment_index' in 'duplicates' {key} must be a list in {from_filepath}")
        
    return is_valid


def validate_metadata_video_folder(folder_globalmetadata_videos, verbose=True):
    
    if not Path(folder_globalmetadata_videos).exists():
        print(f"Warning: folder_globalmetadata_videos does not exist") if verbose else None
        return False
    
    files = list(Path(folder_globalmetadata_videos).glob(f"*.json"))
    good_files = np.zeros(len(files), dtype=bool)
    for i, fff in enumerate(files):
        try:
            _, good_files[i] = validate_metadata_video_json(fff)
        except Exception as e:
            print(f"Warning: Could not validate metadata video json file {fff}: {e}") if verbose else None
    good_global_meta_videos = good_files.sum()==len(files)
    if not good_global_meta_videos:
        print(f"Warning: {len(files) - good_files.sum()} metadata video json files are invalid in {folder_globalmetadata_videos}") if verbose else None

    return good_global_meta_videos


def validate_metadata_segment_folder(folder_globalmetadata_segments, verbose=True):               
    
    if not Path(folder_globalmetadata_segments).exists():
        print(f"Warning: folder_globalmetadata_segments does not exist") if verbose else None
        return False
    
    files = list(Path(folder_globalmetadata_segments).glob(f"*.json"))
    good_files = np.zeros(len(files), dtype=bool)
    for i, fff in enumerate(files):
        try:
            _, good_files[i] = validate_metadata_segment_json(fff, verbose=verbose)
        except Exception as e:
            print(f"Warning: Could not validate metadata segment json file {fff}: {e}") if verbose else None
    good_global_meta_segments = good_files.sum()==len(files)
    if not good_global_meta_segments:
        print(f"Warning: {len(files) - good_files.sum()} metadata segment json files are invalid in {folder_globalmetadata_segments}") if verbose else None

    return good_global_meta_segments


def validate_metadata_recording(folder_metadata_rec, folder_globalmetadata_videos, trials, n_neurons, verbose=True):

    folder_metadata_rec = Path(folder_metadata_rec)
    rec = folder_metadata_rec.name

    # check if a folder for the recording exists in the metadata folder
    if not folder_metadata_rec.exists():
        print(f"Warning: Path does not exist: {folder_metadata_rec}") if verbose else None
        return False
    
    # check the trials metadata file
    file = os.path.join(folder_metadata_rec,f"meta-trials_{rec}.csv")
    if not os.path.exists(file):
        print(f"Warning: File does not exist: {file}") if verbose else None
        good_trials = False
    else:
        try:
            df, good_trials = validate_global_trials_metadata(file)

            # check that the trials in the metadata file are consistent with the trials in the data files
            if 'trial' in df.columns:
                if not df['trial'].isin(trials).all():
                    trials_diff = set(df['trial'].values) - set(trials)
                    good_trials = False
                    print(f"Warning: {len(trials_diff)} trials in {file} are not found in data files for recording {rec}: {trials_diff}") if verbose else None

            # check that the IDs in the metadata file exist in the IDs in the global metadata videos folder (if configured)
            if Path(folder_globalmetadata_videos).exists():
                if 'ID' in df.columns:
                    the_ids = set(df['ID'].values)
                    for id in the_ids:
                        files = list(Path(folder_globalmetadata_videos).glob(f"*{id}.json"))
                        if len(files)==0:
                            good_trials = False
                            print(f"Warning: No metadata file found for ID {id} in folder_globalmetadata_videos") if verbose else None
                        elif len(files)>1:
                            good_trials = False
                            print(f"Warning: Multiple metadata files found for ID {id} in folder_globalmetadata_videos: {[f.name for f in files]}") if verbose else None
        
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}") if verbose else None
            good_trials = False

    # check the neurons metadata file
    file = os.path.join(folder_metadata_rec,f"meta-neurons_{rec}.csv")
    if not os.path.exists(file):
        print(f"Warning: File does not exist: {file}") if verbose else None
        df, good_neurons = None, False
    else:
        try:
            df, good_neurons = validate_global_neurons_metadata(file)
            if len(df)!=n_neurons:
                good_neurons = False
                print(f"Warning: Number of neurons in {file} does not match the number of neurons in the data files for recording {rec}: {len(df)} vs {n_neurons}") if verbose else None
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}") if verbose else None
            df, good_neurons = None, False

    # print if all fine for the recording
    if good_trials and good_neurons:
        print(f"---> Metadata seems ok for recording {rec}.") if verbose else None

    return good_trials and good_neurons


def check_metadata_integrity(folder_metadata, recording, folder_globalmetadata_videos, folder_globalmetadata_segments, info, verbose=True):
    
    if folder_metadata is None:
        return False, None

    if not Path(folder_metadata).exists():
        print("Warning: The metadata folder was set but it does not exist") if verbose else None
        return False, None
        
    # check all the video metadata files
    good_global_meta_videos = validate_metadata_video_folder(folder_globalmetadata_videos)
           
    # check all the segment metadata files
    good_global_meta_segments = validate_metadata_segment_folder(folder_globalmetadata_segments)
    
    # check the metadata for each recording and its consistency with the data files
    good_metadata_per_recording = {}
    for rec in recording:
        folder_metadata_rec = os.path.join(folder_metadata, rec)
        good_metadata_rec = validate_metadata_recording(folder_metadata_rec, folder_globalmetadata_videos, trials=info[rec]['trials'], n_neurons=info[rec]['n_neurons'], verbose=verbose)
        good_metadata_per_recording[rec] = good_metadata_rec

    good_metadata = good_global_meta_videos and good_global_meta_segments and all(good_metadata_per_recording.values())
    results = {}
    results['good_global_meta_videos'] = good_global_meta_videos
    results['good_global_meta_segments'] = good_global_meta_segments
    results['good_metadata_per_recording'] = good_metadata_per_recording

    return good_metadata, results


def check_metadata_per_trial_integrity(folder_metadata_per_trial, recording, info, verbose=True):

    if folder_metadata_per_trial is None:
        return False, None
 
    if not Path(folder_metadata_per_trial).exists():
        print("Warning: The metadata per trial folder was set but it does not exist, you can create it with create_folders_metadata_per_trial()") if verbose else None
        return False, None
     
        
    # check the metadata for each recording and its consistency with the data files
    good_metadata_per_trial_per_recording = {}
    for rec in recording:
        
        # check if a folder for the recording exists in the metadata folder
        path_to_table = Path(folder_metadata_per_trial) / rec / 'videos'
        if not path_to_table.exists():
            print(f"Warning: Path does not exist: {path_to_table}") if verbose else None
            good_metadata_per_trial_per_recording[rec] = False
        
        else:

            all_fine = True
            
            # check all the video metadata files
            files = list(Path(path_to_table).glob(f"*.json"))
            good_files = np.full(len(files), False)
            trials = []
            for i, fff in enumerate(files):
                try:
                    metadata, good_files[i] = validate_metadata_per_trial_json(fff)
                    trials.append(metadata['trial'])
                except Exception as e:
                    print(f"Warning: Could not validate metadata per trial json file {fff}: {e}") if verbose else None

            if set(trials) != set(info[rec]['trials']):
                trials_diff = set(trials) - set(info[rec]['trials'])
                all_fine = False
                print(f"Warning: Trials in metadata per trial json files do not match the trials in the data files for recording {rec}: {trials_diff}") if verbose else None
            
            if good_files.sum() != len(info[rec]['trials']):
                all_fine = False
                print(f"Warning: Number of valid metadata per trial json files does not match the number of trials in the data files for recording {rec}: {good_files.sum()} vs {len(info[rec]['trials'])}") if verbose else None
            
            # print if all fine for the recording
            if all_fine:
                print(f"---> Metadata per trial seems ok for recording {rec}.") if verbose else None

            good_metadata_per_trial_per_recording[rec] = all_fine

    return all(good_metadata_per_trial_per_recording.values()), good_metadata_per_trial_per_recording



