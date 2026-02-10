from pathlib import Path
import json
import os
import pandas as pd
import warnings


def _validate_csv_metadata(filepath, mandatory_columns=None, optional_columns=None):
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
            warnings.warn(f"Warning: '{col}' column not found in {filepath}")

    # # Warn about unexpected columns
    # unexpected_columns = set(df.columns) - mandatory_columns - optional_columns
    # if unexpected_columns:
    #     warnings.warn(f"Warning: Unexpected columns in {filepath}: {unexpected_columns}")

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
    #     warnings.warn(f"Warning: Unexpected fields : {unexpected_fields}")

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
    #     warnings.warn(f"Warning: Unexpected fields: {unexpected_fields}")

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


def _validate_metadata_videos_segment_field(segments, from_filepath=None):
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
        warnings.warn(f"Warning: Different number of segments across entries in 'segments' in {from_filepath}: {n}")

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


def validate_metadata_segment_json(filepath):
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
    #     warnings.warn(f"Warning: Unexpected fields in {filepath}: {unexpected_fields}")
    
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
                raise ValueError(f"'segment_index' in 'duplicates' {key} must be an list in {from_filepath}")
        
    return is_valid
