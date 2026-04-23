# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
from typing import Any

def parse_info_from_recording_name(recording: str, verbose: bool = True) -> dict:
    """Parse recording information from recording folder name.

    Parameters
    ----------
    recording : str
        Recording folder name.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    dict
        Parsed recording information.
    """

    info = {}
    x = recording.split("-")
    start_index = x[0].find('dynamic')
    if start_index != -1:
        end_index = start_index + len('dynamic')
        info["animal_id"] = x[0][end_index:]
    else:
        info["animal_id"] = None
        print("Warning:Substring dynamic not found in the recording folder name. The animal ID could not be set") if verbose else None
    if len(x)>0:
        info["session"] = x[1]
    else:
        info["session"] = None
        print("Warning: Could not parse session from the recording folder name. The session could not be set") if verbose else None
    if len(x)>1:
        info["scan_idx"] = x[2]
    else:
        info["scan_idx"] = None
        print("Warning: Could not parse scan index from the recording folder name. The scan index could not be set") if verbose else None

    return info

def _create_metadata_dict_from_trials_df(df_meta_trials: pd.DataFrame) -> dict[str, Any]:
    """Create a metadata dictionary from a trials DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing trial-level metadata.

    Returns
    -------
    dict
        Metadata dictionary with keys for each column in the DataFrame and values as lists of column values.
    """
    # generate a dictionry with the description of the columns names in the df_meta_trials
    meta_trials_dict = {}
    for col in df_meta_trials.columns:
        if col == "recording":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Recording identifier"
        elif col == "trial":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Trial identifier within the recording. Interger number. They do not match the trials presentation order."
        elif col == "label":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Label indicating the stimulus type presented in the trial"
            meta_trials_dict[col]["values"] = {}
            meta_trials_dict[col]["values"]["NaturalVideo"] = "Video sampled from films or the Sport-1M dataset"
            meta_trials_dict[col]["values"]["NaturalImages"] = "ODD stimuli. Sequence of natural images from ImageNet presented 15 frames interleave with 12-18 frames of gray screen."
            meta_trials_dict[col]["values"]["Gabor"] = "ODD stimuli. Sequence of spatiotemporal drifting Gabor patches (8 possible directions, 3 spacial frequencies, 3 temporal frequencies), lasting 25 frames each"
            meta_trials_dict[col]["values"]["PinkNoise"] = "ODD stimuli. Sequence of directional pink noise with spatial orientation perpendicular to the motion direction, lasting 27 frames each"
            meta_trials_dict[col]["values"]["RandomDots"] = "ODD stimuli. Sequence of random dots kinematogram (8 possible trajectories with 2 possible coherences and 2 velocities), lasting 60 frames each"
            meta_trials_dict[col]["values"]["GaussianDot"] = "ODD stimuli. Sequence of a single Gaussian blob (105 locations, 2 color intensities), lasting 9 frames each"
        elif col == "trial_type":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Phase of the recording in which the trial occurs"
            meta_trials_dict[col]["values"] = {}
            meta_trials_dict[col]["values"]["train"] = "First phase. Trials for trainng. Only NaturalVideo. No video repetition."
            meta_trials_dict[col]["values"]["oracle"] = "Second phase. Trials for training. Only NaturalVideo. Videos repeated 10 times."
            meta_trials_dict[col]["values"]["live_test_main"] = "Third phase. Trials for testing. Only NaturalVideo. Videos repeated 10 times."
            meta_trials_dict[col]["values"]["live_test_bonus"] = "Fourth phase. Trials for testing. Trials from one ODD label. Videos repeated 10 times."
            meta_trials_dict[col]["values"]["final_test_main"] = "Fifth phase. Trials for testing. Only NaturalVideo. Videos repeated 10 times."
            meta_trials_dict[col]["values"]["final_test_bonus"] = "Sixth phase. Trials for testing. Trials from two ODD labels. Videos repeated 10 times."
        elif col == "ID":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Unique video identifier assigned by similarity grouping"
        elif col == "valid_frames":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Number of valid frames in the trial (minimum between video and response valid frames)"
        elif col == "valid_frames_video":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Number of valid frames in the video for this trial (not NaN)"
        elif col == "valid_frames_response":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Number of valid frames in the neural response data for this trial (not NaN)"
        elif col == "valid_trial":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Boolean indicating whether the trial is valid (e.g. no bad segments)"
        elif col == "valid_response":
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = "Boolean indicating whether the neural response is valid (e.g., not all zero or NaN)"
        else:
            meta_trials_dict[col] = {}
            meta_trials_dict[col]["description"] = f"Description for {col}"  # Replace with actual descriptions if available

    return meta_trials_dict

def _validate_dataframe(
    df: pd.DataFrame,
    mandatory_columns: set[str] | list[str] | None = None,
    optional_columns: set[str] | list[str] | None = None,
    nan_accepted: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Validate CSV metadata file structure.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing trial-level metadata.
    mandatory_columns : set or list or None, optional
        Required columns.
    optional_columns : set or list or None, optional
        Allowed optional columns.
    nan_accepted : bool, default=False
        If ``True``, allow NaN values in the DataFrame for mandatory columns.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        Loaded dataframe and validity flag.
    """
    is_valid = True

    # Check if data is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    # Check for mandatory columns and warn about unexpected columns
    if mandatory_columns is None:
        mandatory_columns = set()
    for col in list(mandatory_columns):
        if col not in df.columns:
            is_valid = False
            print(f"Warning: '{col}' column not found") if verbose else None
        elif not nan_accepted and df[col].isna().any():
            is_valid = False
            print(f"Warning: NaN values found in mandatory column '{col}'") if verbose else None

    # Warn about unexpected columns
    if optional_columns is not None:
        unexpected_columns = set(df.columns) - set(mandatory_columns) - set(optional_columns)
        if unexpected_columns:
            print(f"Warning: Unexpected columns: {unexpected_columns}") if verbose else None

    return is_valid



def _validate_csv(
    filepath: str | Path,
    mandatory_columns: set[str] | list[str] | None = None,
    optional_columns: set[str] | list[str] | None = None,
    nan_accepted: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Validate CSV metadata file structure.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to CSV metadata file.
    mandatory_columns : set or list or None, optional
        Required columns.
    optional_columns : set or list or None, optional
        Allowed optional columns.
    nan_accepted : bool, default=False
        If ``True``, allow NaN values in the DataFrame for mandatory columns.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[pandas.DataFrame, bool]
        Loaded dataframe and validity flag.
    """
    filepath = Path(filepath)
    is_valid = True

    # Check if file exists
    if not filepath.exists():
        print(f"Metadata file not found: {filepath}") if verbose else None
        return None, False

    # Try to load CSV and validate columns
    try:
        df = pd.read_csv(filepath)
        if "trial" in df.columns:
            df["trial"] = df["trial"].astype(str)
    except Exception as e:
        print(f"Error reading {filepath}: {e}") if verbose else None
        return None, False

    # Check for mandatory columns and warn about unexpected columns
    is_valid = _validate_dataframe(df, 
                                   mandatory_columns=mandatory_columns, 
                                   optional_columns=optional_columns, 
                                   nan_accepted=nan_accepted, 
                                   verbose=verbose,
                                   )
    
    return df, is_valid

def _validate_dict(metadata: dict[str, Any], 
                   mandatory_fields: set[str] | list[str] | None = None,
                   optional_fields: set[str] | list[str] | None = None,
                   verbose: bool = True) -> bool:
    """Validate metadata dictionary.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary.
    mandatory_fields : set or list
        Required fields.
    optional_fields : set or list or None, optional
        Allowed optional fields.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    bool
        ``True`` when dictionary passes all checks.
    """
    is_valid = True

    if mandatory_fields is not None:
        missing_fields = set(mandatory_fields) - set(metadata.keys())
        if missing_fields:
            is_valid = False
            print(f"Missing mandatory fields: {missing_fields}") if verbose else None

    # Warn about unexpected fields
    if optional_fields is not None:
        unexpected_fields = set(metadata.keys()) - set(mandatory_fields) - set(optional_fields)
        if unexpected_fields:
            print(f"Warning: Unexpected fields: {unexpected_fields}") if verbose else None


    return is_valid

def _validate_json(
    filepath: str | Path,
    mandatory_fields: set[str] | list[str] | None = None,
    optional_fields: set[str] | list[str] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Load and validate one basic metadata JSON file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to JSON metadata file.
    mandatory_fields : set or list
        Required fields.
    optional_fields : set or list or None, optional
        Allowed optional fields.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[dict, bool]
        Metadata dictionary and validity flag.
    """
    filepath = Path(filepath)
    is_valid = True

    # Check if file exists
    if not filepath.exists():
        print(f"Metadata file not found: {filepath}") if verbose else None
        return None, False

    # Try to load JSON
    try:
        with open(filepath, "r") as f:
            metadata = json.load(f)
        is_valid = _validate_dict(metadata, 
                                  mandatory_fields=mandatory_fields, 
                                  optional_fields=optional_fields, 
                                  verbose=verbose)
    except json.JSONDecodeError as e:
        is_valid = False
        metadata = None
        print(f"Invalid JSON in {filepath}: {e}") if verbose else None
    except Exception as e:
        is_valid = False
        metadata = None
        print(f"Error reading {filepath}: {e}") if verbose else None

    return metadata, is_valid

def _json_to_dataframe(
        json_folder: str | Path, 
        file_pattern: str = "*.json", 
        include_file_as_column: bool = True,
        verbose: bool = True) -> None:
    """Convert trial-level metadata from JSON files to a single DataFrame.

    Parameters
    ----------
    json_folder : str or pathlib.Path
        Folder containing trial metadata JSON files.
    file_pattern : str, optional
        Pattern to match JSON files, by default "*.json"
    verbose : bool, optional
        Whether to print warnings, by default True.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing trial-level metadata.
    """
    json_folder = Path(json_folder)
    
    if not json_folder.is_dir():
        print(f"Warning: Trials metadata folder does not exist: {json_folder}") if verbose else None
        return None, False
    
    files = list(Path(json_folder).glob(file_pattern))
    if len(files) == 0:
        print(f"Warning: No JSON files found in {json_folder}") if verbose else None
        return None, False
    
    all_rows = []
    loaded_files = np.zeros(len(files), dtype=bool)
    for i, fff in enumerate(files):
        try:
            with open(fff, "r") as f:
                meta = json.load(f)
            loaded_files[i] = True
            meta_filtered = {
                                k: v
                                for k, v in meta.items()
                                if isinstance(v, (str, int, float, bool, type(None)))
                            }
            if include_file_as_column:
                meta_filtered["file"] = fff.stem
            all_rows.append(meta_filtered)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {fff}: {e}") if verbose else None
        except Exception as e:
            print(f"Error reading {fff}: {e}") if verbose else None
        
    trials_df = pd.DataFrame(all_rows)
    if 'trial' in trials_df.columns:
        trials_df["trial"] = trials_df["trial"].astype(str)
    all_loaded = loaded_files.sum() == len(files)

    return trials_df, all_loaded


def _validate_metadata_videos_segment_field(
    segments: dict[str, Any],
    from_filepath: str | Path | None = None,
    verbose: bool = True,
) -> bool:
    """Validate ``segments`` field structure in video metadata.

    Parameters
    ----------
    segments : dict
        Segment field dictionary.
    from_filepath : str or pathlib.Path or None, optional
        Optional file path used for error messages.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    bool
        ``True`` when field is structurally valid.
    """
    is_valid = True
    if not isinstance(segments, dict):
        is_valid = False
        raise ValueError(f"'segments' field must be a dictionary in {from_filepath}")
    if "frame_start" not in segments or "frame_end" not in segments:
        is_valid = False
        raise ValueError(
            f"'segments' must contain 'frame_start' and 'frame_end' keys in {from_filepath}"
        )
    for seg in segments.values():
        if not isinstance(seg, list):
            is_valid = False
            raise ValueError(
                f"Each entry in 'segments' must be a list in {from_filepath}"
            )
    n = [len(v) for v in segments.values()]
    if len(set(n)) > 1:
        is_valid = False
        (
            print(
                f"Warning: Different number of segments across entries in 'segments' in {from_filepath}: {n}"
            )
            if verbose
            else None
        )

    return is_valid


def _validate_metadata_videos_duplicates_field(
    duplicates: dict[str, Any], from_filepath: str | Path | None = None
) -> bool:
    """Validate ``duplicates`` field structure for video metadata.

    Parameters
    ----------
    duplicates : dict
        Duplicates mapping from recording to trials.
    from_filepath : str or pathlib.Path or None, optional
        Optional file path used for error messages.

    Returns
    -------
    bool
        ``True`` when field is structurally valid.
    """
    is_valid = True
    if not isinstance(duplicates, dict):
        is_valid = False
        raise ValueError(f"'duplicates' field must be a dictionary in {from_filepath}")
    for key, value in duplicates.items():
        if not isinstance(value, dict):
            is_valid = False
            raise ValueError(
                f"Each entry in 'duplicates' must be a dictionary in {from_filepath}"
            )
        if "trials" not in value:
            is_valid = False
            raise ValueError(
                f"Each entry in 'duplicates' must contain 'trials' key in {from_filepath}, missing in {key}"
            )
        if not isinstance(value["trials"], list):
            is_valid = False
            raise ValueError(
                f"'trials' in 'duplicates' {key} must be a list in {from_filepath}"
            )

    return is_valid


def _validate_metadata_segments_duplicates_field(
    duplicates: dict[str, Any], from_filepath: str | Path | None = None
) -> bool:
    """Validate ``duplicates`` field structure for segment metadata.

    Parameters
    ----------
    duplicates : dict
        Duplicates mapping from video IDs to segment indices.
    from_filepath : str or pathlib.Path or None, optional
        Optional file path used for error messages.

    Returns
    -------
    bool
        ``True`` when field is structurally valid.
    """
    is_valid = True
    if not isinstance(duplicates, dict):
        is_valid = False
        raise ValueError(f"'duplicates' field must be a dictionary in {from_filepath}")
    for key, value in duplicates.items():
        if not isinstance(value, dict):
            is_valid = False
            raise ValueError(
                f"Each entry in 'duplicates' must be a dictionary in {from_filepath}"
            )
        if "segment_index" not in value:
            is_valid = False
            raise ValueError(
                f"Each entry in 'duplicates' must contain 'segment_index' key in {from_filepath}, missing in {key}"
            )
        if not isinstance(value["segment_index"], list):
            is_valid = False
            raise ValueError(
                f"'segment_index' in 'duplicates' {key} must be a list in {from_filepath}"
            )

    return is_valid


def _validate_metadata_video_json(filepath: str | Path, verbose: bool = True) -> tuple[dict[str, Any], bool]:
    """Load and validate one trial metadata JSON file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to JSON metadata file.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[dict, bool]
        Metadata dictionary and validity flag.
    """
    filepath = Path(filepath)
    is_valid = True

    metadata, is_valid = _validate_json(filepath,
                                    mandatory_fields={"label", "ID", "valid_frames", "sampling_freq"},
                                    optional_fields={"duplicates", "segments"},
                                    verbose=verbose,)

    if metadata is not None and is_valid:
        # Check the structure of 'segments' if it exists
        if "segments" in metadata:
            valid_segment = _validate_metadata_videos_segment_field(metadata["segments"], from_filepath=filepath, verbose=verbose)
            is_valid = is_valid and valid_segment

        # Check the structure of 'duplicates' if it exists
        if "duplicates" in metadata:
            valid_duplicates = _validate_metadata_videos_duplicates_field(metadata["duplicates"], from_filepath=filepath)
            is_valid = is_valid and valid_duplicates

        # Check the file name matches the ID and label in the metadata
        if metadata["ID"] not in filepath.stem or metadata["label"] not in filepath.stem:
            is_valid = False
            if verbose:
                print(f"Warning: File name {filepath.name} does not match ID {metadata['ID']} or label {metadata['label']} in metadata") if verbose else None

    return metadata, is_valid

def _validate_metadata_video_dict(metadata: dict[str, Any], verbose: bool = True) -> tuple[dict[str, Any], bool]:
    """Validate a video metadata dictionary.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[dict, bool]
        Metadata dictionary and validity flag.
    """

    is_valid = _validate_dict(metadata,
                                    mandatory_fields={"label", "ID", "valid_frames", "sampling_freq"},
                                    optional_fields={"duplicates", "segments"},
                                    verbose=verbose,)
    
    if is_valid:
        # Check the structure of 'segments' if it exists
        if "segments" in metadata:
            valid_segment = _validate_metadata_videos_segment_field(metadata["segments"], from_filepath=None, verbose=verbose)
            is_valid = is_valid and valid_segment

        # Check the structure of 'duplicates' if it exists
        if "duplicates" in metadata:
            valid_duplicates = _validate_metadata_videos_duplicates_field(metadata["duplicates"], from_filepath=None)
            is_valid = is_valid and valid_duplicates

    return is_valid


def _validate_metadata_segment_json(
    filepath: str | Path, verbose: bool = True
) -> tuple[dict[str, Any], bool]:
    """Load and validate one segment metadata JSON file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to JSON metadata file.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[dict, bool]
        Metadata dictionary and validity flag.
    """
    filepath = Path(filepath)
    is_valid = True

    metadata, is_valid = _validate_json(filepath,
                                    mandatory_fields={"label", "ID", "valid_frames", "sampling_freq"},
                                    optional_fields={"duplicates"},
                                    verbose=verbose,)


    if metadata is not None and is_valid:
        # Check the structure of 'duplicates' if it exists
        if "duplicates" in metadata:
            valid_duplicates = _validate_metadata_segments_duplicates_field(
                metadata["duplicates"], from_filepath=filepath
            )
            is_valid = is_valid and valid_duplicates

        # Check the file name matches the ID and label in the metadata
        if metadata["ID"] not in filepath.stem or metadata["label"] not in filepath.stem:
            is_valid = False
            if verbose:
                print(f"Warning: File name {filepath.name} does not match ID {metadata['ID']} or label {metadata['label']} in metadata") if verbose else None

    return metadata, is_valid


def validate_metadata_video_folder(
    folder_globalmetadata_videos: str | Path, verbose: bool = True
) -> bool:
    """Validate all video metadata JSON files in a folder.

    Parameters
    ----------
    folder_globalmetadata_videos : str or pathlib.Path
        Folder with global video metadata JSON files.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    bool
        ``True`` when all files are valid.
    """

    if not Path(folder_globalmetadata_videos).exists():
        (
            print(f"Warning: folder_globalmetadata_videos does not exist")
            if verbose
            else None
        )
        return False

    files = list(Path(folder_globalmetadata_videos).glob(f"*.json"))
    good_files = np.zeros(len(files), dtype=bool)
    for i, fff in enumerate(files):
        try:
            _, good_files[i] = _validate_metadata_video_json(fff)
        except Exception as e:
            (
                print(
                    f"Warning: Could not validate metadata video json file {fff}: {e}"
                )
                if verbose
                else None
            )
    good_global_meta_videos = good_files.sum() == len(files)
    if not good_global_meta_videos:
        (
            print(
                f"Warning: {len(files) - good_files.sum()} metadata video json files are invalid in {folder_globalmetadata_videos}"
            )
            if verbose
            else None
        )

    return good_global_meta_videos


def validate_metadata_segment_folder(
    folder_globalmetadata_segments: str | Path, verbose: bool = True
) -> bool:
    """Validate all segment metadata JSON files in a folder.

    Parameters
    ----------
    folder_globalmetadata_segments : str or pathlib.Path
        Folder with global segment metadata JSON files.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    bool
        ``True`` when all files are valid.
    """

    if not Path(folder_globalmetadata_segments).exists():
        (
            print(f"Warning: folder_globalmetadata_segments does not exist")
            if verbose
            else None
        )
        return False

    files = list(Path(folder_globalmetadata_segments).glob(f"*.json"))
    good_files = np.zeros(len(files), dtype=bool)
    for i, fff in enumerate(files):
        try:
            _, good_files[i] = _validate_metadata_segment_json(fff, verbose=verbose)
        except Exception as e:
            (
                print(
                    f"Warning: Could not validate metadata segment json file {fff}: {e}"
                )
                if verbose
                else None
            )
    good_global_meta_segments = good_files.sum() == len(files)
    if not good_global_meta_segments:
        (
            print(
                f"Warning: {len(files) - good_files.sum()} metadata segment json files are invalid in {folder_globalmetadata_segments}"
            )
            if verbose
            else None
        )

    return good_global_meta_segments


def validate_metadata_recording(
    folder_metadata_rec: str | Path,
    folder_globalmetadata_videos: str | Path,
    trials: list[str],
    n_neurons: int,
    trials_metadata_file_type: str = "csv",
    trials_metadata_subfolder: str = "trials",
    neurons_metadata_subfolder: str = "neurons",
    verbose: bool = True,
) -> bool:
    """Validate recording-level metadata and consistency with data inventory.

    Parameters
    ----------
    folder_metadata_rec : str or pathlib.Path
        Recording metadata folder.
    folder_globalmetadata_videos : str or pathlib.Path
        Folder with global video metadata.
    trials : list[str]
        Trial names detected in data files.
    n_neurons : int
        Expected neuron count from response data.
    trials_metadata_file_type : str (csv or json), default="csv"
        File type for trials metadata files.
    trials_metadata_subfolder : str, default="trials"
        Subfolder name for trials metadata.
    neurons_metadata_subfolder : str, default="neurons"
        Subfolder name for neurons metadata.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    bool
        ``True`` when trial and neuron metadata are valid.
    """

    folder_metadata_rec = Path(folder_metadata_rec)
    rec = folder_metadata_rec.name

    if trials_metadata_file_type not in {"csv", "json"}:
        raise ValueError(f"Invalid trials_metadata_file_type: {trials_metadata_file_type}, expected 'csv' or 'json'")

    # check if a folder for the recording exists in the metadata folder
    if not folder_metadata_rec.is_dir():
        print(f"Warning: Path does not exist: {folder_metadata_rec}") if verbose else None
        return False

    # check the trials metadata
    folder_meta_trials = folder_metadata_rec / trials_metadata_subfolder
    mandatory_columns = {"trial", "ID", "label", "trial_type", "valid_trial"}
    optional_columns = {"recording",
                        "first_label",
                        "sampling_freq",
                        "valid_frames",
                        "valid_frames_video",
                        "valid_frames_response",
                        "valid_response",
                        "n_peaks",
                        "segments_n_peaks",
                        "segments_bad_n",
                        "segments_avg_duration"}    
    if not folder_meta_trials.is_dir():
        print(f"Warning: Trials metadata folder does not exist: {folder_meta_trials}") if verbose else None
        good_trials = False
    else:
        # load trials metadata as dataframe and validate it
        if trials_metadata_file_type == "csv":
            file = folder_meta_trials / f"meta-trials_{rec}.csv"
            trials_df, good_trials = _validate_csv(
                                                    file,
                                                    mandatory_columns=mandatory_columns,
                                                    optional_columns=optional_columns,
                                                    nan_accepted=False,
                                                    verbose=verbose
                                                    )
            
        elif trials_metadata_file_type == "json":
            trials_df, all_loaded = _json_to_dataframe(folder_meta_trials, 
                                                       file_pattern="*.json", 
                                                       include_file_as_column=False, 
                                                       verbose=verbose)
            if all_loaded:
                good_trials = _validate_dataframe(trials_df, 
                                                  mandatory_columns=mandatory_columns, 
                                                  optional_columns=optional_columns, 
                                                  nan_accepted=False, 
                                                  verbose=verbose)
            else:
                good_trials = False

        # check that the trials in the metadata file are consistent with the trials in the data files
        if trials_df is not None:
            if "trial" in trials_df.columns:
                if not trials_df["trial"].isin(trials).all():
                    trials_diff = set(trials_df["trial"].values) - set(trials)
                    good_trials = False
                    print(f"Warning: {len(trials_diff)} trials in {file} are not found in data files for recording {rec}: {trials_diff}") if verbose else None

        # check that the IDs in the metadata file exist in the IDs in the global metadata videos folder (if configured)
        if trials_df is not None:
            if Path(folder_globalmetadata_videos).exists():
                if "ID" in trials_df.columns:
                    the_ids = set(trials_df["ID"].values)
                    for id in the_ids:
                        files = list(Path(folder_globalmetadata_videos).glob(f"*{id}.json"))
                        if len(files) == 0:
                            good_trials = False
                            (
                                print(
                                    f"Warning: No metadata file found for ID {id} in folder_globalmetadata_videos"
                                )
                                if verbose
                                else None
                            )
                        elif len(files) > 1:
                            good_trials = False
                            (
                                print(
                                    f"Warning: Multiple metadata files found for ID {id} in folder_globalmetadata_videos: {[f.name for f in files]}"
                                )
                                if verbose
                                else None
                            )

    # check the neurons metadata file
    folder_meta_neurons = folder_metadata_rec / neurons_metadata_subfolder
    file = os.path.join(folder_meta_neurons, f"meta-neurons_{rec}.csv")
    mandatory_columns = {"ID", "coord_x", "coord_y", "coord_z"}
    optional_columns = {"mean_activation",
                        "std_activation",
                        "median_activation",
                        "min_activation",
                        "max_activation",
                        }
    df, good_neurons = _validate_csv(file, 
                                     mandatory_columns=mandatory_columns, 
                                     optional_columns=optional_columns, 
                                     nan_accepted=False,
                                     verbose=verbose)
    if df is not None:
        if len(df) != n_neurons:
            good_neurons = False
            print(f"Warning: Number of neurons in {file} does not match the number of neurons in the data files for recording {rec}: {len(df)} vs {n_neurons}") if verbose else None
                  

    # check the basic metadata file
    file = os.path.join(folder_metadata_rec, f"meta-basic_{rec}.json")
    mandatory_fields = {"animal_id", "session", "scan_idx", "sampling_freq","n_neurons", "n_trials","samples_per_trial"}
    basic_meta, good_basic = _validate_json(file, mandatory_fields=mandatory_fields, verbose=verbose)
        
    # print if all fine for the recording
    if good_trials and good_neurons and good_basic:
        print(f" > Metadata seems ok for recording {rec}.") if verbose else None

    return good_trials and good_neurons and good_basic


def check_metadata_integrity(
    folder_metadata: str | Path | None,
    recording: list[str],
    folder_globalmetadata_videos: str | Path,
    folder_globalmetadata_segments: str | Path,
    info: dict[str, Any],
    trials_metadata_file_type: str = "csv",
    trials_metadata_subfolder: str = "trials",
    neurons_metadata_subfolder: str = "neurons",
    verbose: bool = True,
) -> tuple[bool, dict[str, Any] | None]:
    """Run full metadata integrity checks for a dataset.

    Parameters
    ----------
    folder_metadata : str or pathlib.Path or None
        Root metadata folder.
    recording : list[str]
        Recordings to validate.
    folder_globalmetadata_videos : str or pathlib.Path
        Global video metadata folder.
    folder_globalmetadata_segments : str or pathlib.Path
        Global segment metadata folder.
    info : dict
        Per-recording data summary from integrity checks.
    trials_metadata_file_type : str, default="csv"
        File type for trials metadata files ('csv' or 'json').
    trials_metadata_subfolder : str, default="trials"
        Subfolder name for trials metadata.
    neurons_metadata_subfolder : str, default="neurons"
        Subfolder name for neurons metadata.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[bool, dict or None]
        Global validity flag and detailed result dictionary.
    """

    if folder_metadata is None:
        return False, None

    if not Path(folder_metadata).is_dir():
        (
            print("Warning: The metadata folder was set but it does not exist")
            if verbose
            else None
        )
        return False, None

    # check all the video metadata files
    good_global_meta_videos = validate_metadata_video_folder(
        folder_globalmetadata_videos
    )

    # check all the segment metadata files
    good_global_meta_segments = validate_metadata_segment_folder(
        folder_globalmetadata_segments
    )

    # check the metadata for each recording and its consistency with the data files
    good_metadata_per_recording = {}
    for rec in recording:
        folder_metadata_rec = Path(folder_metadata, rec)
        good_metadata_rec = validate_metadata_recording(
            folder_metadata_rec,
            folder_globalmetadata_videos,
            trials=info[rec]["trials"],
            n_neurons=info[rec]["n_neurons"],
            trials_metadata_file_type=trials_metadata_file_type,
            trials_metadata_subfolder=trials_metadata_subfolder,
            neurons_metadata_subfolder=neurons_metadata_subfolder,
            verbose=verbose,
        )
        good_metadata_per_recording[rec] = good_metadata_rec

    good_metadata = (
        good_global_meta_videos
        and good_global_meta_segments
        and all(good_metadata_per_recording.values())
    )
    results = {}
    results["good_global_meta_videos"] = good_global_meta_videos
    results["good_global_meta_segments"] = good_global_meta_segments
    results["good_metadata_per_recording"] = good_metadata_per_recording

    return good_metadata, results

