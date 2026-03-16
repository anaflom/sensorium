# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

from tabnanny import verbose
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import json
import operator


from utils.videos import Video, VideoID, VideoSegment, VideoSegmentID
from utils.responses import Responses
from utils.neurons import Neurons, NeuronsData
from utils.behavioral import Gaze, Pupil, Locomotion
from utils.data_handling import (
    load_all_data,
    load_metadata_from_id,
    get_file_with_pattern,
    save_json,
    check_data_integrity,
    check_meta_neurons_integrity,
    check_meta_trials_integrity,
)
from utils.metadata import (
    parse_info_from_recording_name,
    _json_to_dataframe,
    _validate_metadata_video_dict,
    check_metadata_integrity,
)
from utils.videos_duplicates import (
    compute_dissimilarity_video_list,
    compare_with_idvideos,
    find_equal_sets_scipy,
    generate_new_id,
)


def combine_data(
    behavioral_list: list, weights: np.ndarray | None = None
) -> np.ndarray:
    """Combine multiple behavior objects by weighted average.

    Parameters
    ----------
    behavioral_list : list
        Objects with a ``data`` attribute of identical shape.
    weights : array-like or None, optional
        Optional weights. If ``None``, uniform weights are used.

    Returns
    -------
    numpy.ndarray
        Weighted average array.
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


def print_title(s: str, verbose: bool):
    """Print a section title when verbosity is enabled."""
    print(f"\n{s:-<100}") if verbose else None



def filter_dataframe(df,
    query: str | None = None,
    **conditions,
) -> pd.DataFrame:
    """Filter the trials table by one or more conditions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to filter.
    query : str | None, default=None
        A query string to filter the trials using pandas.DataFrame.query().
    conditions : keyword arguments
        Conditions to filter the trials. Keys are column names and values are the values to filter by. Values can be a single value or a list of values.
        
    Returns
    -------
    pandas.DataFrame
        Filtered trial rows.

    Examples
    --------
    filter_dataframe(df, trial_type="A")
    filter_dataframe(df, trial_type=["A", "B"])
    filter_dataframe(df, score__gt=5)
    filter_dataframe(df, score__ge=5, score__lt=10)
    filter_dataframe(df, recording__contains="mouse")
    filter_dataframe(df, score=lambda s: s % 2 == 0)
    filter_dataframe(df, query="score > 5 and trial_type == 'A'")
    """

    mask = pd.Series(True, index=df.index)

    # operator mapping
    ops = {
        "gt": operator.gt,
        "lt": operator.lt,
        "ge": operator.ge,
        "le": operator.le,
        "ne": operator.ne,
        "eq": operator.eq,
    }
        
    for key, value in conditions.items():

        if "__" in key:
            column, op_name = key.split("__", 1)
        else:
            column, op_name = key, "eq"

        # Validate column
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in trials_df")

        series = df[column]

        # Callable support
        if callable(value):
            callable_result = value(series)
            if isinstance(callable_result, pd.Series):
                if not callable_result.index.equals(df.index):
                    raise ValueError(
                        f"Callable filter for column '{column}' must return a Series aligned with trials_df index"
                    )
                mask &= callable_result.fillna(False).astype(bool)
            else:
                callable_array = np.asarray(callable_result)
                if callable_array.shape[0] != len(df):
                    raise ValueError(
                        f"Callable filter for column '{column}' must return one boolean value per row"
                    )
                mask &= pd.Series(callable_array, index=df.index).fillna(False).astype(bool)
            continue

        # List support
        if isinstance(value, (list, tuple, set)) and op_name == "eq":
            mask &= series.isin(value)
            continue

        # String operators
        if op_name in ["contains", "startswith", "endswith"]:
            if not pd.api.types.is_string_dtype(series):
                raise TypeError(f"Column '{column}' is not string type")
            mask &= getattr(series.str, op_name)(value, na=False)
            continue

        # Comparison operators
        if op_name in ops:
            mask &= ops[op_name](series, value)
        else:
            raise ValueError(f"Unsupported operator '__{op_name}'")

    filtered_df = df.loc[mask]

    # Query support (applied after kwargs filtering)
    if query is not None:
        filtered_df = filtered_df.query(query)

    return filtered_df
    

class DataSet:

    def __init__(
        self,
        folder_data: str | Path,
        folder_metadata: str | Path | None = None,
        recording: list[str] | str | None = None,
        check_data: bool = True,
        check_metadata: bool = True,
        check: bool = True,
        trials_metadata_file_type: str = "csv",
        trials_metadata_subfolder: str = "trials",
        neurons_metadata_subfolder: str = "neurons",
        verbose: bool = True,
    ):
        """Initialize dataset paths, integrity checks, and cached metadata.

        Parameters
        ----------
        folder_data : str or pathlib.Path
            Root data folder containing recording directories.
        folder_metadata : str or pathlib.Path or None, optional
            Root metadata folder.
        recording : list[str] or str or None, optional
            Recording names to include; if ``None``, all subfolders are used.
        check_data : bool, default=True
            If ``True``, perform data integrity checks.
        check_metadata : bool, default=True
            If ``True``, perform metadata integrity checks.
        check : bool, default=True
            If ``False``, skip all integrity checks. Overrides other check flags.
        trials_metadata_file_type : str, default="csv"
            File type for trials metadata files ('csv' or 'json').
        trials_metadata_subfolder : str, default="trials"
            Subfolder name for trials metadata files.
        verbose : bool, default=True
            If ``True``, print progress messages.
        """

        print_title("Initializing DataSet ", verbose)

        if check==False:
            check_data = False
            check_metadata = False

        if trials_metadata_file_type not in {"csv", "json"}:
            raise ValueError(f"Invalid trials_metadata_file_type: {trials_metadata_file_type}, expected 'csv' or 'json'")
        self._trials_metadata_file_type = trials_metadata_file_type
        
        # set data folders and check they exist
        self.folder_data = folder_data
        if not os.path.exists(folder_data):
            raise ValueError(f"Path does not exist: {folder_data}")
        if recording is None:
            recording = [p.name for p in Path(folder_data).iterdir() if p.is_dir()]
        self.recording = recording
        
        # Check the data and store some info about it
        self.add_data_info()
        if check_data:
            self.check_data(verbose=verbose)
        else:
            print_title("Data integrity check skipped ", verbose)
            print(" > If you want to check it, set check and check_data to True when initializing the DataSet.") if verbose else None
            print(" > Data is assumed to be valid. Misbehavior may occur if the data is corrupted.") if verbose else None
        
        # set metadata folders
        self.folder_metadata = folder_metadata
        self._trials_metadata_subfolder = trials_metadata_subfolder
        self._neurons_metadata_subfolder = neurons_metadata_subfolder
        self.set_globalmetadata_folders()

        # check the metadata folders and files and their consistency with the data files
        if check_metadata:
            self.check_metadata(verbose=verbose)
        else:
            print_title("Metadata integrity check skipped ", verbose)
            print(" > If you want to check it, set check and check_metadata to True when initializing the DataSet.") if verbose else None
            print(" > Metadata will be assumed to be valid for existing folders and invalid for missing folders.") if verbose else None
            self._check_metadata_folder_structure(verbose=False)
        
        # load neurons data for all recordings and store it in the info variable
        self.load_neurons(verbose=verbose)

        # load trial types for all recordings and store it in the info variable
        self.load_trial_types()

        # create a table with basic info about the trials
        self.trials_df = self.get_trials_metadata_basic(verbose=verbose)

    def __str__(self):
        """Return a human-readable summary of the dataset.

        Returns
        -------
        str
            Dataset summary string.
        """
        s = ""
        s = s + f"The dataset contains {len(self.recording)} recordings:\n"
        for rec in self.recording:
            s = (
                s
                + f"  - {rec} with {self.info[rec]['n_neurons']} neurons recorded, {self.info[rec]['n_trials']} trials, and {self.info[rec]['samples_per_trial']} samples per trial\n"
            )
        if hasattr(self, "good_metadata") and self.good_metadata:
            s = s + "The recording has consistent metadata\n"
        if hasattr(self, "good_metadata_per_trial") and self.good_metadata_per_trial:
            s = s + "The recording has consistent metadata per trial\n"

        return s
    
    def _check_metadata_folder_structure(self, verbose: bool = True):

        if self.folder_metadata is not None and self.folder_metadata.is_dir():
            # check the metadata per recording folders and files structure
            good_structure_per_recording = {}
            for rec in self.recording:
                if Path(self.folder_metadata, rec).is_dir():
                    exists_trials_folder = Path(self.folder_metadata, rec, self._trials_metadata_subfolder).is_dir()
                    exists_neurons_folder = Path(self.folder_metadata, rec, self._neurons_metadata_subfolder).is_dir()
                    exists_basic_meta_file = Path(self.folder_metadata, rec, f'meta-basic_{rec}.json').is_file()
                    if exists_trials_folder and exists_neurons_folder and exists_basic_meta_file:
                        good_structure_per_recording[rec] = True
                    else:
                        good_structure_per_recording[rec] = False
                else:
                    good_structure_per_recording[rec] = False
            self._good_metadata_per_recording = good_structure_per_recording
            if all(good_structure_per_recording.values()):
                print("   - All metadata directories for recordings have a good structure.") if verbose else None
            else:
                print("   - Some metadata directories for recordings are missing or do not have a good structure.") if verbose else None
                print("     Metadata for recordings is assumed to be valid for the directories having a good structure.") if verbose else None
            # check the metadata global for videos folder structure
            if (self.folder_globalmetadata_videos is not None) and (self.folder_globalmetadata_videos.is_dir()):
                print("   - The metadata folders for global metadata videos exist.") if verbose else None
                print("     Global metadata for videos is assumed to be valid.") if verbose else None
                self._good_global_meta_videos = True
            else:
                print("   - The metadata folders for global metadata videos are not set or do not exist.") if verbose else None
                print("     Global metadata for videos is invalid.") if verbose else None
                self._good_global_meta_videos = False
            # check the metadata global for segments folder structure
            if (self.folder_globalmetadata_segments is not None) and (self.folder_globalmetadata_segments.is_dir()):
                print("   - The metadata folders for global metadata segments exist.") if verbose else None
                print("     Global metadata for segments is assumed to be valid.") if verbose else None
                self._good_global_meta_segments = True
            else:
                print("   - The metadata folders for global metadata segments are not set or do not exist.") if verbose else None
                print("     Global metadata for segments is invalid.") if verbose else None
                self._good_global_meta_segments = False
        else:
            print("   - The metadata folder is not set or does not exist.") if verbose else None
            print("     Metadata for recordings, global metadata for videos and global metadata for segments are all invalid.") if verbose else None
            self._good_metadata_per_recording = {rec: False for rec in self.recording}
            self._good_global_meta_videos = False
            self._good_global_meta_segments = False


    def load_neurons(
        self, recording: str | list[str] | None = None, verbose: bool = True
    ) -> None:
        """Load neuron metadata for selected recordings.

        Parameters
        ----------
        recording : str or list[str] or None, optional
            Recordings to process. If ``None``, process all recordings.
        verbose : bool, default=True
            If ``True``, print progress messages.
        """

        print_title("Loading neurons metadata ", verbose)

        if recording is None:
            recording = self.recording
        elif isinstance(recording, str):
            recording = [recording]

        for rec in recording:
            if self._good_metadata_per_recording[rec]:
                (
                    print(f" > Loading neurons for recording {rec} from metadata")
                    if verbose
                    else None
                )
                self.info[rec]["neurons"] = Neurons(self.folder_metadata, rec, subfolder=self._neurons_metadata_subfolder)
                if (
                    self.info[rec]["neurons"].coord_xyz.shape[0]
                    != self.info[rec]["n_neurons"]
                ):
                    print(
                        f"Warning: The number of neurons in the metadata for recording {rec} does not match the number of neurons in the data, Neurons coordinates data will be set to None."
                    )
                    self.info[rec]["neurons"].coord_xyz = None
                if (
                    self.info[rec]["neurons"].IDs.shape[0]
                    != self.info[rec]["n_neurons"]
                ):
                    print(
                        f"Warning: The number of neuron IDs in the metadata for recording {rec} does not match the number of neurons in the data, Neurons IDs data will be set to None."
                    )
                    self.info[rec]["neurons"].IDs = None
            else:
                (
                    print(f" > Loading neurons for recording {rec} from data folder")
                    if verbose
                    else None
                )
                try:
                    path_to_meta_neurons = os.path.join(
                        self.folder_data, rec, "meta", "neurons"
                    )
                    neurons_coord, neurons_ids = check_meta_neurons_integrity(
                        path_to_meta_neurons,
                        n_neurons=self.info[rec]["n_neurons"],
                        verbose=True,
                    )
                    self.info[rec]["neurons"] = NeuronsData(neurons_coord, neurons_ids)
                    if neurons_coord is None or neurons_ids is None:
                        print(
                            f"Error loading neurons for recording {rec} from data folder: neurons_coord or neurons_ids is None."
                        )
                except Exception as e:
                    print(
                        f"Error loading neurons for recording {rec} from data folder: {e}. Neurons data will be set to None."
                    )
                    self.info[rec]["neurons"] = NeuronsData(None, None)

    def load_trial_types(
        self, recording: str | list[str] | None = None, verbose: bool = True
    ) -> None:
        """Load trial-type labels for selected recordings.

        Parameters
        ----------
        recording : str or list[str] or None, optional
            Recordings to process. If ``None``, process all recordings.
        verbose : bool, default=True
            If ``True``, print progress messages.
        """
        if recording is None:
            recording = self.recording
        elif isinstance(recording, str):
            recording = [recording]

        for rec in recording:
            loaded = False
            if self._good_metadata_per_recording[rec] and self._trials_metadata_file_type=='csv':
                try:
                    path_to_table = os.path.join(self.folder_metadata, rec, self._trials_metadata_subfolder)
                    file = os.path.join(path_to_table, f"meta-trials_{rec}.csv")
                    df = pd.read_csv(file)
                    if "trial_type" in df.columns:
                        self.info[rec]["trial_type"] = df["trial_type"].copy().to_list()
                        loaded = True
                    else: 
                        print("trial_type not found in the trials metadata. It will be loaded from the data folder.")
                except Exception as e:
                    print(
                        f"Error loading trial types for recording {rec} from metadata: {e}. It will be loaded from the data folder."
                    ) if verbose else None
                    
            if not loaded:
                try:
                    path_to_meta_trials = os.path.join(
                        self.folder_data, rec, "meta", "trials"
                    )
                    trial_types = check_meta_trials_integrity(
                        path_to_meta_trials,
                        n_trials=self.info[rec]["n_trials"],
                        verbose=verbose,
                    )
                    self.info[rec]["trial_type"] = trial_types
                except Exception as e:
                    print(
                        f"Error loading trial types for recording {rec} from metadata: {e}. Trial types will be set to None."
                    )
                    self.info[rec]["trial_type"] = None

    def add_data_info(self):
        """
        Add information to the info varaible about the data
        """
        if not hasattr(self,"info"):
            self.info = {}

        for rec in self.recording:
            # parse some information from the recording name, to be stored in the info variable
            info_data_rec_general = parse_info_from_recording_name(rec, verbose=verbose)
            # get some information about the data for the recording
            info_data_rec = {}
            files = self.get_data_list(recording=rec, what_data="responses")
            data_response_example = np.load(files[0], mmap_mode="r")
            info_data_rec["n_trials"] = len(files)
            info_data_rec["trials"] = set([Path(f).stem for f in files])
            info_data_rec["samples_per_trial"] = data_response_example.shape[-1]
            info_data_rec["n_neurons"] = data_response_example.shape[0]
            # store the information about the data for the recording
            self.info[rec] = {**info_data_rec_general, **info_data_rec}

    def check_data(self, verbose: bool = True) -> None:
        """Check consistency of raw data files across recordings.

        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, print warnings and status messages.
        """

        print_title("Checking the data ", verbose)

        self.info = {}
        good_data_per_recording = {}
        for rec in self.recording:

            # parse some information from the recording name, to be stored in the info variable
            info_data_rec_general = parse_info_from_recording_name(rec, verbose=verbose)
            
            # check the data folder
            path_to_data = os.path.join(self.folder_data, rec, "data")
            data_ok, info_data_rec = check_data_integrity(path_to_data, verbose=verbose)

            # store the information about the data for the recording
            self.info[rec] = {**info_data_rec_general, **info_data_rec}

            # store the information about data quality
            good_data_per_recording[rec] = data_ok

            # print some final info and store some information
            if good_data_per_recording[rec]:
                (
                    print(
                        f" > All data files seem consistent across trials and data types for recording {rec}."
                    )
                    if verbose
                    else None
                )

        # Check if data is valid across all recordings
        is_valid = all([valid_rec for valid_rec in good_data_per_recording.values()])

        # store the information about data quality
        self._good_data_per_recording = good_data_per_recording
        self._good_data = is_valid

        if is_valid:
            (
                print(" > VALID data for all recordings in the dataset")
                if verbose
                else None
            )
        else:
            print(" > INVALID data") if verbose else None

    def set_globalmetadata_folders(self, verbose: bool = True) -> None:
        '''Set paths for global metadata folders based on the main metadata folder.'''

        if self.folder_metadata is None:
            self.folder_globalmetadata_videos = None
            self.folder_globalmetadata_segments = None

        elif not Path(self.folder_metadata).exists():
            (
                print(
                    "Warning: The metadata folder was set but it does not exist, you can create it with create_folders_metadata()"
                )
                if verbose
                else None
            )
            self.folder_globalmetadata_videos = None
            self.folder_globalmetadata_segments = None

        else:
            self.folder_globalmetadata_videos = (
                Path(self.folder_metadata) / "global_meta" / "videos"
            )
            self.folder_globalmetadata_segments = (
                Path(self.folder_metadata) / "global_meta" / "segments"
            )


    def check_metadata(self, verbose: bool = True) -> None:
        """Validate global metadata folders and files.

        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, print warnings and status messages.
        """
        # check that the metadata folder has the correct structure and that the files are consistent with the data files

        print_title("Checking metadata ", verbose)

        if not hasattr(self, "info"):
            raise ValueError(
                "Data must have info attribute before checking metadata"
            )
        
        if (self.folder_globalmetadata_videos is not None) and (self.folder_globalmetadata_segments is not None):
            # if the global metadata folders are already set, we can check the metadata integrity
            is_valid, results = check_metadata_integrity(
                self.folder_metadata,
                self.recording,
                self.folder_globalmetadata_videos,
                self.folder_globalmetadata_segments,
                self.info,
                trials_metadata_file_type=self._trials_metadata_file_type,
                trials_metadata_subfolder=self._trials_metadata_subfolder,
                neurons_metadata_subfolder=self._neurons_metadata_subfolder,
                verbose=True,
            )
        else:
            is_valid = False
            results = None

        self._good_metadata = is_valid
        if results is not None:
            self._good_metadata_per_recording = results["good_metadata_per_recording"]
            self._good_global_meta_videos = results["good_global_meta_videos"]
            self._good_global_meta_segments = results["good_global_meta_segments"]
        else:
            self._good_metadata_per_recording = {rec: False for rec in self.recording}
            self._good_global_meta_videos = False
            self._good_global_meta_segments = False

        if is_valid:
            (
                print(" > VALID metadata for all recordings in the dataset")
                if verbose
                else None
            )
        else:
            print(" > INVALID metadata") if verbose else None


    def get_data_list(self, recording: str, what_data: str = "videos") -> list[Path]:
        """Return list of data files for one recording/data type.

        Parameters
        ----------
        recording : str
            Recording name.
        what_data : str, default='videos'
            Data subfolder name.

        Returns
        -------
        list[pathlib.Path]
            Matching ``.npy`` files.
        """
        path_to_data = os.path.join(self.folder_data, recording, "data", what_data)
        return list(Path(path_to_data).glob("*.npy"))

    def create_folders_metadata(
        self,
        recording: str | list[str] | None = None,
        what_global_data: str | list[str] = ["videos", "segments"],
        verbose: bool = True,
    ) -> None:
        """Create global and per-recording metadata folders when missing.

        Parameters
        ----------
        recording : str or list[str] or None, optional
            Recordings to process.
        what_global_data : str or list[str], default=['videos', 'segments']
            Global metadata categories to create.
        verbose : bool, default=True
            If ``True``, print created paths.
        """

        if self.folder_metadata is None:
            raise ValueError(
                "folder_metadata is None, cannot create folder for metadata"
            )

        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]

        if isinstance(what_global_data, str):
            what_global_data = [what_global_data]

        print_title("Creating metadata folders if necessary ", verbose)
        for rec in recording:

            path_to_meta = Path(self.folder_metadata) / rec
            created = not path_to_meta.exists()
            path_to_meta.mkdir(parents=True, exist_ok=True)
            
            path_to_meta_trials = path_to_meta / self._trials_metadata_subfolder
            path_to_meta_trials.mkdir(parents=True, exist_ok=True)
            
            path_to_meta_neurons = path_to_meta / self._neurons_metadata_subfolder
            path_to_meta_neurons.mkdir(parents=True, exist_ok=True)
            
            if created:
                print(f"- Metadata folder for recording {rec} was created in {path_to_meta}") if verbose else None
                    
        for w in what_global_data:
            path_to_meta_global = Path(self.folder_metadata) / "global_meta" / w
            created = not path_to_meta_global.exists()
            path_to_meta_global.mkdir(parents=True, exist_ok=True)
            if created:
                print(f"- Metadata folder for global metadata {w} was created in {path_to_meta_global}") if verbose else None
                
        if "videos" in what_global_data:
            self.folder_globalmetadata_videos = os.path.join(
                self.folder_metadata, "global_meta", "videos"
            )
        if "segments" in what_global_data:
            self.folder_globalmetadata_segments = os.path.join(
                self.folder_metadata, "global_meta", "segments"
            )

    def get_trials_metadata(
        self, 
        set_trials_df: bool = True, 
        verbose: bool = True) -> pd.DataFrame:
        """Load trial metadata from recording CSV tables or JSON files.

        Parameters
        ----------
        set_trials_df : bool, default=True
            If ``True``, store result in ``self.trials_df``.
        verbose : bool, default=True
            If ``True``, print progress messages.

        Returns
        -------
        pandas.DataFrame
            Combined trial metadata table.
        """

        if self.folder_metadata is None:
            raise ValueError("folder_metadata is None, cannot load trials metadata")
        
        print_title("Creating trials metadata DataFrame from meta-trials files per recording ", verbose)
        recording = self.recording
        trials_df = []

        for rec in recording:
            
            if self._trials_metadata_file_type == "csv":
                file = self.folder_metadata / rec / self._trials_metadata_subfolder / f"meta-trials_{rec}.csv"
                if not file.is_file():
                    print(f"Warning: Trials metadata file not found for recording {rec} at expected path {file}, skipping this recording.") if verbose else None
                    continue
                df = pd.read_csv(file)
                df["trial"] = df["trial"].astype(str)
                if "recording" not in df.columns:
                    df.insert(0, "recording", [rec] * len(df))

            elif self._trials_metadata_file_type == "json":
                folder = self.folder_metadata / rec / self._trials_metadata_subfolder
                df, _ = _json_to_dataframe(folder, 
                                        file_pattern="*.json", 
                                        include_file_as_column=False, 
                                        verbose=verbose)
                if "recording" not in df.columns:
                    df.insert(0, "recording", [rec] * len(df))

            if len(trials_df) == 0:
                trials_df = df.copy()
            else:
                trials_df = pd.concat([trials_df, df], axis=0)

        if len(trials_df) == 0:
            raise ValueError(f"No trials metadata found in {self.folder_metadata}")

        trials_df = trials_df.reset_index(drop=True)

        if set_trials_df:
            self.trials_df = trials_df
                


        return trials_df
    
    def get_trials_metadata_basic(self, set_trials_df: bool = True, verbose: bool = True) -> pd.DataFrame:
        """Create a basic trial metadata table from the data folder structure and the info variable.

        Parameters
        ----------
        set_trials_df : bool, default=True
            If ``True``, store result in ``self.trials_df``.
        verbose : bool, default=True
            If ``True``, print progress messages.

        Returns
        -------
        pandas.DataFrame
            Basic trial metadata table.
        """

        print_title("Creating basic trials metadata DataFrame from data ", verbose)

        all_rows = []
        for rec in self.recording:
            for t in self.info[rec]["trials"]:
                row = {
                    "recording": rec,
                    "trial": t,
                }
                all_rows.append(row)

        trials_df = pd.DataFrame(all_rows)
        trials_df["trial"] = trials_df["trial"].astype(str)

        if set_trials_df:
            self.trials_df = trials_df

        return trials_df

    def filter_trials(
        self,
        query: str | None = None,
        **conditions,
    ) -> pd.DataFrame:
        """Filter the trials table by one or more conditions.

        Parameters
        ----------
        query : str | None, default=None
            A query string to filter the trials using pandas.DataFrame.query().
        conditions : keyword arguments
            Conditions to filter the trials. Keys are column names and values are the values to filter by. Values can be a single value or a list of values.
            
        Returns
        -------
        pandas.DataFrame
            Filtered trial rows.

        """

        # check if a table with trials metadata is loaded, if not raise an error
        if not hasattr(self, "trials_df"):
            raise ValueError(
                "trials_df is not loaded, please run get_trials_metadata() first"
            )
        df = self.trials_df.copy()

        return filter_dataframe(df, query=query, **conditions)

    def get_indexes_of_trials(
        self,
        query: str | None = None,
        **conditions,
    ) -> list[int]:
        """Map filtered trial names to numeric trial indices.

        Returns
        -------
        list[int]
            Trial indices in recording order.
        """

        if 'recording' not in conditions:
            raise ValueError("The 'recording' condition is required to get trial indexes")
        recording = conditions.get('recording')
        if not isinstance(recording, str) or recording not in self.recording:
            raise ValueError(
                f"'recording' must be a string indicating a recording of the dataset, {recording} is not valid"
            )

        # get a data frame with the filtered trials
        filtered_trials_df = self.filter_trials(**conditions, query=query)
        the_trials = sorted(filtered_trials_df["trial"].to_list())

        # get the indexes
        trial_indexes = [self.info[recording]["trials"].index(t) for t in the_trials]

        return trial_indexes

    def count_videos_across(self, subset: list[str] | tuple[str]) -> pd.DataFrame:
        """Count trial rows grouped by selected columns.

        Parameters
        ----------
        subset : list[str] or tuple[str]
            Group-by columns.

        Returns
        -------
        pandas.DataFrame
            Group counts.
        """

        all_conditions = {"recording", "label", "trial_type", "ID"}
        if not (set(subset) <= all_conditions):
            raise ValueError(f"The subset must be included in {all_conditions}")

        if not isinstance(subset, (list, tuple)):
            subset = sorted(list(subset))

        # check the subset is included in the columns of the trials_df, if not raise an error
        if not hasattr(self, "trials_df"):
            raise ValueError(
                "trials_df is not loaded, please run get_trials_metadata() first"
            )
        if not set(subset) <= set(self.trials_df.columns):
            raise ValueError(
                f"The subset must be included in the columns of trials_df, but {set(subset) - set(self.trials_df.columns)} are not"
            )
        all_trials_df = self.trials_df.copy()

        # count
        counts = all_trials_df.value_counts(subset=subset)
        counts_df = counts.reset_index()
        counts_df.columns = subset + ["count"]

        return counts_df

    def load_video_by_id(self, id: str) -> VideoID:
        """Load a ``VideoID`` object by unique video ID.

        Parameters
        ----------
        id : str
            Unique video identifier.

        Returns
        -------
        VideoID
            Loaded video object.
        """

        if self.folder_globalmetadata_videos is None:
            raise ValueError("folder_metadata is None, cannot load video by id")

        files = list(Path(self.folder_globalmetadata_videos).glob(f"*{id}.json"))
        if len(files) != 1:
            raise ValueError(
                f"{len(files)} files were found with the pattern *{id}.json in {self.folder_globalmetadata_videos}, but 1 was expected"
            )

        return VideoID(
            self.folder_data,
            self.folder_globalmetadata_videos,
            files[0].stem.split("-")[1],
        )

    def load_segment_by_id(self, id: str) -> VideoSegmentID:
        """Load a ``VideoSegmentID`` object by unique segment ID.

        Parameters
        ----------
        id : str
            Unique segment identifier.

        Returns
        -------
        VideoSegmentID
            Loaded segment object.
        """

        if (
            self.folder_globalmetadata_segments is None
            or self.folder_globalmetadata_videos is None
        ):
            raise ValueError("folder_metadata is None, cannot load segment by id")

        files = list(Path(self.folder_globalmetadata_segments).glob(f"*{id}.json"))
        if len(files) != 1:
            raise ValueError(
                f"{len(files)} files were found with the pattern *{id}.json in {self.folder_globalmetadata_segments}, but 1 was expected"
            )

        return VideoSegmentID(
            self.folder_data,
            self.folder_globalmetadata_videos,
            self.folder_globalmetadata_segments,
            files[0].stem.split("-")[1],
        )

    def load_video_by_trial(
        self,
        recording: str,
        trial: str,
        verbose: bool = True,
        load_metadata_from_dataframe : bool = True,
        raise_on_mismatch: bool = False,
        load_metadata_from_global_video: bool = True,
        load_metadata_from_trials: bool = True,
    ) -> Video:
        """Load one trial video and attach available metadata.

        Parameters
        ----------
        recording : str
            Recording name.
        trial : str
            Trial name.
        verbose : bool, default=True
            If ``True``, print warnings.
        try_global_first : bool, default=True
            If ``True``, prioritize loading global-ID metadata.

        Returns
        -------
        Video
            Loaded video object.
        """

        # lookup trial metadata
        if load_metadata_from_dataframe:
            trials_meta = self.filter_trials(recording=recording, trial=trial)
            if len(trials_meta) != 1:
                raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")
            if "ID" in trials_meta.columns:
                ID = trials_meta["ID"].iloc[0]
            else:
                ID = None
            if "label" in trials_meta.columns:
                label = trials_meta["label"].iloc[0]
            else:
                label = None
            if "valid_frames" in trials_meta.columns:
                valid_frames = trials_meta["valid_frames"].iloc[0]
                valid_frames = None if pd.isna(valid_frames) else valid_frames
            else:
                valid_frames = None
        else:
            ID = None
            label = None
            valid_frames = None

        # load the data
        recording_folder = os.path.join(self.folder_data, recording)
        video = Video(recording_folder, trial, ID=ID, label=label, valid_frames=valid_frames)

        # try loading metadata from global metadata folder (if configured)
        if self._good_global_meta_videos and load_metadata_from_global_video:
            try:
                file = get_file_with_pattern(
                    f"*-{video.ID}.json", self.folder_globalmetadata_videos
                )
                video.load_metadata_from_id(file, 
                                            raise_on_mismatch = raise_on_mismatch, 
                                            verbose = verbose)
                return video
            except Exception as e:
                if verbose:
                    print(f"load_metadata_from_id failed: {e}")

        # try loading metadata from metadata per trials
        if self._good_metadata_per_recording and self._trials_metadata_file_type == "json" and load_metadata_from_trials:
            path_to_metadata_file = self.folder_metadata / recording / self._trials_metadata_subfolder / f"{trial}.json"
            if path_to_metadata_file.exists():
                try:
                    video.load_metadata(path_to_metadata_file,
                                           attributes_to_check_match = ["label", "ID", "valid_frames","sampling_freq"],
                                           attributes_to_add = None,
                                           raise_on_mismatch = raise_on_mismatch,
                                           verbose = verbose)
                except Exception as e:
                    if verbose:
                        print(f"Could not load metadata from JSON trial file: {e}")
            else:
                if verbose:
                    print(
                        f"Could not load metadata from JSON trial file: Metadata file not found: {path_to_metadata_file}"
                    )

        return video

    def load_response_by_trial(
        self,
        recording: str,
        trial: str,
        verbose: bool = True,
        load_metadata_from_dataframe : bool = True,
        raise_on_mismatch: bool = False,
        load_metadata_from_global_video: bool = True,
        load_metadata_from_trials: bool = True,
    ) -> Responses:
        """Load one trial responses object and attach metadata.

        Returns
        -------
        Responses
            Loaded responses object.
        """

        # lookup trial metadata
        if load_metadata_from_dataframe:
            trials_meta = self.filter_trials(recording=recording, trial=trial)
            if len(trials_meta) != 1:
                raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")
            if "ID" in trials_meta.columns:
                ID = trials_meta["ID"].iloc[0]
            else:
                ID = None
            if "label" in trials_meta.columns:
                label = trials_meta["label"].iloc[0]
            else:
                label = None
            if "valid_frames" in trials_meta.columns:
                valid_frames = trials_meta["valid_frames"].iloc[0]
                valid_frames = None if pd.isna(valid_frames) else valid_frames
            else:
                valid_frames = None
        else:
            ID = None
            label = None
            valid_frames = None

        # load the data
        recording_folder = os.path.join(self.folder_data, recording)
        response = Responses(recording_folder, trial, ID=ID, label=label, valid_frames=valid_frames)

        # load neurons metadata
        response.neurons = self.info[recording]["neurons"]

        # try loading metadata from global metadata folder (if configured)
        if self._good_global_meta_videos and load_metadata_from_global_video:
            try:
                file = get_file_with_pattern(
                    f"*-{response.ID}.json", self.folder_globalmetadata_videos
                )
                response.load_metadata(file,
                                       attributes_to_check_match = ["label", "ID", "valid_frames","sampling_freq"],
                                       attributes_to_add = ["segments","duplicates"],
                                       raise_on_mismatch = raise_on_mismatch,
                                       verbose = verbose)
            except Exception as e:
                if verbose:
                    print(
                        f"Loading metadata from {self.folder_globalmetadata_videos} failed: {e}"
                    )

        # fallback: try loading metadata from metadata per trials
        if self._good_metadata_per_recording and self._trials_metadata_file_type == "json" and load_metadata_from_trials:
            path_to_metadata_file = self.folder_metadata / recording / self._trials_metadata_subfolder / f"{trial}.json"
            if path_to_metadata_file.exists():
                try:
                    response.load_metadata(path_to_metadata_file,
                                           attributes_to_check_match = ["label", "ID", "valid_frames","sampling_freq"],
                                           attributes_to_add = None,
                                           raise_on_mismatch = raise_on_mismatch,
                                           verbose = verbose)
                except Exception as e:
                    if verbose:
                        print(f"Could not load metadata from JSON trial file: {e}")
            else:
                if verbose:
                    print(
                        f"Could not load metadata from JSON trial file: Metadata file not found: {path_to_metadata_file}"
                    )

        return response

    def load_behavior_by_trial(
        self,
        recording: str,
        trial: str,
        behavior_type: str = "pupil",
        verbose: bool = True,
        load_metadata_from_dataframe : bool = True,
        raise_on_mismatch: bool = False,
        load_metadata_from_global_video: bool = True,
        load_metadata_from_trials: bool = True,
    ) -> Pupil | Gaze | Locomotion:
        """Load one trial behavior object and attach metadata.

        Parameters
        ----------
        behavior_type : {'pupil', 'gaze', 'locomotion'}, default='pupil'
            Behavior modality.

        Returns
        -------
        Pupil | Gaze | Locomotion
            Loaded behavior object.
        """

        # lookup trial metadata
        if load_metadata_from_dataframe:
            trials_meta = self.filter_trials(recording=recording, trial=trial)
            if len(trials_meta) != 1:
                raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")
            if "ID" in trials_meta.columns:
                ID = trials_meta["ID"].iloc[0]
            else:
                ID = None
            if "label" in trials_meta.columns:
                label = trials_meta["label"].iloc[0]
            else:
                label = None
            if "valid_frames" in trials_meta.columns:
                valid_frames = trials_meta["valid_frames"].iloc[0]
                valid_frames = None if pd.isna(valid_frames) else valid_frames
            else:
                valid_frames = None
        else:
            ID = None
            label = None
            valid_frames = None

        # load the data
        recording_folder = os.path.join(self.folder_data, recording)
        if behavior_type.lower() == "pupil":
            behavior = Pupil(recording_folder, trial, ID=ID, label=label, valid_frames=valid_frames)
        elif behavior_type.lower() == "gaze":
            behavior = Gaze(recording_folder, trial, ID=ID, label=label, valid_frames=valid_frames)
        elif behavior_type.lower() == "locomotion":
            behavior = Locomotion(recording_folder, trial, ID=ID, label=label, valid_frames=valid_frames)
        else:
            raise ValueError(
                f"behavior_type must be 'pupil', 'gaze', or 'locomotion', got {behavior_type}"
            )

        
        # try loading metadata from global metadata folder (if configured)
        if self._good_global_meta_videos and load_metadata_from_global_video:
            try:
                file = get_file_with_pattern(
                    f"*-{behavior.ID}.json", self.folder_globalmetadata_videos
                )
                behavior.load_metadata(file,
                                       attributes_to_check_match = ["label", "ID", "valid_frames","sampling_freq"],
                                       attributes_to_add = ["segments","duplicates"],
                                       raise_on_mismatch = raise_on_mismatch,
                                       verbose = verbose)
            except Exception as e:
                if verbose:
                    print(
                        f"Loading metadata from {self.folder_globalmetadata_videos} failed: {e}"
                    )

        # fallback: try loading metadata from metadata per trials (if configured)
        if self._good_metadata_per_recording and self._trials_metadata_file_type == "json" and load_metadata_from_trials:
            path_to_metadata_file = self.folder_metadata / recording / self._trials_metadata_subfolder / f"{trial}.json"
            if path_to_metadata_file.exists():
                try:
                    behavior.load_metadata(path_to_metadata_file,
                                           attributes_to_check_match = ["label", "ID", "valid_frames","sampling_freq"],
                                           attributes_to_add = None,
                                           raise_on_mismatch = raise_on_mismatch,
                                           verbose = verbose)
                except Exception as e:
                    if verbose:
                        print(f"Could not load metadata from JSON trial file: {e}")
            else:
                if verbose:
                    print(
                        f"Could not load metadata from JSON trial file: Metadata file not found: {path_to_metadata_file}"
                    )

        return behavior

    def load_responses_by(
        self,
        verbose: bool = True,
        raise_on_mismatch: bool = False,
        load_metadata_from_global_video: bool = True,
        load_metadata_from_trials: bool = True,
        query: str | None = None,
        **conditions,
    ) -> tuple[list[Responses], pd.DataFrame]:
        """Load responses for all trials matching filters.

        Returns
        -------
        tuple[list[Responses], pandas.DataFrame]
            Loaded objects and the filtered trial table.
        """

        trials_df = self.filter_trials(**conditions, query=query)
        responses = []
        for index, row in trials_df.iterrows():
            resp = self.load_response_by_trial(
                                            recording=row["recording"], 
                                            trial=row["trial"], 
                                            raise_on_mismatch=raise_on_mismatch,
                                            load_metadata_from_global_video=load_metadata_from_global_video,
                                            load_metadata_from_trials=load_metadata_from_trials,
                                            verbose=verbose
                                        )
            responses.append(resp)

        return responses, trials_df

    def load_videos_by(
        self,
        verbose: bool = True,
        raise_on_mismatch: bool = False,
        load_metadata_from_global_video: bool = True,
        load_metadata_from_trials: bool = True,
        query: str | None = None,
        **conditions,
    ) -> tuple[list[Video], pd.DataFrame]:
        """Load videos for all trials matching filters.

        Returns
        -------
        tuple[list[Video], pandas.DataFrame]
            Loaded objects and the filtered trial table.
        """

        trials_df = self.filter_trials(query=query, **conditions)
        videos = []
        for index, row in trials_df.iterrows():
            vi = self.load_video_by_trial(
                                        recording=row["recording"], 
                                        trial=row["trial"], 
                                        raise_on_mismatch=raise_on_mismatch,
                                        load_metadata_from_global_video=load_metadata_from_global_video,
                                        load_metadata_from_trials=load_metadata_from_trials,
                                        verbose=verbose
                                    )
            videos.append(vi)

        return videos, trials_df

    def load_behavior_by(
        self,
        behavior_type: str,
        verbose: bool = True,
        raise_on_mismatch: bool = False,
        load_metadata_from_global_video: bool = True,
        load_metadata_from_trials: bool = True,
        query: str | None = None,
        **conditions,
    ) -> tuple[list[Pupil | Gaze | Locomotion], pd.DataFrame]:
        """Load behavior objects for all trials matching filters.

        Returns
        -------
        tuple[list[Pupil | Gaze | Locomotion], pandas.DataFrame]
            Loaded objects and the filtered trial table.
        """

        trials_df = self.filter_trials(**conditions, query=query)
        behavior = []
        for index, row in trials_df.iterrows():
            beh = self.load_behavior_by_trial(
                                        recording=row["recording"],
                                        trial=row["trial"],
                                        behavior_type=behavior_type,
                                        raise_on_mismatch=raise_on_mismatch,
                                        load_metadata_from_global_video=load_metadata_from_global_video,
                                        load_metadata_from_trials=load_metadata_from_trials,
                                        verbose=verbose,
                                    )
            behavior.append(beh)

        return behavior, trials_df

    def compute_dissimilarity_videos(
        self,
        dissimilarity_measure: str = "mse",
        check_edges_first: bool = True,
        verbose: bool = True,
        query: str | None = None,
        **conditions,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute pairwise video dissimilarity for filtered trials.

        Returns
        -------
        tuple[numpy.ndarray, pandas.DataFrame]
            Dissimilarity matrix and filtered trial table.
        """
        videos, trials_df = self.load_videos_by(**conditions, query=query, verbose=verbose)
        dissimilarity = compute_dissimilarity_video_list(
            videos,
            dissimilarity_measure=dissimilarity_measure,
            check_edges_first=check_edges_first,
        )
        return dissimilarity, trials_df

    def find_segment(self, segment_id: str) -> pd.DataFrame:
        """Build a table of trial occurrences for one segment ID.

        Parameters
        ----------
        segment_id : str
            Unique segment identifier.

        Returns
        -------
        pandas.DataFrame
            Table with trial-level occurrences of the segment.
        """

        if (
            self.folder_globalmetadata_segments is None
            or self.folder_globalmetadata_videos is None
        ):
            raise ValueError("folder_metadata is None, cannot find segments by id")

        # load metadata
        metadata_segment, _ = load_metadata_from_id(
            segment_id, self.folder_globalmetadata_segments
        )

        # get the duplicates of the segment
        duplicates_segment = metadata_segment.get("duplicates", {})
        if not duplicates_segment:
            raise ValueError("No duplicates found in segment metadata")

        # loop over each video id containig the segment
        trials = []
        recording = []
        video_label = []
        video_id = []
        segment_label = []
        segment_index = []
        frame_start = []
        frame_end = []
        for v_id, s_duplicates_val in duplicates_segment.items():
            # load metadata video id
            metadata_video, _ = load_metadata_from_id(
                v_id, self.folder_globalmetadata_videos
            )
            duplicates_video = metadata_video.get("duplicates", {})
            if not duplicates_video:
                raise ValueError(f"No duplicates found in video metadata for {v_id}")
            for segm_idx in s_duplicates_val["segment_index"]:
                for rec, v_duplicates_val in duplicates_video.items():
                    trl = list(v_duplicates_val["trials"])
                    trials = trials + trl
                    recording = recording + [rec for i in range(len(trl))]
                    video_label = video_label + [
                        metadata_video["label"] for i in range(len(trl))
                    ]
                    video_id = video_id + [v_id for i in range(len(trl))]
                    segment_label = segment_label + [
                        metadata_segment["label"] for i in range(len(trl))
                    ]
                    segment_index = segment_index + [segm_idx for i in range(len(trl))]
                    frame_start = frame_start + [
                        metadata_video["segments"]["frame_start"][segm_idx]
                        for i in range(len(trl))
                    ]
                    frame_end = frame_end + [
                        metadata_video["segments"]["frame_end"][segm_idx]
                        for i in range(len(trl))
                    ]

        return pd.DataFrame(
            {
                "segment_ID": [segment_id for i in range(len(trials))],
                "segment_label": segment_label,
                "video_ID": video_id,
                "video_label": video_label,
                "recording": recording,
                "trial": trials,
                "segment_index": segment_index,
                "frame_start": frame_start,
                "frame_end": frame_end,
            }
        )

    def get_segments_metadata(
        self, set_segments_df: bool = True, verbose: bool = True
    ) -> pd.DataFrame:
        """Load and aggregate all global segment metadata.

        Returns
        -------
        pandas.DataFrame
            Segment occurrence table.
        """

        if (
            self.folder_globalmetadata_segments is None
            or self.folder_globalmetadata_videos is None
        ):
            raise ValueError("folder_metadata is None, cannot load segment by id")

        # load the segments metadata
        s = f"Creating segments metadata DataFrame from JSON files in {self.folder_globalmetadata_segments} "
        print_title(s, verbose)
        files = list(Path(self.folder_globalmetadata_segments).glob("*.json"))
        if len(files) == 0:
            raise ValueError(
                f"No json files found in {self.folder_globalmetadata_segments}"
            )
        for i, fff in enumerate(files):
            df = self.find_segment(Path(fff).stem.split("-")[1])
            if i == 0:
                segments_df = df.copy()
            else:
                segments_df = pd.concat([segments_df, df])

        segments_df = segments_df.reset_index(drop=True)

        # store as an attribute
        if set_segments_df:
            self.segments_df = segments_df

        return segments_df

    def filter_segments(
        self,
        query: str | None = None,
        **conditions,
   ) -> pd.DataFrame:
        """Filter segment table by one or more conditions.

        Returns
        -------
        pandas.DataFrame
            Filtered segment rows.
        """


        # check if segments metadata is loaded
        if not hasattr(self, "segments_df"):
            raise ValueError(
                "segments_df is not loaded, please run get_segments_metadata() first"
            )
        df = self.segments_df.copy()

        return filter_dataframe(df, query=query, **conditions)

    def count_segments_across(
        self, subset: list[str] | tuple[str] | set[str]
    ) -> pd.DataFrame:
        """Count segment rows grouped by selected columns.

        Parameters
        ----------
        subset : list[str] or tuple[str]
            Group-by columns.

        Returns
        -------
        pandas.DataFrame
            Group counts.
        """

        all_conditions = {
            "recording",
            "video_label",
            "segment_label",
            "video_ID",
            "segment_ID",
            "segment_index",
        }
        if not (set(subset) <= all_conditions):
            raise ValueError(f"The subset must be included in {all_conditions}")

        if not isinstance(subset, (list, tuple)):
            subset = sorted(list(subset))

        # load a table with segments metadata if not loaded yet
        if not hasattr(self, "segments_df"):
            raise ValueError(
                "segments_df is not loaded, please run get_segments_metadata() first"
            )
        all_segments_df = self.segments_df.copy()

        # count
        counts = all_segments_df.value_counts(subset=subset)
        counts_df = counts.reset_index()
        counts_df.columns = subset + ["count"]

        return counts_df

    def load_all_data(
        self, recording: str, what_data: str, data_slice: slice | tuple | None = None
    ) -> np.ndarray:
        """Load all data arrays for one recording/data type.

        Parameters
        ----------
        recording : str
            Recording name.
        what_data : str
            Data category (for example, ``'responses'`` or ``'videos'``).
        data_slice : slice or tuple or None, optional
            Optional slice applied to each loaded array.

        Returns
        -------
        numpy.ndarray
            Stacked trial array.
        """

        return load_all_data(
            os.path.join(self.folder_data, recording), what_data, data_slice=data_slice
        )

    def compute_neurons_stats(
        self, recording: str, trials_for_stats: list[str] | None = None
    ) -> pd.DataFrame:
        """Compute per-neuron descriptive statistics across selected trials.

        Parameters
        ----------
        recording : str
            Recording name.
        trials_for_stats : list or None, optional
            List of trial to include in the statistics. If ``None``, use all trials.

        Returns
        -------
        pandas.DataFrame
            Columns include mean, std, median, min, and max activation.
        """

        # get the number of neurons
        if trials_for_stats is None:
            trials_for_stats = self.info[recording]["trials"]
        n_neurons = self.info[recording]["n_neurons"]

        # initialize
        stats = {}
        val_sum = np.zeros(n_neurons)
        n = 0
        val_min = np.full(n_neurons, np.inf)
        val_max = np.full(n_neurons, -np.inf)
        trials_included = []
        for trial in tqdm(trials_for_stats, desc="MEAN, MAX, MIN computation", unit="trial",total=len(trials_for_stats), disable=False):
            try:
                resp = self.load_response_by_trial(recording, 
                                                   trial, 
                                                   load_metadata_from_global_video=False, 
                                                   load_metadata_from_trials=False)
                trials_included.append(trial)
            except Exception as e:
                print(f"Could not load responses for trial {trial} in recording {recording}: {e}")
                continue

            data = resp.get_data()
            val_sum += np.sum(data, axis=-1)
            n += data.shape[-1]
            val_min = np.minimum(val_min, np.min(data, axis=-1))
            val_max = np.maximum(val_max, np.max(data, axis=-1))

        stats["mean_activation"] = val_sum / n
        stats["min_activation"] = val_min
        stats["max_activation"] = val_max

        val_sum_squared = np.zeros(n_neurons)
        for trial in tqdm(trials_for_stats, desc="STD computation", unit="trial",total=len(trials_for_stats), disable=False):
            try:
                resp = self.load_response_by_trial(recording, 
                                                   trial,
                                                   load_metadata_from_global_video=False, 
                                                   load_metadata_from_trials=False)
            except Exception as e:
                print(f"Could not load responses for trial {trial} in recording {recording}: {e}")
                continue
            data = resp.get_data()
            val_sum_squared += ((data - stats["mean_activation"][:,np.newaxis]) ** 2).sum(axis=-1)
        stats["std_activation"] = np.sqrt(val_sum_squared / n)

        return pd.DataFrame.from_dict(stats)
    

    def compute_neurons_stats_per_neuron(
        self, recording: str, trials_for_stats: list[str] | None = None
    ) -> pd.DataFrame:
        """Compute per-neuron descriptive statistics across selected trials.

        Parameters
        ----------
        recording : str
            Recording name.
        trials_for_stats : list[str] or None, optional
            List of trial to include in the statistics. If ``None``, use all trials.

        Returns
        -------
        pandas.DataFrame
            Columns include mean, std, median, min, and max activation.
        """

        # get the number of neurons
        n_neurons = self.info[recording]["n_neurons"]
        n_trials = self.info[recording]["n_trials"]

        # initialize
        stats = {}
        stats["mean_activation"] = np.full(n_neurons, np.nan)
        stats["std_activation"] = np.full(n_neurons, np.nan)
        stats["median_activation"] = np.full(n_neurons, np.nan)
        stats["min_activation"] = np.full(n_neurons, np.nan)
        stats["max_activation"] = np.full(n_neurons, np.nan)

        if trials_for_stats is None:
            trials_for_stats = self.info[recording]["trials"]
        
        idx_trials_stats = np.isin(np.arange(n_trials), trials_for_stats)

        n_included = np.sum(idx_trials_stats)
        if n_included == 0:
            raise ValueError("No trials included")

        print(
            f"Computing neurons stats for {n_neurons} neurons over {np.sum(idx_trials_stats)} out of {n_trials} total trials"
        )

        for ni in tqdm(
            range(n_neurons),
            total=n_neurons,
            desc="Computing neuron stats",
            disable=False,
        ):

            # load all responses
            resp_all = self.load_all_data(
                recording, what_data="responses", data_slice=(ni, slice(None))
            )

            # compute
            stats["mean_activation"][ni] = np.nanmean(resp_all[idx_trials_stats, :])
            stats["std_activation"][ni] = np.nanstd(resp_all[idx_trials_stats, :])
            stats["median_activation"][ni] = np.nanmedian(resp_all[idx_trials_stats, :])
            stats["min_activation"][ni] = np.nanmin(resp_all[idx_trials_stats, :])
            stats["max_activation"][ni] = np.nanmax(resp_all[idx_trials_stats, :])

        return pd.DataFrame.from_dict(stats)

    def generates_neurons_metadata(
        self,
        recording: str | list[str] | None = None,
        trials_for_stats: list[str] | None = None,
        verbose: bool = True,
    ) -> None:
        """Generate and save neuron metadata tables per recording.

        Parameters
        ----------
        recording : str or list[str] or None, optional
            Recordings to process.
        trials_for_stats : list[str] or None, optional
            List of trial to include in the statistics. If ``None``, use all trials.
        verbose : bool, default=True
            If ``True``, print progress messages.
        """

        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]

        # create a folder for the outputs if it doesn't exists
        self.create_folders_metadata(what_global_data=[])

        # compute for all recordings
        print_title("Computing metadata for neurons ", verbose)
        for rec in recording:

            print(f"\nMetadata for recording {rec}") if verbose else None

            try:

                # compute the stats
                stats = self.compute_neurons_stats(
                    rec, trials_for_stats=trials_for_stats
                )

                # get neurons metadata
                neurons_coord = self.info[rec]["neurons"].coord_xyz
                if neurons_coord is not None and len(neurons_coord) > 0:
                    df_coord = pd.DataFrame(
                        neurons_coord, columns=["coord_x", "coord_y", "coord_z"]
                    )
                else:
                    (
                        print(
                            f"Warning: Neurons coordinates were not defined for recording {rec}"
                        )
                        if verbose
                        else None
                    )
                    df_coord = pd.DataFrame(
                        np.full((self.info[rec]["n_neurons"], 3), None),
                        columns=["coord_x", "coord_y", "coord_z"],
                    )

                neurons_ids = self.info[rec]["neurons"].IDs
                if neurons_ids is not None and len(neurons_ids) > 0:
                    df_id = pd.DataFrame(neurons_ids, columns=["ID"])
                else:
                    (
                        print(
                            f"Warning: Neurons IDs were not defined for recording {rec}"
                        )
                        if verbose
                        else None
                    )
                    df_id = pd.DataFrame(
                        np.full((self.info[rec]["n_neurons"], 1), None), columns=["ID"]
                    )

                # generate a dataframe with all neurons info
                meta_neurons = pd.concat([df_id, df_coord, stats], axis=1)

                # save
                folder_recording_meta_neurons = Path(self.folder_metadata) / rec / "neurons"
                folder_recording_meta_neurons.mkdir(parents=True, exist_ok=True)
                out_path = folder_recording_meta_neurons / f"meta-neurons_{rec}.csv"
                meta_neurons.to_csv(out_path, index=False)
                print(f"Saved neurons metadata: {out_path}") if verbose else None

            except Exception as e:
                print(f"Error processing recording {rec}: {e}")
                continue

    def generates_basic_metadata_per_recording(self, recording: str | list[str] | None = None, sampling_freq: float | int = 30) -> None:
        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]

        for rec in recording:
            keys = ['animal_id', 'session', 'scan_idx', 'n_trials', 'samples_per_trial', 'n_neurons']
            info_save = {k: self.info[rec][k] for k in keys if k in self.info[rec]}
            info_save['sampling_freq'] = sampling_freq
            file_out = self.folder_metadata / rec / f"meta-basic_{rec}.json"
            save_json(info_save, file_out)


    def classify_videos(
        self, recording: str | list[str] | None = None, verbose: bool = True
    ) -> None:
        """Classify trial videos and save per-trial metadata.

        Parameters
        ----------
        recording : str or list[str] or None, optional
            Recordings to process.
        verbose : bool, default=True
            If ``True``, print progress messages.
        """

        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]

        # create a folder for the outputs if it doesn't exists
        self.create_folders_metadata(what_global_data=[])

        print_title("Classifying videos ", verbose)
        for rec in recording:

            path_to_video_trials = self.get_data_list(rec, what_data="videos")
            (
                print(
                    f"\nRecording {rec} - {len(path_to_video_trials)} video files found"
                )
                if verbose
                else None
            )

            # folder for the outputs
            path_to_results_metavideos = Path(self.folder_metadata) / rec / self._trials_metadata_subfolder
            path_to_results_metavideos.mkdir(parents=True, exist_ok=True)

            # load the trials descriptor
            trial_types = self.info[rec]["trial_type"]
            if len(trial_types) != len(path_to_video_trials):
                raise ValueError(
                    "The number of trials in the descriptor does not match the number of video files"
                )

            # compute for each video (trial)
            for video_trial, trial_type in tqdm(
                zip(path_to_video_trials, trial_types),
                total=len(path_to_video_trials),
                desc=f"Processing {rec}",
                disable=False,
            ):

                try:
                    # initialize class and load video
                    video = self.load_video_by_trial(
                        rec, Path(video_trial).stem, verbose=False
                    )

                    # run all the classification
                    labels, segments = video.run_all()

                    first_label_i = labels[0] if labels else None
                    n_segments_peaks_i = (
                        len(segments[1]["duration"]) if len(segments) > 1 else 0
                    )

                    # store some other info in the Video object
                    video.first_label = first_label_i
                    video.trial_type = trial_type
                    video.segments_n_peaks = n_segments_peaks_i
                    video.segments_bad_n = np.sum(video.segments["bad_properties"])
                    video.segments_avg_duration = np.mean(video.segments["duration"])

                    # save some metadata for each video to avoid recomputing later
                    fields_to_save = [
                        "recording",
                        "trial",
                        "trial_type",
                        "first_label",
                        "label",
                        "ID",
                        "sampling_freq",
                        "valid_frames",
                        "peaks",
                        "n_peaks",
                        "segments_n_peaks",
                        "segments_bad_n",
                        "segments_avg_duration",
                    ]
                    video.save_metadata(
                        path_to_results_metavideos,
                        metadata_for="exemplar",
                        main_fields=fields_to_save,
                    )

                except Exception as e:
                    print(
                        f"Error processing video {os.path.basename(video_trial)} in {rec}: {e}"
                    )
                    continue

    def define_videos_id(
        self,
        recording: str | list[str] | None = None,
        limit_dissimilarity: float | int = 5,
        output_subfolder: str = "trials",
        verbose: bool = True,
    ) -> None:
        """Assign unique video IDs by similarity grouping.

        Parameters
        ----------
        recording : str or list[str] or None, optional
            Recordings to process.
        limit_dissimilarity : float or int, default=5
            Maximum dissimilarity to treat videos as duplicates.
        output_subfolder : str, default="trials"
            Subfolder in the recording metadata folder where the trial metadata with IDs will be saved.
        verbose : bool, default=True
            If ``True``, print progress messages.
        """

        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]

        # create a folder for the outputs if it doesn't exists
        self.create_folders_metadata(what_global_data=["videos"])

        # Load the classification tables for all recordings
        videos_df = self.get_trials_metadata()
        if "ID" not in videos_df.columns:
            videos_df["ID"] = None

        print_title("Defining videos IDs ", verbose)
        for rec in recording:

            print(f"\nComputing for recording {rec}...") if verbose else None

            path_to_data = os.path.join(self.folder_data, rec)
            path_to_results_metavideos = Path(self.folder_metadata) / rec / self._trials_metadata_subfolder
            path_to_results_metavideos.mkdir(parents=True, exist_ok=True)

            try:

                vdf_rec = videos_df[(videos_df["recording"] == rec)]
                all_labels = list(set(vdf_rec["label"].to_list()))

                for thelabel in all_labels:

                    print(f">>> Label {thelabel}") if verbose else None

                    try:

                        # compute the dissimilarity
                        dissimilarity, trials_df = self.compute_dissimilarity_videos(
                            recording=rec, label=thelabel, verbose=False
                        )

                        # mask the dissimilarity to find identical videos
                        dissimilarity_masked = dissimilarity < limit_dissimilarity

                        # find the groups of videos
                        list_distinct_videos = find_equal_sets_scipy(
                            dissimilarity_masked,
                            elements_names=trials_df["trial"].to_list(),
                        )

                        # compare each of them with the videos already identified for other recordings
                        new_ids = compare_with_idvideos(
                            thelabel,
                            list_distinct_videos,
                            path_to_data,
                            path_to_results_metavideos,
                            self.folder_globalmetadata_videos,
                            limit_dissimilarity=limit_dissimilarity,
                        )

                        # Validate new_ids
                        if len(new_ids) != len(list_distinct_videos):
                            raise ValueError(
                                f"Expected {len(list_distinct_videos)} IDs, got {len(new_ids)}"
                            )

                        # add the info to the trials table
                        for i, duplicate_trials in enumerate(list_distinct_videos):
                            mask = (
                                (videos_df["recording"] == rec)
                                & (videos_df["label"] == thelabel)
                                & videos_df["trial"].isin(duplicate_trials)
                            )
                            if np.sum(mask) != len(duplicate_trials):
                                raise ValueError(
                                    f"Label {thelabel}: Expected {len(duplicate_trials)} trials, found {np.sum(mask)}"
                                )
                            videos_df.loc[mask, "ID"] = new_ids[i]

                    except Exception as e:
                        print(f"Error processing label {thelabel} in {rec}: {e}")
                        continue

                # save the trials metadata
                df_meta_trials_rec = videos_df[videos_df["recording"] == rec].copy()
                df_meta_trials_rec["valid_trial"] = (
                    df_meta_trials_rec["segments_bad_n"] == 0
                )
                df_meta_trials_rec = df_meta_trials_rec[
                    [
                        "label",
                        "ID",
                        "trial",
                        "trial_type",
                        "valid_frames",
                        "valid_trial",
                    ]
                ]

                folder_recording_meta = Path(self.folder_metadata) / rec / output_subfolder
                folder_recording_meta.mkdir(parents=True, exist_ok=True)
                filename = folder_recording_meta / f"meta-trials_{rec}.csv"
                df_meta_trials_rec.to_csv(filename, index=False)
                print(f"Saved: {filename}")

            except Exception as e:
                print(f"Error processing recording {rec}: {e}")
                continue

    def define_segments_id(
        self,
        labels: str | list[str],
        limit_dissimilarity: float | int = 20,
        verbose: bool = True,
    ) -> None:
        """Assign unique segment IDs by similarity grouping.

        Parameters
        ----------
        labels : str or list[str]
            Segment labels to process.
        limit_dissimilarity : float or int, default=20
            Maximum dissimilarity to treat segments as duplicates.
        verbose : bool, default=True
            If ``True``, print progress messages.
        """

        # create a folder for the outputs if it doesn't exists
        self.create_folders_metadata(what_global_data=["segments"])

        # validate required folders
        if (
            self.folder_globalmetadata_videos is None
            or self.folder_globalmetadata_segments is None
        ):
            raise ValueError(
                "folder_globalmetadata_videos and folder_globalmetadata_segments must be set"
            )

        if any(Path(self.folder_globalmetadata_segments).iterdir()):
            raise ValueError(f"Files were found in {self.folder_globalmetadata_segments}. The folder should be empty before running this function.")

        # Find identical segments for each label and save metadata
        print_title("Finding identical segments ", verbose)
        all_used_ids = []

        for lab in labels:

            print(f">>> Label {lab}") if verbose else None

            all_segments = []
            folder = Path(self.folder_globalmetadata_videos)
            json_files = list(folder.glob(f"{lab}*.json"))
            print(f"- {len(json_files)} distinct videos found") if verbose else None

            if len(json_files) == 0:
                print(f"Warning: No videos found for label {lab}") if verbose else None
                continue

            # load all segments
            for file_videoID in json_files:
                video_id = None
                try:
                    video_id = Path(file_videoID).stem.split("-")[1]
                    video = self.load_video_by_id(video_id)

                    if (
                        not hasattr(video, "segments")
                        or "frame_start" not in video.segments
                    ):
                        (
                            print(f"Warning: Video {video_id} has no valid segments")
                            if verbose
                            else None
                        )
                        continue

                    for seg_idx in range(len(video.segments["frame_start"])):
                        try:
                            segment = VideoSegment(video, seg_idx)
                            segment.label_from_parentvideo()
                            all_segments.append(segment)
                        except Exception as e:
                            (
                                print(
                                    f"Warning: Could not load segment {seg_idx} from video {video_id}: {e}"
                                )
                                if verbose
                                else None
                            )
                            continue

                except Exception as e:
                    if video_id is None:
                        print(f"Warning: Could not load video from {file_videoID}: {e}")
                    else:
                        print(f"Warning: Could not load video {video_id}: {e}")
                    continue

            (
                print(f"- {len(all_segments)} segments were found and loaded")
                if verbose
                else None
            )

            if len(all_segments) == 0:
                (
                    print(f"Warning: No segments found for label {lab}")
                    if verbose
                    else None
                )
                continue

            # compute dissimilarity
            try:
                (
                    print("Computing dissimilarity between segments...")
                    if verbose
                    else None
                )
                dissimilarity = compute_dissimilarity_video_list(
                    all_segments, dissimilarity_measure="mse", check_edges_first=False
                )
            except Exception as e:
                (
                    print(f"Error computing dissimilarity for label {lab}: {e}")
                    if verbose
                    else None
                )
                continue

            # extract sets of identical segments
            mask = dissimilarity <= limit_dissimilarity
            list_identical = find_equal_sets_scipy(mask)
            (
                print(f"- {len(list_identical)} different segments were found")
                if verbose
                else None
            )

            # loop over identical segments and save metadata
            print("Saving metadata...") if verbose else None
            for setiden in list_identical:
                try:
                    # generate a new id
                    the_id = generate_new_id(all_used_ids, prefix="s")
                    all_used_ids.append(the_id)

                    # generate a SegmentID object from the exemplar segment and add the duplicates
                    segment_i_id = all_segments[next(iter(setiden))].copy(deep=True)
                    segment_i_id.ID = the_id
                    for k in setiden:
                        segment_i_id.add_duplicates(
                            all_segments[k].parentvideo["ID"],
                            all_segments[k].parentvideo["segment_index"],
                        )

                    # save a json file with the video metadata
                    segment_i_id.save_metadata(self.folder_globalmetadata_segments)

                except Exception as e:
                    print(f"Error processing segment set: {e}") if verbose else None
                    continue

    def add_segments_id_to_video_metadata(self, verbose: bool = True) -> None:
        """Inject segment IDs into each global video metadata file.

        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, print warnings and progress messages.
        """

        segments_df = self.get_segments_metadata()
        segm = segments_df[
            ["segment_ID", "segment_label", "video_ID", "video_label", "segment_index"]
        ].drop_duplicates()

        print_title("Adding segments IDs info to the videos metadata ", verbose)
        for index, row in segm.iterrows():

            try:

                # load the video ID metadata
                metadata_video, file_path = load_metadata_from_id(
                    row["video_ID"], self.folder_globalmetadata_videos
                )

                # create the segment_ID key if not present
                if not "segment_ID" in metadata_video["segments"]:
                    metadata_video["segments"]["segment_ID"] = [
                        ""
                        for _ in range(len(metadata_video["segments"]["frame_start"]))
                    ]

                # add the segment ID
                metadata_video["segments"]["segment_ID"][row["segment_index"]] = row[
                    "segment_ID"
                ]

                # validate the video metadata and save
                is_valid = _validate_metadata_video_dict(metadata_video)

                if is_valid:
                    save_json(metadata_video, file_path)
                else:
                    (
                        print(
                            f"Warning: segment ID info could not be added to video {row['video_ID']} from segment {row['segment_index']}"
                        )
                        if verbose
                        else None
                    )

            except Exception as e:
                (
                    print(
                        f"Error {e} processing segment: {row['segment_ID']} - video: {row['video_ID']} - segment index {row['segment_index']}"
                    )
                    if verbose
                    else None
                )
                continue


    def define_valid_frames(self, recording: str | list[str] | None = None) -> None:
        """Define per-trial valid frames from video/response metadata.

        The value is the minimum between ``video.valid_frames`` and
        ``response.valid_frames``. Results are stored in ``self.trials_df`` under
        the ``valid_frames`` column.

        Parameters
        ----------
        recording : str or list[str] or None, optional
            Recordings to process. If ``None``, process ``self.recording``.
        """

        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]

        if "valid_frames" not in self.trials_df.columns:
            self.trials_df["valid_frames"] = pd.NA
        if "valid_frames_video" not in self.trials_df.columns:
            self.trials_df["valid_frames_video"] = pd.NA
        if "valid_frames_response" not in self.trials_df.columns:
            self.trials_df["valid_frames_response"] = pd.NA

        print_title("Defining valid frames for the videos and responses ", verbose)
        
        for rec in recording:
            trials_df_rec = self.filter_trials(recording=rec)
            iterator = tqdm(
                trials_df_rec.iterrows(),
                total=len(trials_df_rec),
                desc=f"Recording {rec}",
                unit="trial",
                disable=False,
            )
            for _, row in iterator:
                trial = row["trial"]
                rec_row = row["recording"]
                video = self.load_video_by_trial(rec_row, trial, load_metadata_from_global_video=False, load_metadata_from_dataframe=False)
                resp = self.load_response_by_trial(rec_row, trial, load_metadata_from_global_video=False, load_metadata_from_dataframe=False)
                video_valid = getattr(video, "valid_frames", None)
                resp_valid = getattr(resp, "valid_frames", None)
                if video_valid is None or resp_valid is None:
                    valid_frames = pd.NA
                else:
                    valid_frames = min(video_valid, resp_valid)
                idx = (self.trials_df["recording"] == rec_row) & (self.trials_df["trial"] == trial)
                self.trials_df.loc[idx, "valid_frames"] = valid_frames
                self.trials_df.loc[idx, "valid_frames_video"] = video_valid
                self.trials_df.loc[idx, "valid_frames_response"] = resp_valid

    def save_trials_metadata(self, 
                             recording: str | list[str] | None = None, 
                             output_subfolder: str | None = None, 
                             verbose: bool = True) -> None:

        if recording is None:
            recording = self.recording

        if isinstance(recording, str):
            recording = [recording]

        if output_subfolder is None:
            output_subfolder = self._trials_metadata_subfolder

        for rec in recording:
            df_meta_trials_rec = self.trials_df[self.trials_df["recording"] == rec].copy()
            folder_recording_meta = Path(self.folder_metadata) / rec / output_subfolder
            folder_recording_meta.mkdir(parents=True, exist_ok=True)
            filename = folder_recording_meta / f"meta-trials_{rec}.csv"
            df_meta_trials_rec.to_csv(filename, index=False)
            print(f"Saved: {filename}") if verbose else None

