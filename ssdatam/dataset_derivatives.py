import json
import os
from typing import Self, Any

import numpy as np
import pandas as pd
from pathlib import Path
import copy

from ssdatam.metadata import parse_info_from_recording_name
from ssdatam.data_handling import get_file_with_pattern
from ssdatam.dataset import (DataSet, print_title, filter_dataframe)
from ssdatam.responses import Responses


class DataSetDerivatives(DataSet):

    def __init__(
        self,
        folder_data: str | Path | None = None,
        folder_metadata: str | Path | None = None,
        folder_derivatives: str | Path | None = None,
        recording: list[str] | str | None = None,
        check_data: bool = True,
        check_metadata: bool = True,
        check: bool = True,
        trials_metadata_file_type: str = "csv",
        trials_metadata_subfolder: str = "trials",
        neurons_metadata_subfolder: str = "neurons",
        verbose: bool = True,
    ):

        print_title("Initializing DataSetDerivatives ", verbose)

        if check==False:
            check_data = False
            check_metadata = False

        if trials_metadata_file_type not in {"csv", "json"}:
            raise ValueError(f"Invalid trials_metadata_file_type: {trials_metadata_file_type}, expected 'csv' or 'json'")
        self._trials_metadata_file_type = trials_metadata_file_type

        # path to the derivatives folder
        if folder_derivatives is None:
            raise ValueError("the derivatives folder must be set")
        self.folder_derivatives = Path(folder_derivatives) 
        if not os.path.exists(self.folder_derivatives):
            raise ValueError(f"Path does not exist: {self.folder_derivatives}")
        if recording is None:
            recording = [p.name for p in self.folder_derivatives.iterdir() if p.is_dir()]
        self.recording = recording

        # set data folders and check they exist
        self.folder_data = folder_data
        if self.folder_data is None:
            print(f"folder_data not set. methods requiring it will fail")
        else:
            if not os.path.exists(folder_data):
                raise ValueError(f"Path does not exist: {folder_data}")
        
        # Add info field
        self.add_info()

        # Check the data and store some info about it
        if self.folder_data is not None:
            self.add_data_info()
        if check_data and self.folder_data is not None:
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
        
        # Add some info about the trials from the metadata
        if self.folder_data is None:
            self.get_trialsinfo_from_metadata()

        # load neurons data for all recordings and store it in the info variable
        if self.folder_data is not None:
            self.load_neurons(verbose=verbose)

        # load trial types for all recordings and store it in the info variable
        self.load_trial_types()

        # create a table with basic info about the trials
        self.trials_df = self.get_trials_metadata_basic(verbose=verbose)       

    def add_info(self, verbose: bool = True):
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
            info_data_rec["n_trials"] = None
            info_data_rec["trials"] = None
            info_data_rec["samples_per_trial"] = None
            info_data_rec["n_neurons"] = None
            # store the information about the data for the recording
            self.info[rec] = {**info_data_rec_general, **info_data_rec}

    def get_trialsinfo_from_metadata(self, recording: str = None, verbose: bool = True):
        if recording is None:
            recording = self.recording
        elif isinstance(recording, str):
            recording = [recording]

        if not hasattr(self, "info"):
            self.info = {}

        for rec in recording:
            loaded = False
            if self._good_metadata_per_recording[rec] and self._trials_metadata_file_type=='csv':
                try:
                    path_to_table = os.path.join(self.folder_metadata, rec, self._trials_metadata_subfolder)
                    file = os.path.join(path_to_table, f"meta-trials_{rec}.csv")
                    df = pd.read_csv(file)
                    if "trial" in df.columns:
                        if rec not in self.info:
                            self.info[rec] = {}
                        self.info[rec]["trials"] = df["trial"].astype(str).to_numpy()
                        self.info[rec]["n_trials"] = len(self.info[rec]["trials"])
                except Exception as e:
                    print(
                        f"Error loading info for recording {rec} from metadata: {e}."
                    ) if verbose else None
                


    def load_derivativedata_by(
        self,
        verbose: bool = True,
        query: str | None = None,
        **conditions,
    ) -> tuple[list[np.ndarray], pd.DataFrame]:
        """Load responses for all trials matching filters.

        Returns
        -------
        tuple[list[np.ndarray], pandas.DataFrame]
            Loaded objects and the filtered trial table.
        """

        if self.folder_derivatives is None:
            raise ValueError("folder_derivatives must be set to load the grid.")
        
        trials_df = self.filter_trials(**conditions, query=query)
        data_der = []
        for index, row in trials_df.iterrows():
            
            # find a file in the derivatives folder matching the recording and trial and load it
            recording = row["recording"]
            trial = row["trial"]
            file_pattern  = f"*rec-{recording}_trial-{trial}.npy"
            folder = self.folder_derivatives / recording / "trials"
            file_path = get_file_with_pattern(file_pattern, folder)
            if file_path is None:
                raise ValueError(f"No file found in {folder} matching pattern {file_pattern}")
            act = np.load(file_path)
            data_der.append(act)

        return data_der, trials_df