import json
import os
from typing import Self, Any

import numpy as np
import pandas as pd
from pathlib import Path
import copy

from utils.data_handling import get_file_with_pattern
from utils.dataset import (DataSet, print_title, filter_dataframe)
from utils.responses import Responses


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