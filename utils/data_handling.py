# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import json
from pathlib import Path
from typing import Any


def to_json_safe(obj: Any) -> Any:
    """Recursively convert NumPy objects into JSON-serializable Python types.

    Parameters
    ----------
    obj : Any
        Input object potentially containing NumPy arrays/scalars.

    Returns
    -------
    Any
        JSON-serializable representation of ``obj``.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return to_json_safe(
            obj.tolist()
        )  # recursively convert elements if it's an array of objects
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # converts numpy scalar → Python scalar
    else:
        return obj


def save_json(metadata: dict, full_file_name: str | Path) -> None:
    """Save metadata dictionary to JSON file.

    Parameters
    ----------
    metadata : dict
        Metadata content.
    full_file_name : str or pathlib.Path
        Output JSON path.
    """
    metadata = to_json_safe(metadata)
    with open(full_file_name, "w") as f:
        json.dump(metadata, f, indent=4)


def load_all_data(
    recording_folder: str | Path,
    what_data: str,
    data_slice: slice | tuple | None = None,
) -> np.ndarray:
    """Load and stack all ``.npy`` files for a data type.

    Parameters
    ----------
    recording_folder : str or pathlib.Path
        Recording root folder.
    what_data : str
        Data subfolder inside ``recording_folder/data``.
    data_slice : slice or tuple or None, optional
        Optional slice applied to each loaded array.

    Returns
    -------
    numpy.ndarray
        Stacked data array with trial dimension first.
    """

    path_to_data = os.path.join(recording_folder, "data", what_data)
    if not os.path.exists(path_to_data):
        raise ValueError(f"Path does not exist: {path_to_data}")

    # Get sorted list of .npy files only
    data_files_list = sorted(
        [f for f in os.listdir(path_to_data) if f.endswith(".npy")]
    )

    if len(data_files_list) == 0:
        raise ValueError(f"No .npy files found in {path_to_data}")

    data_all = []
    for file in data_files_list:
        try:
            data = np.load(
                os.path.join(recording_folder, "data", what_data, file), mmap_mode="r"
            )
            if data_slice is None:
                d = data
            else:
                d = data[data_slice]
            data_all.append(d)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
            continue

    if len(data_all) == 0:
        raise ValueError(f"No data successfully loaded from {path_to_data}")

    return np.stack(data_all, axis=0)


def _is_valid_value(value: Any) -> bool:
    """Return whether a trial descriptor value is valid."""
    if value is None:
        return False
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item() is not None
    if (
        value == ""
        or value == "None"
        or value == "none"
        or value == "nan"
        or value == "NaN"
        or value == "NAN"
    ):
        return False
    return True

def load_trials_descriptor(
    trials_descriptor_file: str | Path, verbose: bool = False
) -> list:
    """Load and clean trial descriptors from ``tiers.npy``.

    Parameters
    ----------
    trials_descriptor_file : str or pathlib.Path
        Path to descriptor ``.npy`` file.
    verbose : bool, default=False
        If ``True``, print summary of excluded values.

    Returns
    -------
    list
        Valid descriptor entries.
    """

    loaded_trials_descriptor = np.load(trials_descriptor_file)
    loaded_trials_descriptor = loaded_trials_descriptor.tolist()
    trials_descriptor = {str(i): v for i, v in enumerate(loaded_trials_descriptor) if _is_valid_value(v)}

    if verbose and len(trials_descriptor.keys()) != len(loaded_trials_descriptor):
        print(f"Total valid trials: {len(trials_descriptor)}")
        
    return trials_descriptor

def load_trials_descriptor_old(
    trials_descriptor_file: str | Path, verbose: bool = False
) -> list:
    """Load and clean trial descriptors from ``tiers.npy``.

    Parameters
    ----------
    trials_descriptor_file : str or pathlib.Path
        Path to descriptor ``.npy`` file.
    verbose : bool, default=False
        If ``True``, print summary of excluded values.

    Returns
    -------
    list
        Valid descriptor entries.
    """

    arr_trials_descriptor = np.load(trials_descriptor_file)
    trials_descriptor = arr_trials_descriptor.tolist()
    valid_trials_descriptor = [v for v in trials_descriptor if _is_valid_value(v)]

    if verbose and len(valid_trials_descriptor) != len(trials_descriptor):
        print(f"Total valid trials: {len(valid_trials_descriptor)}")
        print(
            "Excluded values: "
            + ", ".join(
                f'"{x}"'
                for x in sorted(set(trials_descriptor) - set(valid_trials_descriptor))
            )
        )

    return valid_trials_descriptor


def get_file_with_pattern(file_pattern: str, folder: str | Path, verbose: bool = True):

    files = list(Path(folder).glob(file_pattern))
    if len(files) == 0:
        print(f"No file matches {file_pattern} in {folder}") if verbose else None
        return None
    if len(files) > 1:
        print(
            f"Multiple files ({len(files)}) match {file_pattern} in {folder}"
        ) if verbose else None
        return None
    return files[0]


def load_metadata_json_to_obj(obj, 
                    file_metadata: str | Path,
                    attributes_to_check_match: list[str] = ["label", "ID", "valid_frames","sampling_freq"],
                    attributes_to_add: list[str] = None,
                    raise_on_mismatch: bool = True,
                    verbose: bool = True) -> None:
    """Load metadata from JSON file into an object.

    Parameters
    ----------
    file_metadata : str or pathlib.Path
        Path to a JSON file with the metadata.
    attributes_to_check_match : list of str, default=["label", "ID", "valid_frames","sampling_freq"]
        List of attributes to check for consistency between the object and the metadata file. If an attribute in this list is not set in the object, it will be set from the metadata file. If it is already set in the object, it will be checked that it matches the value in the metadata file. If there is a mismatch, a ValueError will be raised.
    attributes_to_add : list of str, optional
        List of additional attributes to add from the metadata file. If None, all attributes not in attributes_to_check_match will be added.
    raise_on_mismatch : bool, default=True
        If True, raise a ValueError if there is a mismatch between the object and the metadata file for any attribute in attributes_to_check_match. If False, just print a warning.
    verbose : bool, default=True
        If True, print warnings.

    """

    try:
        with open(file_metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        if verbose:
            print(f"Warning. load_metadata: Could not load metadata: {e}")

    # check th input attributes
    if attributes_to_add is None:
        attributes_to_add = metadata.keys()
    attributes_to_add = [attr for attr in attributes_to_add if attr not in attributes_to_check_match]

    # for attributes in attributes_to_check_match if they are not set in the object, set them from the metadata, otherwise check they match
    for attr in attributes_to_check_match:
        if attr not in metadata.keys():
            if verbose:
                print(f"Warning. load_metadata: '{attr}' was not found in the metadata file.")
            continue

        if getattr(obj, attr) is None:
            setattr(obj, attr, metadata[attr])
        else:
            if metadata[attr] != getattr(obj, attr):
                if raise_on_mismatch:
                    raise ValueError(
                        f"The metadata file contains a {attr} different from the object"
                    )
                elif verbose:
                    print(f"Warning. load_metadata: '{attr}' mismatch between object and metadata file.")
    
    # add some other metadata
    for attr in attributes_to_add:
        if attr not in metadata.keys():
            if verbose:
                print(f"Warning. load_metadata: '{attr}' was not found in the metadata file.")
        else:
            setattr(obj, attr, metadata[attr])
            if attr == "segments":
                for k in obj.segments.keys():
                    obj.segments[k] = np.asarray(obj.segments[k])

def load_metadata_from_id(id: str, folder: str | Path, verbose: bool = True) -> tuple[dict, Path]:
    """Load one metadata JSON matching an ID pattern.

    Parameters
    ----------
    id : str
        Identifier suffix used in file pattern ``*-{id}.json``.
    folder : str or pathlib.Path
        Metadata folder.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[dict, pathlib.Path]
        Loaded metadata dictionary and matching file path.
    """
    try:
        file = get_file_with_pattern(f"*-{id}.json", folder, verbose=verbose)
    except Exception as e:
        print(f"Warning: Could not get a a file name with error {e}") if verbose else None
        return {}, None
    if file is None:
        return {}, None
    with open(file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata, file


def check_data_integrity(
    path_to_data: str | Path, verbose: bool = True
) -> tuple[bool, dict]:
    """Check integrity and consistency of trial data files.

    Parameters
    ----------
    path_to_data : str or pathlib.Path
        Path to recording ``data`` directory.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[bool, dict]
        Global validity flag and summary information dictionary.
    """

    info = {}
    data_ok = True

    if not os.path.exists(path_to_data):
        print(f"Warning: Path does not exist: {path_to_data}") if verbose else None
        data_ok = False
        n_trials = None
        samples_per_trial = None
        the_trials = None
        n_neurons = None

    else:
        n_trials = {}
        data_shape = {}
        samples_per_trial = {}
        the_trials = {}
        n_neurons = []
        for what_data in ["responses", "videos", "behavior", "pupil_center"]:
            path_to_whatdata = os.path.join(path_to_data, what_data)
            if not os.path.exists(path_to_whatdata):
                (
                    print(f"Warning: Path does not exist: {path_to_whatdata}")
                    if verbose
                    else None
                )
                data_ok = False
                samples_per_trial[what_data] = None
                n_trials[what_data] = 0
                the_trials[what_data] = []
                continue
            files = list(Path(path_to_whatdata).glob("*.npy"))
            if len(files) == 0:
                (
                    print(f"Warning: No .npy files found in {path_to_whatdata}")
                    if verbose
                    else None
                )
                data_ok = False
                samples_per_trial[what_data] = None
                n_trials[what_data] = 0
                the_trials[what_data] = []
                continue

            n_trials[what_data] = len(files)
            the_trials[what_data] = set([Path(f).stem for f in files])
            for fff in files:
                try:
                    data = np.load(fff, mmap_mode="r")
                    if what_data not in data_shape:
                        data_shape[what_data] = data.shape
                        samples_per_trial[what_data] = data.shape[-1]
                    else:
                        if data.shape != data_shape[what_data]:
                            (
                                print(
                                    f"Warning: Different data shapes across {what_data} files in {path_to_data}: {data.shape} vs {data_shape[what_data]}"
                                )
                                if verbose
                                else None
                            )
                            data_ok = False
                        if data.shape[-1] != samples_per_trial[what_data]:
                            (
                                print(
                                    f"Warning: Different number of samples per trial across {what_data} files in {path_to_data}: {data.shape[-1]} vs {samples_per_trial[what_data]}"
                                )
                                if verbose
                                else None
                            )
                            data_ok = False
                    if what_data == "responses":
                        n_neurons.append(data.shape[0])
                except Exception as e:
                    print(f"Warning: Could not load {fff}: {e}") if verbose else None
                    data_ok = False

        if len(set(n_trials.values())) > 1:
            (
                print(
                    f"Warning: Different number of trials across data types in {path_to_data}: {n_trials}"
                )
                if verbose
                else None
            )
            data_ok = False
            n_trials = None
        else:
            n_trials = set(n_trials.values()).pop()

        if not all(s == the_trials["responses"] for s in the_trials.values()):
            (
                print(
                    f"Warning: Different trial files across data types in {path_to_data}"
                )
                if verbose
                else None
            )
            data_ok = False
            the_trials = None
        else:
            the_trials = sorted(the_trials["responses"])

        if len(set(samples_per_trial.values())) > 1:
            (
                print(
                    f"Warning: Different number of samples per trial across data types in {path_to_data}: {samples_per_trial}"
                )
                if verbose
                else None
            )
            data_ok = False
            samples_per_trial = None
        else:
            samples_per_trial = set(samples_per_trial.values()).pop()

        unique_n_neurons = set(n_neurons)
        if len(unique_n_neurons) == 0:
            data_ok = False
            n_neurons = None
        elif len(unique_n_neurons) > 1:
            (
                print(
                    f"Warning: Different number of neurons across response files in {path_to_data}: {unique_n_neurons}"
                )
                if verbose
                else None
            )
            data_ok = False
            n_neurons = None
        else:
            n_neurons = unique_n_neurons.pop()

        # save some information
        info["n_trials"] = n_trials
        info["trials"] = the_trials
        info["samples_per_trial"] = samples_per_trial
        info["n_neurons"] = n_neurons

    return data_ok, info


def check_meta_neurons_integrity(
    path_to_meta_neurons: str | Path, n_neurons: int | None = None, verbose: bool = True
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Check and load neuron coordinates/IDs metadata.

    Parameters
    ----------
    path_to_meta_neurons : str or pathlib.Path
        Path to neurons metadata folder.
    n_neurons : int or None, optional
        Expected neuron count for consistency checks.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    tuple[numpy.ndarray or None, numpy.ndarray or None]
        Neuron coordinates and IDs, or ``None`` if unavailable/invalid.
    """

    # check information in meta folder for the neurons coordinates
    neurons_coord_path = Path(path_to_meta_neurons) / "cell_motor_coordinates.npy"
    if not neurons_coord_path.exists():
        (
            print(
                f"Warning: No neurons coordinate file was found in {neurons_coord_path}, coordinates are set to None"
            )
            if verbose
            else None
        )
        neurons_coord = None
    else:
        try:
            neurons_coord = np.load(neurons_coord_path)
            if n_neurons is not None and neurons_coord.shape[0] != n_neurons:
                coord_count = neurons_coord.shape[0]
                neurons_coord = None
                (
                    print(
                        f"Warning: The coordinates file has {coord_count} neurons but {n_neurons} neurons were detected in the data, coordinates are set to None"
                    )
                    if verbose
                    else None
                )
        except Exception as e:
            neurons_coord = None
            (
                print(
                    f"Warning: Could not load neurons coordinates, error {e}, coordinates are set to None"
                )
                if verbose
                else None
            )

    # check information in meta folder for the neurons IDs
    neurons_ids_path = Path(path_to_meta_neurons) / "unit_ids.npy"
    if not neurons_ids_path.exists():
        (
            print(
                f"Warning: No neurons IDs file was found in {neurons_ids_path}, IDs are set to None"
            )
            if verbose
            else None
        )
        neurons_ids = None
    else:
        try:
            neurons_ids = np.load(neurons_ids_path)
            if n_neurons is not None and neurons_ids.shape[0] != n_neurons:
                ids_count = neurons_ids.shape[0]
                neurons_ids = None
                (
                    print(
                        f"Warning: The IDs file has {ids_count} neurons but {n_neurons} neurons were detected in the data, IDs are set to None"
                    )
                    if verbose
                    else None
                )
        except Exception as e:
            neurons_ids = None
            (
                print(
                    f"Warning: Could not load neurons IDs, error {e}, IDs are set to None"
                )
                if verbose
                else None
            )

    return neurons_coord, neurons_ids

def check_meta_trials_integrity(
    path_to_meta_trials: str | Path, trials: list | None = None, verbose: bool = True
) -> list | None:
    """Check and load trial descriptor metadata.

    Parameters
    ----------
    path_to_meta_trials : str or pathlib.Path
        Path to trials metadata folder.
    trials : list or None, optional
        Expected trial list for consistency checks.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    list or None
        Trial descriptor list if valid, otherwise ``None``.
    """

    # check information in meta folder for the trials description

    file_path = Path(path_to_meta_trials) / "tiers.npy"
    if not file_path.exists():
        (
            print(
                f"Warning: No trial description file was found in {file_path}, description will be set to None"
            )
            if verbose
            else None
        )
        return None

    try:
        trial_type = load_trials_descriptor(file_path, verbose=False)
        if trials:
            if set(trials) != set(trial_type.keys()):
                (
                    print(
                        f"Warning: Different trials detected, description will be set to None"
                    )
                    if verbose
                    else None
                )
                trial_type = None

    except Exception as e:
        (
            print(
                f"Warning: Could not load trial description, error {e}, description will be set to None"
            )
            if verbose
            else None
        )
        return None

    return trial_type

def check_meta_trials_integrity_old(
    path_to_meta_trials: str | Path, n_trials: int | None = None, verbose: bool = True
) -> list | None:
    """Check and load trial descriptor metadata.

    Parameters
    ----------
    path_to_meta_trials : str or pathlib.Path
        Path to trials metadata folder.
    n_trials : int or None, optional
        Expected trial count for consistency checks.
    verbose : bool, default=True
        If ``True``, print warnings.

    Returns
    -------
    list or None
        Trial descriptor list if valid, otherwise ``None``.
    """

    # check information in meta folder for the trials description

    file_path = Path(path_to_meta_trials) / "tiers.npy"
    if not file_path.exists():
        (
            print(
                f"Warning: No trial description file was found in {file_path}, description will be set to None"
            )
            if verbose
            else None
        )
        return None

    try:
        trial_type = load_trials_descriptor(file_path, verbose=False)
        if n_trials:
            if len(trial_type) != n_trials:
                (
                    print(
                        f"Warning: Wrong number of trials ({n_trials}) and descriptors ({len(trial_type)}), description will be set to None"
                    )
                    if verbose
                    else None
                )
                trial_type = None

    except Exception as e:
        (
            print(
                f"Warning: Could not load trial description, error {e}, description will be set to None"
            )
            if verbose
            else None
        )
        return None

    return trial_type
