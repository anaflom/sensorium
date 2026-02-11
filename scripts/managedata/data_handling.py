
import numpy as np
import os
import json
from pathlib import Path
import warnings



def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())  # recursively convert elements if it's an array of objects
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # converts numpy scalar → Python scalar
    else:
        return obj


def save_json(metadata, full_file_name):
    metadata = to_json_safe(metadata)
    with open(full_file_name, "w") as f:
        json.dump(metadata, f, indent=4)


def load_all_data(recording_folder, what_data, data_slice=None):

    path_to_data = os.path.join(recording_folder, "data", what_data)
    if not os.path.exists(path_to_data):
            raise ValueError(f"Path does not exist: {path_to_data}")

    # Get sorted list of .npy files only
    data_files_list = sorted([f for f in os.listdir(path_to_data) if f.endswith('.npy')])

    if len(data_files_list) == 0:
        raise ValueError(f"No .npy files found in {path_to_data}")
        
       
    data_all = []
    for file in data_files_list:
        try:
            data = np.load(os.path.join(recording_folder, "data", what_data, file), mmap_mode="r")
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


def _is_valid_value(value):
    if value is None:
        return False
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item() is not None
    if value == '' or value == 'None' or value == 'none' or value == 'nan' or value == 'NaN' or value == 'NAN':
        return False
    return True


def load_trials_descriptor(trials_descriptor_file, verbose=False):
        
    arr_trials_descriptor = np.load(trials_descriptor_file)
    trials_descriptor = arr_trials_descriptor.tolist()
    valid_trials_descriptor = [v for v in trials_descriptor if _is_valid_value(v)]
    
    if verbose and len(valid_trials_descriptor)!=len(trials_descriptor):
        print(f'Total valid trials: {len(valid_trials_descriptor)}') 
        print("Excluded values: " + ", ".join(f'"{x}"' for x in sorted(set(trials_descriptor)-set(valid_trials_descriptor))))
    
    return valid_trials_descriptor


def load_metadata_from_id(id, folder):
    file_pattern = f"*-{id}.json"
    files = list(Path(folder).glob(file_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No file matches {file_pattern} in {folder}")
    if len(files) > 1:
        raise ValueError(f"Multiple files ({len(files)}) match {file_pattern} in {folder}")

    with open(files[0], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata, files[0]







def check_data_integrity(the_path_to_recording, verbose=True):

    path_to_data = os.path.join(the_path_to_recording, "data")
    path_to_meta = os.path.join(the_path_to_recording, "meta")

    info = {}
    data_ok = True


    if not os.path.exists(path_to_data):
        warnings.warn(f"Warning: Path does not exist: {path_to_data}")
        data_ok = False
        info['n_trials'] = None
        info['trials'] = None
        info['samples_per_trial'] = None
        info['n_neurons'] = None
        info['trial_type'] = None 
    
    else:
        n_trials = {}
        data_shape = {}
        samples_per_trial = {}
        the_trials = {}
        n_neurons = []
        for what_data in ['responses', 'videos', 'behavior','pupil_center']:
            path_to_whatdata = os.path.join(path_to_data, what_data)
            if not os.path.exists(path_to_whatdata):
                warnings.warn(f"Warning: Path does not exist: {path_to_whatdata}")
                data_ok = False
                samples_per_trial[what_data] = None
                n_trials[what_data] = 0
                the_trials[what_data] = []
                continue
            files = list(Path(path_to_whatdata).glob("*.npy"))
            if len(files) == 0:
                warnings.warn(f"Warning: No .npy files found in {path_to_whatdata}")
                data_ok = False
                samples_per_trial[what_data] = None
                n_trials[what_data] = 0
                the_trials[what_data] = []
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
                            warnings.warn(f"Warning: Different data shapes across {what_data} files in {path_to_data}: {data.shape} vs {data_shape[what_data]}")
                            data_ok = False
                        if data.shape[-1] != samples_per_trial[what_data]:
                            warnings.warn(f"Warning: Different number of samples per trial across {what_data} files in {path_to_data}: {data.shape[-1]} vs {samples_per_trial[what_data]}")
                            data_ok = False
                    if what_data=='responses':
                        n_neurons.append(data.shape[0])
                except Exception as e:
                    warnings.warn(f"Warning: Could not load {fff}: {e}")
                    data_ok = False

        if len(set(n_trials.values()))>1:
            warnings.warn(f"Warning: Different number of trials across data types in {path_to_data}: {n_trials}")
            data_ok = False
            n_trials = None
        else:
            n_trials = set(n_trials.values()).pop()
        
        if not all(s == the_trials['responses'] for s in the_trials.values()):
            warnings.warn(f"Warning: Different trial files across data types in {path_to_data}")
            data_ok = False
            the_trials = None
        else:
            the_trials = sorted(the_trials['responses'])

        if len(set(samples_per_trial.values()))>1:
            warnings.warn(f"Warning: Different number of samples per trial across data types in {path_to_data}: {samples_per_trial}")
            data_ok = False
            samples_per_trial = None
        else:
            samples_per_trial = set(samples_per_trial.values()).pop()

        unique_n_neurons = set(n_neurons)
        if len(unique_n_neurons)==0:
            data_ok = False
            n_neurons = None
        elif len(unique_n_neurons)>1:
                warnings.warn(f"Warning: Different number of neurons across response files in {path_to_data}: {unique_n_neurons}")
                data_ok = False
                n_neurons = None
        else:
            n_neurons = unique_n_neurons.pop()

        # save some information
        info['n_trials'] = n_trials
        info['trials'] = the_trials
        info['samples_per_trial'] = samples_per_trial
        info['n_neurons'] = n_neurons


    # check information in meta folder for the trials description
    trials_meta_ok = True

    if not os.path.exists(path_to_meta):
        warnings.warn(f"Warning: Path does not exist: {path_to_meta}")
        trials_meta_ok = False
    else:
        if info['n_trials']:
            file_path = Path(path_to_meta) / 'trials' / 'tiers.npy'
            if not file_path.exists():
                warnings.warn(f"No trial description file was founs in {file_path}, description will be set to None")
                info['trial_type'] = None
                trials_meta_ok = False
                
            trials_description = load_trials_descriptor(file_path, verbose=False)
            if len(trials_description)!=info['n_trials']:
                raise ValueError(f"Wrong number of trials ({info['n_trials']}) and descriptors ({len(trials_description)})")
            info['trial_type'] = trials_description
        
    # check information in meta folder for the neurons
    info['neurons'] = {}
    neurons_meta_ok = True
    
    neurons_coord_path = Path(path_to_meta) / 'neurons' / 'cell_motor_coordinates.npy'
    if not neurons_coord_path.exists():
        warnings.warn(f"No neurons coordinate file was founs in {neurons_coord_path}, coordinates are set to None")
        info['neurons']['coord'] = None
        neurons_meta_ok = False
    else:
        try:
            neurons_coord = np.load(neurons_coord_path)
            if neurons_coord.shape[0]!=info['n_neurons']:
                info['neurons']['coord'] = None
                warnings.warn(f"The coordinates file has {neurons_coord.shape[0]} neurons but {info['n_neurons']} neurons were detected in the data, coordinates are set to None")
                neurons_meta_ok = False
            else:
                info['neurons']['coord'] = neurons_coord
        except Exception as e:
            info['neurons']['coord'] = None
            neurons_meta_ok = False
            warnings.warn(f"Could not load neurons coordinates, error {e}, coordinates are set to None")
        
    neurons_ids_path = Path(path_to_meta) / 'neurons' / 'unit_ids.npy'
    if not neurons_ids_path.exists():
        warnings.warn(f"No neurons IDs file was founs in {neurons_ids_path}, IDs are set to None")
        info['neurons']['IDs'] = None
        neurons_meta_ok = False
    else:
        try:
            neurons_ids = np.load(neurons_ids_path)
            if neurons_ids.shape[0]!=info['n_neurons']:
                info['neurons']['IDs'] = None
                neurons_meta_ok = False
                warnings.warn(f"The IDs file has {neurons_ids.shape[0]} neurons but {info['n_neurons']} neurons were detected in the data, IDs are set to None")
            else:
                info['neurons']['IDs'] = neurons_ids
        except Exception as e:
            neurons_meta_ok = False
            info['neurons']['IDs'] = None
            warnings.warn(f"Could not load neurons IDs, error {e}, IDs are set to None")
        
    # update the varaible holding wheter data is ok for recording
    all_fine = data_ok and neurons_meta_ok and trials_meta_ok
           
    return all_fine, info


