
import numpy as np
import os
import json
from pathlib import Path



def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # converts numpy scalar → Python scalar
    else:
        return obj


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


def load_trials_descriptor(recording_folder, verbose=False):
    
    trials_possible_values = ['train','oracle','live_test_main','live_test_bonus','final_test_main','final_test_bonus']
    
    trials = np.load(os.path.join(recording_folder, "meta", "trials", "tiers.npy"))
    if verbose:
        print(f'Total trials: {len(trials)}')
    
    # keep only trials with certain values
    if verbose:
        print("Trials existing values: " + ", ".join(f'"{x}"' for x in sorted(set(trials))))
        print("Trials possible values: " + ", ".join(f'"{x}"' for x in sorted(set(trials_possible_values))))
    idx_trials_valid = np.full(trials.shape[0], False)
    for c in trials_possible_values:
        idx_trials_valid = np.logical_or(trials==c, idx_trials_valid)
    trials_valid = trials[idx_trials_valid]
    if verbose:
        print("Excluded values: " + ", ".join(f'"{x}"' for x in sorted(set(trials)-set(trials_valid))))
        print(f'Valid trials: {len(trials_valid)}')
    
    return trials_valid


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

def save_json(metadata, full_file_name):
    metadata = to_json_safe(metadata)
    with open(full_file_name, "w") as f:
        json.dump(metadata, f, indent=4)



