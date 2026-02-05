
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from pathlib import Path

from skimage.metrics import mean_squared_error

import math





def pick_key_frames(segments_frame_start, segments_frame_end, frames_per_segment=None, distance_between_key_frames=None):
    '''
    This function picks some frames to compare the videos
    '''
    if (frames_per_segment is None or isinstance(frames_per_segment, int))==False:
        raise ValueError("frames_per_segment must be either None or an integer")
    if (distance_between_key_frames is None or isinstance(distance_between_key_frames, int))==False:
        raise ValueError("distance_between_key_frames must be either None or an integer")
    if (distance_between_key_frames is None) and (frames_per_segment is None):
        raise ValueError("at least one of distance_between_key_frames or frames_per_segment must be an integer")
    
    # pick the frames
    key_frames  = []
    for start_frame, end_frame in zip(segments_frame_start, segments_frame_end):
        if isinstance(frames_per_segment, int) and distance_between_key_frames is None:
            new_frames = np.linspace(start_frame, end_frame, frames_per_segment + 2)[1:-1]
        elif isinstance(distance_between_key_frames, int) and frames_per_segment is None:
            new_frames = np.arange(start_frame+distance_between_key_frames, end_frame, distance_between_key_frames)
        elif isinstance(frames_per_segment, int) and isinstance(distance_between_key_frames, int):
            new_frames_a = np.linspace(start_frame, end_frame, frames_per_segment + 2)[1:-1]
            new_frames_b = np.arange(start_frame+distance_between_key_frames, end_frame, distance_between_key_frames)
            if len(new_frames_a)>=len(new_frames_b):
                new_frames = new_frames_a
            else:
                new_frames = new_frames_b
        new_frames = np.rint(new_frames).astype(int)
        key_frames.extend(list(new_frames))
        
    return np.asarray(key_frames)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # converts numpy scalar → Python scalar
    else:
        return obj

def load_all_data(recording_folder, what):

    data_files_list = os.listdir(os.path.join(recording_folder, "data", what))
    data_all = []
    for file in data_files_list:
        data = np.load(os.path.join(recording_folder, "data", what, file))
        data_all.append(data)
        
    return np.array(data_all)


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
    return metadata


def compute_neurons_stats(recording_folder, trials_to_include=None):
    '''
    Computes descriptive statistics for each neuron activation pattern across all
    trials belonging to certain conditions.

    :param recording_folder: path to the folder with the recording
    :param trials_to_include: list with the possible values the trials to include
                              can have. If None, all trials are included
    '''
    # load all responses
    print("Loading all responses...")
    resp_all = load_all_data(recording_folder, 'responses')
    
    # compute responses statistics per neuron
    stats = {}
    stats['mean'] = np.full(resp_all.shape[1], np.nan)
    stats['std'] = np.full(resp_all.shape[1], np.nan)
    stats['median'] = np.full(resp_all.shape[1], np.nan)
    stats['min'] = np.full(resp_all.shape[1], np.nan)
    stats['max'] = np.full(resp_all.shape[1], np.nan)
    if trials_to_include!=None:
        # load the trials descriptor
        trials = load_trials_descriptor(recording_folder)
        # select the trials
        idx_trials_stats = np.full(resp_all.shape[0], False)
        for c in trials_to_include:
            idx_trials_stats = np.logical_or(trials==c, idx_trials_stats)
        print("Included trials values: " + ", ".join(f'"{x}"' for x in sorted(set(trials[idx_trials_stats]))))
        print("Excluded trials values: " + ", ".join(f'"{x}"' for x in sorted(set(trials) - set(trials[idx_trials_stats]))))
    else:
        idx_trials_stats = np.full(resp_all.shape[0], True)
    print(f"Computing neurons stats over {np.sum(idx_trials_stats)} out of {resp_all.shape[0]} total trials")
    print(f"Computing for {resp_all.shape[1]} neurons...")
    for ni in tqdm(range(resp_all.shape[1])):
        stats['mean'][ni] = np.nanmean(resp_all[idx_trials_stats,ni,:])
        stats['std'][ni] = np.nanstd(resp_all[idx_trials_stats,ni,:])
        stats['median'][ni] = np.nanmedian(resp_all[idx_trials_stats,ni,:])
        stats['min'][ni] = np.nanmin(resp_all[idx_trials_stats,ni,:])
        stats['max'][ni] = np.nanmax(resp_all[idx_trials_stats,ni,:])

    return pd.DataFrame.from_dict(stats)






def set_parameter_value(val_default, val_optional):
    if val_optional==None:
        val = val_default
    else:
        val = val_optional
    return val
            
def find_outliers(y, threshold=2):
    '''
    This function finds outliers in a 1-D array based on the number of standard
    deviations from the mean.

    y : 1-D array
    threshold : number of standard deviations to define outliers

    returns
    idx_outlier : boolean indexes indicating the outliers
    '''
    # remove nans
    y_ = y[np.isnan(y)==False]
    # define outliers 
    sd = np.std(y_)
    thresh_l = np.mean(y_) - threshold*np.std(y_)
    thresh_u = np.mean(y_) + threshold*np.std(y_)
    idx_outlier = np.logical_or(y<thresh_l, y>thresh_u)
    return idx_outlier


def remove_outliers(y, threshold=2):
    '''
    Remove outliers from a 1-D array based on standard-deviation thresholding.

    y : 1-D array
    threshold : number of standard deviations to define outliers

    returns
    an array without the outliers
    '''
    idx_outlier = find_outliers(y, threshold=threshold)
    return y[idx_outlier==False]


def find_edges(x, max_transition_frames, limit, revert=False):
    '''
    Determine whether the first/last up to `max_transition_frames` elements of
    array `x` are transition frames. If so, return how many edge frames can be
    discarded so that the remaining values are below `limit`.

    x : 1-D array
    max_transition_frames : maximum number of transition frames (2*max_transition_frames
                            must be < len(x))
    limit : threshold limit to consider
    revert : False uses the array as-is; True flips the array before analysis

    returns
    n_first : the number of edge frames to discard from the beginning (or end
              if `revert` is True) so that the remainder is < limit
    '''
    
    if revert:
        x = np.flip(x)

    n_first = 0
    if len(x)>2*max_transition_frames:

        max_change = np.max(x[max_transition_frames:-max_transition_frames])
        if max_change<limit:
            n_first = max_transition_frames
            while n_first>=0:
                max_change = np.max(x[n_first:-max_transition_frames])
                if max_change<limit:
                    n_first-=1
                else:
                    break
            n_first = n_first+1

    return n_first



def select_peaks(peaks, priority, distance):
    """Select peaks such that kept peaks are at least ``distance`` apart.

    The function keeps peaks in order of descending ``priority``: higher
    priority peaks are preferred and lower priority peaks within ``distance``
    of a kept peak are discarded.

    Parameters
    ----------
    peaks : array-like
        1-D array of peak indices (integers). Can be unsorted.
    priority : array-like
        1-D array of same length as ``peaks`` containing numeric priority
        values. Higher numeric value -> higher priority.
    distance : float or int
        Minimal required distance between two kept peaks. Non-positive values
        mean no distance constraint.

    Returns
    -------
    keep : ndarray(bool)
        Boolean mask of same length as ``peaks`` indicating which peaks are
        kept (True) or discarded (False).
    """

    peaks = np.asarray(peaks)
    priority = np.asarray(priority)

    if peaks.ndim != 1 or priority.ndim != 1:
        raise ValueError("`peaks` and `priority` must be 1-D arrays")
    if peaks.shape[0] != priority.shape[0]:
        raise ValueError("`peaks` and `priority` must have the same length")

    n = peaks.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # No distance constraint -> keep all
    if distance is None or distance <= 0:
        return np.ones(n, dtype=bool)

    # Round up to integer distance (indices are integer positions)
    min_dist = int(math.ceil(distance))

    # Work on peaks sorted by position so neighbour checks are straightforward
    pos_order = np.argsort(peaks)
    peaks_pos = peaks[pos_order]
    priority_pos = priority[pos_order]

    # Keep flags in position-sorted ordering
    keep_pos = np.ones(n, dtype=bool)

    # Process peaks in descending priority; ties broken by keeping earlier (lower index)
    priority_order = np.argsort(priority_pos)
    for idx in priority_order[::-1]:
        if not keep_pos[idx]:
            continue
        # Remove earlier peaks that are too close
        k = idx - 1
        while k >= 0 and (peaks_pos[idx] - peaks_pos[k]) < min_dist:
            keep_pos[k] = False
            k -= 1
        # Remove later peaks that are too close
        k = idx + 1
        while k < n and (peaks_pos[k] - peaks_pos[idx]) < min_dist:
            keep_pos[k] = False
            k += 1

    # Map back to original ordering
    keep = np.zeros(n, dtype=bool)
    # pos_order maps position-index -> original index, so iterate
    for pos_idx, orig_idx in enumerate(pos_order):
        keep[orig_idx] = keep_pos[pos_idx]

    return keep



def find_peaks(y, window, distance=None, threshold=3, relative_threshold=True, min_thresh=4, threshold_outliers=2):
    '''
    Find peaks in a one-dimensional array where values are larger than their
    neighboring samples by a threshold.

    y : 1-D array
    window : integer denoting the number of samples to take on both sides
             around a sample to test whether it is a peak
    distance : minimum peak distance
    threshold : either the absolute threshold (if relative_threshold is False)
                or the multiplier of the standard deviation used to determine
                the threshold (if relative_threshold is True)
    relative_threshold : whether to use relative thresholds
    min_thresh : minimum threshold used when computing relative thresholds
    threshold_outliers : threshold used to remove extreme values in the window

    returns
    peaks : indexes indicating the positions of the peaks in y
    '''

    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')
                    
    # loop over samples checking if they are peaks
    peaks= []
    for i in range(len(y)):

        if not np.isnan(y[i]):

            # get the indexes to take the data before and after i
            idx_pre = np.arange(i-window, i)
            idx_pre = idx_pre[np.logical_and(idx_pre>=0, idx_pre<len(y))]
            idx_post = np.arange(i+1, i+1+window,1)
            idx_post = idx_post[np.logical_and(idx_post>=0, idx_post<len(y))]

            if len(idx_pre)>0 and len(idx_post)>0:
            
                # compute thresholds pre and post
                if relative_threshold:
                    
                    # define the theshold based on previous samples  
                    y_pre = y[idx_pre]
                    if threshold_outliers is not None:
                        y_pre = remove_outliers(y_pre, threshold=threshold_outliers)
                    else:
                        y_pre = y_pre
                    if len(y_pre)>2:
                        threshold_pre = threshold*np.std(y_pre)
                        threshold_pre = max(threshold_pre, min_thresh)
                    else:
                        threshold_pre = None
                    
                    # define the theshold based on posterior samples
                    y_post = y[idx_post]
                    if threshold_outliers is not None:
                        y_post = remove_outliers(y_post, threshold=threshold_outliers)
                    else:
                        y_post = y_post
                    if len(y_post)>2:
                        threshold_post = threshold*np.std(y_post)
                        threshold_post = max(threshold_post, min_thresh)
                    else:
                        threshold_post = None

                else:
                    threshold_pre = threshold
                    threshold_post = threshold

                # define the value relatively to which the limit is estimated
                # y_pre_est = np.mean(y[idx_pre])
                y_pre_est = y[i-1]
                # y_post_est = np.mean(y[idx_post])
                y_post_est = y[i+1]
                    
                # determine if it is peak  
                if threshold_pre is not None:
                    is_pre = y[i] > (y_pre_est + threshold_pre)
                else:
                    is_pre = False
                if threshold_post is not None:
                    is_post = y[i] > (y_post_est + threshold_post)
                else:
                    is_post = False
                if is_pre and is_post:
                    peaks.append(i)

    # conver to array
    peaks = np.array(peaks)    
    
    # remove close peaks
    if distance is not None and len(peaks)>1:
        # keep = _select_by_peak_distance(peaks, np.float64(y[peaks]), np.float64(distance))
        keep = select_peaks(peaks, y[peaks], distance)
        peaks = peaks[keep]

    return peaks


def find_margin(data, limit=0, axis=0, revert=False):

    """
    This function finds the number of pixels that can be taken from the
    selected dimension that have an intensity range lower than `limit`.

    data : 3-D array (last dimension are the frames)
    limit : upper limit to the intensity range within data[:m,:,:]
    axis : the axis to consider
    revert : whether to flip the data along the selected axis first

    returns the number of pixels such that data[:m,:,:] has a range lower than limit
    """
    if axis==1:
        data = np.transpose(data,(1,0,2))
    if revert:
        data = np.flip(data, axis=0)
    m = 0
    while m < np.shape(data)[0]:
        if (np.max(data[:m+1,:,:])-np.min(data[:m+1,:,:]))<=limit:
            m+=1
        else:
            break
    return m


def compute_videos_time_change(data, valid_frames):
    change = np.zeros(valid_frames-1)
    for i in range(valid_frames-1):
        change[i] = mean_squared_error(data[:,:,i+1], data[:,:,i])
    return change

