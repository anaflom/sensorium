import os
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import random

from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error

import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from utils.videos import (Video, VideoID)




def generate_new_id(existing_ids, prefix='v'):
    """Generate a new unique video identifier.

    Parameters
    ----------
    existing_ids : iterable of str
        IDs that are already in use.
    prefix : str, default='v'
        Prefix used when creating IDs.

    Returns
    -------
    str
        Randomly selected ID not present in ``existing_ids``.

    Raises
    ------
    ValueError
        If no IDs are available in the supported ID space.
    """
    all_ids = {f"{prefix}{i:06d}" for i in range(1000000)}
    available_ids = all_ids - set(existing_ids)

    if not available_ids:
        raise ValueError("No IDs left")

    new_id = random.choice(tuple(available_ids))
    return new_id



def same_segments_edges(video_i, video_j, frames_tolerance=2):
    """Check whether two videos have matching segment boundaries.

    Parameters
    ----------
    video_i : Video
        First video object with ``segments`` metadata.
    video_j : Video
        Second video object with ``segments`` metadata.
    frames_tolerance : int, default=2
        Maximum allowed absolute frame difference for start/end boundaries.

    Returns
    -------
    bool
        ``True`` if all segment boundaries match within tolerance.
    """
    
    n_segments_i = len(video_i.segments['frame_start'])
    n_segments_j = len(video_j.segments['frame_start'])
    if n_segments_i!=n_segments_j:
        same_edges = False
    else:
        same_edges_ = np.full(n_segments_i, False)
        for k in range(n_segments_i):
            same_start = np.abs(video_i.segments['frame_start'][k]-video_j.segments['frame_start'][k])<=frames_tolerance
            same_end = np.abs(video_i.segments['frame_end'][k]-video_j.segments['frame_end'][k])<=frames_tolerance
            same_edges_[k] = same_start & same_end
        same_edges = np.all(same_edges_)
    
    return same_edges


def compute_dissimilarity_mse(data_i, data_j, key_frames):
    """Compute mean MSE dissimilarity between two videos on key frames.

    Parameters
    ----------
    data_i : numpy.ndarray
        First video array with shape ``(height, width, n_frames)``.
    data_j : numpy.ndarray
        Second video array with shape ``(height, width, n_frames)``.
    key_frames : array-like
        Frame indices used for comparison.

    Returns
    -------
    float
        Average MSE across valid compared key frames.
    """

    # mse per frame
    dissimilarity = np.full(len(key_frames), np.nan)
    for k, frame in enumerate(key_frames):
        d_i = data_i[:,:,frame]
        d_j = data_j[:,:,frame]
        if (np.all(np.isnan(d_i.flatten())==False)) & (np.all(np.isnan(d_j.flatten())==False)):
            dissimilarity[k] = mean_squared_error(d_i,d_j)

    return np.nanmean(dissimilarity)


def compute_dissimilarity_ssim(data_i, data_j, key_frames, data_range=255):
    """Compute mean SSIM-based dissimilarity between two videos.

    Parameters
    ----------
    data_i : numpy.ndarray
        First video array with shape ``(height, width, n_frames)``.
    data_j : numpy.ndarray
        Second video array with shape ``(height, width, n_frames)``.
    key_frames : array-like
        Frame indices used for comparison.
    data_range : float or int, default=255
        Data range parameter passed to ``structural_similarity``.

    Returns
    -------
    float
        Average SSIM-based dissimilarity ``(1 - SSIM) / 2`` across frames.
    """

    # ssim per frame
    dissimilarity = np.full(len(key_frames), np.nan)
    for k, frame in enumerate(key_frames):
        d_i = data_i[:,:,frame]
        d_j = data_j[:,:,frame]
        if (np.all(np.isnan(d_i.flatten())==False)) & (np.all(np.isnan(d_j.flatten())==False)):
            ssim = structural_similarity(d_i, d_j, data_range=data_range)
            dissimilarity[k] = (1-ssim)/2

    return np.nanmean(dissimilarity)


def compute_dissimilarity_videos(video_i, video_j, dissimilarity_measure='mse'):
    """Compute dissimilarity between two ``Video`` objects.

    Parameters
    ----------
    video_i : Video
        First video object.
    video_j : Video
        Second video object.
    dissimilarity_measure : {'mse', 'ssim'}, default='mse'
        Frame-level dissimilarity function.

    Returns
    -------
    float
        Aggregate dissimilarity over combined key frames.

    Raises
    ------
    ValueError
        If ``dissimilarity_measure`` is not supported.
    """
    
    if dissimilarity_measure=='mse':
        dissimilarity_fun = compute_dissimilarity_mse
    elif dissimilarity_measure=='ssim':
        dissimilarity_fun = compute_dissimilarity_ssim
    else:
        raise ValueError("The dissimilarity_measure can be either mse or ssim")

    # pick some representative frames for each video
    key_frames_i = video_i.pick_key_frames()
    key_frames_j = video_j.pick_key_frames()
    
    # join the key frames
    key_frames = np.unique(np.concatenate((key_frames_i, key_frames_j)))
    max_frame = min(video_i.valid_frames, video_j.valid_frames)
    key_frames = key_frames[key_frames<=max_frame]

    # compute the difference between videos
    dissimilarity = dissimilarity_fun(video_i.data, video_j.data, key_frames)
                   
    return dissimilarity


def compute_dissimilarity_video_list(videos, dissimilarity_measure='mse', check_edges_first=True, frames_tolerance=2):
    """Compute pairwise dissimilarity matrix for a list of videos.

    Parameters
    ----------
    videos : list[Video]
        Sequence of videos to compare.
    dissimilarity_measure : {'mse', 'ssim'}, default='mse'
        Metric used for pairwise dissimilarity.
    check_edges_first : bool, default=True
        If ``True``, only compare videos with matching segment boundaries.
    frames_tolerance : int, default=2
        Segment-boundary tolerance used when ``check_edges_first`` is enabled.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix of pairwise dissimilarity values.
    """

    n_videos = len(videos)
    dissimilarity = np.full((n_videos,n_videos), np.nan)

    for i, video_i in enumerate(tqdm(videos, total=len(videos), desc="Computing dissimilarity", disable=False)):
        
        for j in np.arange(i,n_videos):
            
            # load
            video_j = videos[j]
            
            # check if the segments in which the video is split are the same
            if check_edges_first:
                do_comparison = same_segments_edges(video_i, video_j, frames_tolerance=frames_tolerance)
            else:
                do_comparison = True
            
            # if same segments, compare
            if do_comparison:
                dissimilarity[i,j] = compute_dissimilarity_videos(video_i, video_j, dissimilarity_measure=dissimilarity_measure)

    # fill the lower triangle
    i_lower = np.tril_indices(dissimilarity.shape[0], -1)
    dissimilarity[i_lower] = dissimilarity.T[i_lower]

    return  dissimilarity


def find_equal_sets(mask, elements_names=None):
    """Group connected elements from a boolean adjacency mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Square boolean adjacency matrix.
    elements_names : list or None, optional
        Optional element names aligned with mask rows/columns.

    Returns
    -------
    list[set]
        List of groups, each containing mutually connected elements.
    """

    if elements_names is None:
        elements_names = [i for i in range(mask.shape[0])]

    # find the groups of videos
    n_elements = len(elements_names)
    list_distinct_videos = []
    possible_idxs = set([i for i in range(n_elements)])
    while len(possible_idxs)>0:
        idx_ref = next(iter(possible_idxs))
        idx_same = list(np.where(mask[idx_ref,:])[0])
        if idx_same:
            group_same = set([elements_names[i] for i in idx_same])
            list_distinct_videos.append(group_same)
            possible_idxs = possible_idxs - set(idx_same)
        else:
            possible_idxs.remove(idx_ref)
    
    return list_distinct_videos



def find_equal_sets_scipy(mask, elements_names=None):
    """Group connected elements using SciPy connected components.

    Parameters
    ----------
    mask : numpy.ndarray
        Square boolean adjacency matrix.
    elements_names : list or None, optional
        Optional element names aligned with mask rows/columns.

    Returns
    -------
    list[list]
        Connected-component groups in label order.
    """

    if elements_names is None:
        elements_names = [i for i in range(mask.shape[0])]

    # Convert boolean mask to integer adjacency (True = 1, False = 0), and sparse format
    graph = sp.csr_matrix(mask.astype(int))

    # find the connected elements
    n_components, labels = connected_components(graph, directed=False, return_labels=True)
    
    # create a list of sets with the connected elements
    groups = [[] for _ in range(n_components)]
    for lbl, name in zip(labels, elements_names):
        groups[lbl].append(name)
    
    return groups


def create_table_all_video_ids(folder_globalmetavideos, label=None):
    """Create a table of known unique-video IDs from metadata files.

    Parameters
    ----------
    folder_globalmetavideos : str or pathlib.Path
        Directory containing unique-video metadata JSON files.
    label : str or None, optional
        If provided, restrict to files with this label prefix.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``ID`` and ``label``.
    """
    # create a table with the existing files
    if label is None:
        json_files = glob.glob(os.path.join(folder_globalmetavideos,"*.json"))
    else:
        json_files = glob.glob(os.path.join(folder_globalmetavideos,f"{label}*.json"))
    ids = []
    labels = []
    for filename in json_files:
        basename = os.path.basename(filename)
        parts = os.path.splitext(basename)[0].split('-')
        labels.append(parts[0])
        ids.append(parts[1])
    
    return pd.DataFrame({'ID':ids, 'label':labels})


def compare_with_idvideos(label, list_distinct_videos, folder_videos, folder_metavideos, folder_globalmetavideos, limit_dissimilarity=5):
    """Match duplicate groups to existing IDs or create new unique IDs.

    Parameters
    ----------
    label : str
        Label of the videos being processed.
    list_distinct_videos : list[set[str]]
        Groups of trial names believed to represent the same video content.
    folder_videos : str or pathlib.Path
        Recording folder containing raw trial videos.
    folder_metavideos : str or pathlib.Path
        Directory with per-trial video metadata files.
    folder_globalmetavideos : str or pathlib.Path
        Directory storing unique-video metadata files.
    limit_dissimilarity : float or int, default=5
        Maximum dissimilarity accepted to consider two videos equivalent.

    Returns
    -------
    list[str]
        Assigned unique ID for each input duplicate group.

    Raises
    ------
    Exception
        If one group matches more than one existing unique ID.
    """

    folder_data = os.path.dirname(folder_videos)
    
    # find the IDs for videos of the same label already identified
    df = create_table_all_video_ids(folder_globalmetavideos, label=label)
    same_label_ids = df['ID'].to_list()

    # initialize list holding the IDs
    list_new_ids = []
    for i, duplicate_trials in enumerate(tqdm(list_distinct_videos, total=len(list_distinct_videos), desc="Comparing with existing ID videos", disable=False)):

        # load a video representative from the group to test
        video_file_i = next(iter(duplicate_trials))
        video_i = Video(folder_videos, video_file_i)
        video_i.load_metadata(os.path.join(folder_metavideos, video_file_i+'.json'))
        
        # compare with the unique IDs previously identified 
        equal_to = np.full(len(same_label_ids), False)
        already_included_flags = np.full(len(same_label_ids), False)
        for j, video_id in enumerate(same_label_ids):
            
            # load a representative video for that unique ID
            video_j_id = VideoID(folder_data, folder_globalmetavideos, video_id)

            # check if the video is not already included as a duplicate in the metadata
            if (video_i.recording in video_j_id.duplicates) and (video_i.trial in video_j_id.duplicates[video_i.recording].get('trials', [])):
                already_included_flags[j] = True
                equal_to[j] = True
            else:
                # compare the videos
                dissimilarity_ij = compute_dissimilarity_videos(video_j_id, video_i)
                equal_to[j] = dissimilarity_ij<limit_dissimilarity

        # assign a new ID or the one corresponding to the video found as equivalent
        if np.sum(equal_to)==0:
            
            # find all IDs already used
            df = create_table_all_video_ids(folder_globalmetavideos, label=None)
            all_used_ids = df['ID'].to_list()

            # generate a new id
            the_id = generate_new_id(all_used_ids)
            
            # generate a VideoID object from the exemplar video and add the duplicates
            video_i_id = video_i.copy(deep=True)
            video_i_id.ID = the_id
            video_i_id.add_duplicates( video_i_id.recording, duplicate_trials)
            
            # save a json file with the video metadata
            video_i_id.save_metadata(folder_globalmetavideos, metadata_for='videoID')
            
        elif np.sum(equal_to)==1:
            # get the matching index and id
            j_match = int(np.where(equal_to)[0][0])
            the_id = same_label_ids[j_match]

            # add the duplicate and save again if it wasn't already included
            if not already_included_flags[j_match]:
                video_j_id = VideoID(folder_data, folder_globalmetavideos, the_id)
                video_j_id.add_duplicates(video_i.recording, duplicate_trials)
                video_j_id.save_metadata(folder_globalmetavideos, metadata_for='videoID')

        else:
            raise Exception(f"Something went wrong, group {i} found identical to two existing unique videos")

        list_new_ids.append(the_id)

    return list_new_ids
