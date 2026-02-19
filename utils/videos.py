import os
import numpy as np
import pandas as pd
import copy
import random
import json
import math
from pathlib import Path
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from typing import Any, Self


from utils.data_handling import (load_metadata_from_id, 
                               to_json_safe,
                               )





def display_video_clip(video_tensor: np.ndarray, interval_ms: int = 33) -> HTML:
    """Render a video tensor as an inline HTML animation.

    Parameters
    ----------
    video_tensor : numpy.ndarray
        Video array with shape ``(height, width, n_frames)``.
    interval_ms : int or float, default=33
        Delay between displayed frames in milliseconds.

    Returns
    -------
    IPython.display.HTML
        HTML object containing a JavaScript animation.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    img = ax.imshow(video_tensor[:, :, 0], cmap="gray", animated=True)
    ax.axis("off")

    def update(frame_idx: int) -> tuple[plt.Image,]:
        """Update displayed frame for animation callback."""
        img.set_data(video_tensor[:, :, frame_idx])
        ax.set_title(f"Frame {frame_idx}")
        return (img,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=video_tensor.shape[2],
        interval=interval_ms,
        blit=True,
    )
    plt.close(fig)  # prevent duplicate static plot output
    return HTML(anim.to_jshtml())



def pick_key_frames(segments_frame_start: np.ndarray | list[int], segments_frame_end: np.ndarray | list[int], frames_per_segment: int | None = None, distance_between_key_frames: int | None = None) -> np.ndarray:
    """Select representative frame indices within each segment.

    Parameters
    ----------
    segments_frame_start : array-like
        Start frame indices for segments.
    segments_frame_end : array-like
        End frame indices for segments.
    frames_per_segment : int or None, optional
        Number of uniformly spaced frames per segment.
    distance_between_key_frames : int or None, optional
        Step size between consecutive key frames.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of selected frame indices.

    Raises
    ------
    ValueError
        If both frame-selection parameters are ``None`` or have invalid types.
    """
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


def set_parameter_value(val_default: Any, val_optional: Any) -> Any:
    """Return an optional parameter value with a fallback default.

    Parameters
    ----------
    val_default : Any
        Default value used when ``val_optional`` is ``None``.
    val_optional : Any
        Optional override value.

    Returns
    -------
    Any
        ``val_optional`` if provided; otherwise ``val_default``.
    """
    if val_optional==None:
        val = val_default
    else:
        val = val_optional
    return val
            

def find_outliers(y: np.ndarray, threshold: float | int = 2) -> np.ndarray:
    """Return a boolean mask that flags outliers in a 1-D array.

    Parameters
    ----------
    y : numpy.ndarray
        One-dimensional numeric array.
    threshold : float or int, default=2
        Number of standard deviations from the mean used as cutoff.

    Returns
    -------
    numpy.ndarray
        Boolean mask with ``True`` where values are considered outliers.
    """
    # remove nans
    y_ = y[np.isnan(y)==False]
    # define outliers 
    sd = np.std(y_)
    thresh_l = np.mean(y_) - threshold*np.std(y_)
    thresh_u = np.mean(y_) + threshold*np.std(y_)
    idx_outlier = np.logical_or(y<thresh_l, y>thresh_u)
    return idx_outlier


def remove_outliers(y: np.ndarray, threshold: float | int = 2) -> np.ndarray:
    """Return a 1-D array with outliers removed.

    Parameters
    ----------
    y : numpy.ndarray
        One-dimensional numeric array.
    threshold : float or int, default=2
        Number of standard deviations used to define outliers.

    Returns
    -------
    numpy.ndarray
        Input array filtered to non-outlier elements.
    """
    idx_outlier = find_outliers(y, threshold=threshold)
    return y[idx_outlier==False]


def find_edges(x: np.ndarray, max_transition_frames: int, limit: float | int, revert: bool = False) -> int:
    """Estimate how many transition samples can be trimmed from one edge.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional signal (for example, frame-to-frame changes).
    max_transition_frames : int
        Maximum number of edge samples to consider transition frames.
    limit : float or int
        Upper bound for accepted values after trimming.
    revert : bool, default=False
        If ``True``, evaluate the reversed signal (useful for trailing edge).

    Returns
    -------
    int
        Number of edge samples that can be discarded.
    """
    
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



def select_peaks(peaks: np.ndarray, priority: np.ndarray, distance: float | int) -> np.ndarray:
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



def find_peaks(y: np.ndarray, window: int, distance: int | None = None, threshold: float | int = 3, relative_threshold: bool = True, min_thresh: float | int = 4, threshold_outliers: float | int | None = 2) -> np.ndarray:
    """Detect local peaks in a 1-D signal using windowed thresholding.

    Parameters
    ----------
    y : numpy.ndarray
        One-dimensional signal.
    window : int
        Number of neighbor samples inspected on each side.
    distance : int or None, optional
        Minimum separation between accepted peaks.
    threshold : float or int, default=3
        Absolute or relative threshold value.
    relative_threshold : bool, default=True
        If ``True``, derive thresholds from local standard deviation.
    min_thresh : float or int, default=4
        Lower bound for relative thresholds.
    threshold_outliers : float or int or None, default=2
        Outlier-removal cutoff for local windows.

    Returns
    -------
    numpy.ndarray
        One-dimensional array with peak indices.
    """

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
                    
                    # define the threshold based on previous samples  
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
                    
                    # define the threshold based on posterior samples
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

    # convert to array
    peaks = np.array(peaks)    
    
    # remove close peaks
    if distance is not None and len(peaks)>1:
        # keep = _select_by_peak_distance(peaks, np.float64(y[peaks]), np.float64(distance))
        keep = select_peaks(peaks, y[peaks], distance)
        peaks = peaks[keep]

    return peaks


def find_margin(data: np.ndarray, limit: float | int = 0, axis: int = 0, revert: bool = False) -> int:
    """Return the largest edge width whose intensity range stays below ``limit``.

    Parameters
    ----------
    data : numpy.ndarray
        Video array with shape ``(height, width, n_frames)``.
    limit : float or int, default=0
        Maximum allowed intensity range in the selected edge region.
    axis : int, default=0
        Spatial axis to evaluate (``0`` for rows, ``1`` for columns).
    revert : bool, default=False
        If ``True``, evaluate the edge from the opposite side.

    Returns
    -------
    int
        Number of pixels that satisfy the edge-range criterion.
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


def compute_videos_time_change(data: np.ndarray, valid_frames: int) -> np.ndarray:
    """Compute frame-to-frame mean squared error over valid frames.

    Parameters
    ----------
    data : numpy.ndarray
        Video array with shape ``(height, width, n_frames)``.
    valid_frames : int
        Number of valid frames considered in ``data``.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of MSE values between adjacent frames.
    """
    change = np.zeros(valid_frames-1)
    for i in range(valid_frames-1):
        change[i] = mean_squared_error(data[:,:,i+1], data[:,:,i])
    return change



class Video:

    """Represent a full video and provide segmentation/classification helpers."""
    
    
    def __init__(self, recording_folder: str | Path, trial: str, sampling_freq: float | int = 30, label: str | None = None, ID: str | None = None) -> None:
        """Initialize a ``Video`` from recording folder and trial file.

        Parameters
        ----------
        recording_folder : str or pathlib.Path
            Path to one recording directory containing ``data/videos``.
        trial : str
            Trial name or path to trial file (extension is ignored).
        sampling_freq : float or int, default=30
            Video sampling frequency in Hz.
        label : str or None, optional
            Optional video class label.
        ID : str or None, optional
            Optional unique video identifier.
        """
        trial, ext = os.path.splitext(trial)
        trial = os.path.basename(trial)
        
        self.recording = os.path.basename(recording_folder)
        self.trial = trial
        self.data = np.load(os.path.join(recording_folder, 'data', 'videos', trial+'.npy'))
        
        self.sampling_freq = sampling_freq
        
        n_emptyframes = np.sum(np.all(np.all(np.isnan(self.data),axis=0),axis=0))
        self.valid_frames = np.shape(self.data)[2]-n_emptyframes

        self.label = label
        self.ID = ID
        
        
    def __copy__(self) -> Self:
        """Return a shallow copy of the video object.

        Returns
        -------
        Video
            Shallow-copied video instance.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new


    def __deepcopy__(self, memo) -> Self:
        """Return a deep copy of the video object.

        Parameters
        ----------
        memo : dict
            Internal memoization dictionary used by ``copy.deepcopy``.

        Returns
        -------
        Video
            Deep-copied video instance.
        """
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new
    
    def copy(self, deep=False) -> Self:
        """Copy the video object.

        Parameters
        ----------
        deep : bool, default=False
            If True, return a deep copy. Otherwise return a shallow copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def compute_time_change(self) -> np.ndarray:
        """Compute frame-to-frame changes for the video.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of MSE values for adjacent frames.
        """
        return compute_videos_time_change(self.data, self.valid_frames)
    

    def find_peaks(self, label: str | None = None, window: int | None = None, distance: int | None = None, threshold: float | int | None = None, relative_threshold: bool | None = None, min_thresh: float | int | None = None, threshold_outliers: float | int | None = None) -> None:
        """Detect change peaks and store them in ``self.peaks``.

        Parameters
        ----------
        label : str or None, optional
            Label-specific preset used to fill missing parameters.
        window : int or None, optional
            Number of samples inspected before and after each candidate peak.
        distance : int or None, optional
            Minimum accepted distance between detected peaks.
        threshold : float or int or None, optional
            Absolute or relative threshold used by peak detection.
        relative_threshold : bool or None, optional
            If True, compute threshold relative to local variability.
        min_thresh : float or int or None, optional
            Lower bound applied to relative thresholds.
        threshold_outliers : float or int or None, optional
            Outlier-removal threshold for local windows.

        Notes
        -----
        Missing arguments are populated from class presets, optionally
        conditioned on ``label``.
        """

        # compute the change
        change = self.compute_time_change()

        # define the parameters to find the peaks
        if label==None or label=='unknown':
            window = set_parameter_value(window, 10)
            distance = set_parameter_value(distance, 7)
            threshold = set_parameter_value(threshold, 5)
            relative_threshold = set_parameter_value(relative_threshold, True)
            min_thresh = set_parameter_value(min_thresh, 10)
            threshold_outliers = set_parameter_value(threshold_outliers, 2)
        else:
            match label:
                case 'NaturalVideo':
                    window = set_parameter_value(window, 30)
                    distance = set_parameter_value(distance, 7)
                    threshold = set_parameter_value(threshold, 5)
                    threshold_outliers = set_parameter_value(threshold_outliers, None)      
                    relative_threshold = set_parameter_value(relative_threshold, True)
                    min_thresh = set_parameter_value(min_thresh, 10)
                                 
                case 'RandomDots':
                    window = set_parameter_value(window, 20)
                    distance = set_parameter_value(distance, 45)
                    threshold = set_parameter_value(threshold, 5)
                    threshold_outliers = set_parameter_value(threshold_outliers, 2)                   
                    relative_threshold = set_parameter_value(relative_threshold, True)
                    min_thresh = set_parameter_value(min_thresh, 10)
                    
                case 'NaturalImages':
                    window = set_parameter_value(window, 10)
                    distance = set_parameter_value(distance, 10)
                    threshold = set_parameter_value(threshold, 3)
                    threshold_outliers = set_parameter_value(threshold_outliers, 2)                   
                    relative_threshold = set_parameter_value(relative_threshold, True)
                    min_thresh = set_parameter_value(min_thresh, 10)
                    
                case 'Gabor':
                    window = set_parameter_value(window, 10)
                    distance = set_parameter_value(distance, 20)
                    threshold = set_parameter_value(threshold, 3)
                    threshold_outliers = set_parameter_value(threshold_outliers, 2)                   
                    relative_threshold = set_parameter_value(relative_threshold, True)
                    min_thresh = set_parameter_value(min_thresh, 10)

                case 'PinkNoise':
                    window = set_parameter_value(window, 10)
                    distance = set_parameter_value(distance, 20)
                    threshold = set_parameter_value(threshold, 5)
                    threshold_outliers = set_parameter_value(threshold_outliers, 2)                   
                    relative_threshold = set_parameter_value(relative_threshold, True)
                    min_thresh = set_parameter_value(min_thresh, 10)

                case 'GaussianDot':
                    window = set_parameter_value(window, 5)
                    distance = set_parameter_value(distance, 7)
                    threshold = set_parameter_value(threshold, 3)
                    threshold_outliers = set_parameter_value(threshold_outliers, 2)                   
                    relative_threshold = set_parameter_value(relative_threshold, True)
                    min_thresh = set_parameter_value(min_thresh, 10)

                case _:
                    raise ValueError(f'{label} is not a valid label')

            
        # find the peaks
        peaks = find_peaks(change, window, distance=distance, threshold=threshold, relative_threshold=relative_threshold, min_thresh=min_thresh, threshold_outliers=threshold_outliers)

        # store in the object
        self.peaks = peaks 
        self.n_peaks = len(peaks)
    

    def define_segments(self, frame_start: list[int] | None = None, frame_end: list[int] | None = None) -> dict:
        """Define segment boundaries from peaks or explicit frame ranges.

        Parameters
        ----------
        frame_start : list[int] or None, optional
            Segment start frames. Must be provided together with ``frame_end``
            to bypass peak-based segmentation.
        frame_end : list[int] or None, optional
            Segment end frames (exclusive).

        Returns
        -------
        dict
            Dictionary with ``frame_start``, ``frame_end``, and ``duration``.

        Raises
        ------
        AttributeError
            If peak-based segmentation is requested before calling ``find_peaks``.
        """

        # Determine how to define the segments
        if frame_start==None and frame_end==None:
            set_based_on_peaks = True
        else:
            set_based_on_peaks = False
       
        if set_based_on_peaks:

            # check the peaks have been defined before
            if not hasattr(self, 'peaks'):
                raise AttributeError("The peaks attribute is missing. Run find_peaks before")

            # initialize the segments dictionary
            self.n_segments = self.n_peaks+1
            segments = {'frame_start':np.empty(self.n_segments, dtype=int), 
                        'frame_end': np.empty(self.n_segments, dtype=int), 
                        'duration':np.empty(self.n_segments, dtype=int),
                        }
            
            # peaks initial and last sample
            if self.n_peaks>0:
                previous_peak = 0
                for k, ipeak in enumerate(self.peaks):
                    segments['frame_start'][k] = previous_peak
                    segments['frame_end'][k] = ipeak+1
                    previous_peak = ipeak+1
                segments['frame_start'][k+1] = previous_peak
                segments['frame_end'][k+1] = self.valid_frames
            else:
                segments['frame_start'][0] = 0
                segments['frame_end'][0] = self.valid_frames
        
        else:

            # check the input
            if not isinstance(frame_start, list):
                raise TypeError("frame_start must be a list")
            if not isinstance(frame_end, list):
                raise TypeError("frame_end must be a list")
            if len(frame_start)!=len(frame_end):
                raise ValueError("frame_start and frame_end must have the same length")
            
            # set the segments based on the inputs provided
            self.n_segments = len(frame_start)
            segments = {}
            segments['frame_start']= np.array(frame_start)
            segments['frame_end']= np.array(frame_end)
            segments['duration'] = np.empty(self.n_segments, dtype=int)

        # compute the segments duration
        for idx, (ki,kf) in enumerate(zip(segments['frame_start'], segments['frame_end'])):
            segments['duration'][idx] = kf-ki
             
        # store it into the object
        self.segments = segments

        return segments
    

    def split_long_segments_by_label(self, label: str) -> dict | None:
        """Split long segments when class-specific durations suggest missing peaks.

        Parameters
        ----------
        label : str
            Current video label used to select expected segment duration.

        Returns
        -------
        dict or None
            Updated segment dictionary, or ``None`` when ``label`` is ``'unknown'``.
        """

        # check the segments have been defined before
        if not hasattr(self, 'segments'):
            raise AttributeError("The segments attribute is missing. Run define_segments before")
        
        if label=='unknown':
            print('unknown label, nothing will be done')
            return

        # define the parameters per video label
        match label:
            case 'NaturalVideo':
                segm_n = None
                segm_dur = None 
                segm_dur_tolerance = None                         
            case 'RandomDots':
                segm_n = 4
                segm_dur = 60  
            case 'NaturalImages':
                segm_n = 20
                segm_dur = 15  
            case 'Gabor':
                segm_n = 12
                segm_dur = 25 
            case 'PinkNoise':
                segm_n = 12
                segm_dur = 27 
            case 'GaussianDot':
                segm_n = 35
                segm_dur = 9 
            case _:
                raise ValueError(f'{label} is not a valid label')

        # detect possible segment that should be split   
        segments_new = {}
        segments_new['frame_start'] = self.segments['frame_start'].copy()
        segments_new['frame_end'] = self.segments['frame_end'].copy()
        if segm_dur!=None:
            n_times = np.round(self.segments['duration']/segm_dur).astype(int)
            idx_segm = (n_times>1) & (np.abs(self.segments['duration']-n_times*segm_dur) < 4)
            if np.any(idx_segm):
                add_segment_start = []
                add_segment_end = []
                idx_segm = np.where(idx_segm)[0]
                for k in idx_segm:
                    n_frames_segm = np.round(self.segments['duration'][k]/n_times[k]).astype(int)
                    segments_new['frame_end'][k] = self.segments['frame_start'][k]+n_frames_segm
                    for j in np.arange(1, n_times[k], 1):
                        add_segment_start.append(self.segments['frame_start'][k] + j*n_frames_segm)
                        add_segment_end.append(self.segments['frame_start'][k] + (j+1)*n_frames_segm)
                    add_segment_end[-1] = self.segments['frame_end'][k]
                segments_new['frame_start'] = np.sort(np.concatenate((segments_new['frame_start'], add_segment_start)))
                segments_new['frame_end'] = np.sort(np.concatenate((segments_new['frame_end'], add_segment_end)))
                
        # compute the segments duration
        n_segments = len(segments_new['frame_start'])
        segments_new['duration'] = np.empty(n_segments, dtype=int)
        for idx, (ki,kf) in enumerate(zip(segments_new['frame_start'], segments_new['frame_end'])):
            segments_new['duration'][idx] = kf-ki
            
        # store it into the object
        self.segments = segments_new
        self.n_segments = n_segments
        
        return segments_new

    def apply_method_to_segments(self, method: str) -> None:
        """Run a ``VideoSegment`` method on each segment and store results.

        Parameters
        ----------
        method : str
            Name of a callable ``VideoSegment`` method (for example, ``'describe'``).
        """

        # check the segments have been defined before
        if not hasattr(self, 'segments') or not self.segments:
            raise AttributeError("The segments attribute is missing. Run define_segments before")
        
        # compute for each segment
        for i in range(self.n_segments):
            segm_i = VideoSegment(self, i)
            getattr(segm_i, method)()
            if i==0:
                for key in segm_i.properties.keys():
                    self.segments[key] = np.empty(self.n_segments, dtype=type(segm_i.properties[key]))
            for key in segm_i.properties.keys():
                self.segments[key][i] = segm_i.properties[key]


    def print_segments_table(self) -> pd.DataFrame:
        """Return segment metadata as a pandas table.

        Returns
        -------
        pandas.DataFrame
            Tabular representation of ``self.segments``.
        """
        df = pd.DataFrame(self.segments)
        return df


    def is_gaussiandot(self) -> tuple[bool, np.ndarray]:
        """Evaluate whether segments match the GaussianDot pattern.

        Returns
        -------
        tuple[bool, numpy.ndarray]
            Classification decision and per-segment boolean mask of failures.
        """
        
        # parameters
        n_segments = 35
        n_segments_tolerance = 2
        duration = 9  # expected duration in frames
        duration_tolerance = 1 # tolerance to consider good duration 
        thresh_background_proportion = 0.5 # minimun proprotion of background
        max_segments_not_maching = 2 # maximun number of segments not matching the requierments
        
        # check if the segments are static
        bad_not_static = self.segments['is_static']==False

        # check if the segments have the good duration
        good_duration = np.logical_and( self.segments['duration']>=(duration-duration_tolerance),  self.segments['duration']<=(duration+duration_tolerance))
        bad_duration = good_duration==False

        # check if all have background
        has_background = self.segments['background_proportion']>=thresh_background_proportion
        bad_background = has_background==False

        # count how many segments do not match the requierments
        segments_bad_properties = (bad_not_static | bad_duration | bad_background)
        n_segments_bad_properties = np.sum(segments_bad_properties)

        # compute the difference between the expected number of segments and the obtained
        n_segments_diff = np.abs(self.n_segments-n_segments)
  
        # decide whether it is a GaussianDots
        if (n_segments_diff<=n_segments_tolerance) & (n_segments_bad_properties<=max_segments_not_maching):
            is_gaussian = True
        else:
            is_gaussian = False

        return is_gaussian, segments_bad_properties
    
    
    def is_naturalimages(self) -> tuple[bool, np.ndarray]:
        """Evaluate whether segments match the NaturalImages pattern.

        Returns
        -------
        tuple[bool, numpy.ndarray]
            Classification decision and per-segment boolean mask of failures.
        """

        # parameters
        n_segments = 10*2
        n_segments_tolerance = 2
        duration_black = [12, 18]  # expected duration black images
        duration_image = 15  # expected duration image
        duration_tolerance = 2     # tolerance to consider good duration
        max_segments_not_maching = 4 # maximun number of segments not matching the requierments
             
        # check sequence structure
        if any(self.segments['is_uniform']):

            # check if the segments are static
            bad_not_static = self.segments['is_static']==False

            # find the segments that are a "black" image or a natural image
            is_black = self.segments['is_uniform']
            is_image = self.segments['is_uniform']==False
            
            # check alternation of black image segments
            transition_alternation = is_black[0:-1]==is_black[1:]
            same_next = np.append(transition_alternation, False)
            same_pre = np.insert(transition_alternation, 0, False)
            bad_alternation = same_next | same_pre
            
            # check duration 
            dur_black = self.segments['duration'][is_black]
            dur_image = self.segments['duration'][is_image]
            good_duration_black = np.logical_and( dur_black>=(duration_black[0]-duration_tolerance),  dur_black<=(duration_black[1]+duration_tolerance))
            good_duration_immage = np.logical_and( dur_image>=(duration_image-duration_tolerance),  dur_image<=(duration_image+duration_tolerance))
            bad_duration = np.full(self.n_segments, False)
            bad_duration[np.where(is_black)[0][good_duration_black==False]] = True
            bad_duration[np.where(is_image)[0][good_duration_immage==False]] = True
            
            # count how many segments do not match the requierments
            segments_bad_properties = (bad_not_static | bad_alternation | bad_duration)
            n_segments_bad_properties = np.sum(segments_bad_properties)

            # compute the difference between the expected number of segments and the obtained
            n_segments_diff = np.abs(self.n_segments-n_segments)
  
            # decide whether it is a NaturalImages
            if (n_segments_diff<=n_segments_tolerance) & (n_segments_bad_properties<=max_segments_not_maching):
                is_naturalimage = True
            else:
                is_naturalimage = False
        
        else:
            is_naturalimage = False
            segments_bad_properties = np.full(self.n_segments, True)

        return is_naturalimage, segments_bad_properties
    

    def is_gabor(self) -> tuple[bool, np.ndarray]:
        """Evaluate whether segments match the Gabor pattern.

        Returns
        -------
        tuple[bool, numpy.ndarray]
            Classification decision and per-segment boolean mask of failures.
        """
        
        # parameters
        n_segments = 12
        n_segments_tolerance = 2
        duration = 25
        duration_tolerance = 2  
        min_margin_size = 5
        thresh_background_proportion = 0.5 # minimun proprotion of background
        max_segments_not_maching = 2 # maximun number of segments not matching the requierments

        # compute the difference between the expected number of segments and the obtained
        n_segments_diff = np.abs(self.n_segments-n_segments)
    
        # check if the segments have the good duration
        good_duration = np.logical_and( self.segments['duration']>=(duration-duration_tolerance),  self.segments['duration']<=(duration+duration_tolerance))
        bad_duration = good_duration==False

        # check it has margins
        marg = ['margin_left','margin_right']
        has_marg = np.full(self.n_segments,True)
        for margi in marg:
            has_marg = np.logical_and(has_marg, self.segments[margi]>=min_margin_size)
        bad_marg = has_marg==False

        # check if all have background
        has_background = self.segments['background_proportion']>=thresh_background_proportion
        bad_background = has_background==False

        # count how many segments do not match the requierments
        segments_bad_properties = (bad_marg | bad_duration | bad_background)
        n_segments_bad_properties = np.sum(segments_bad_properties)
        
        # decide whether it is a Gabor
        if (n_segments_diff<=n_segments_tolerance) & (n_segments_bad_properties<=max_segments_not_maching):
            is_gabor = True
        else:
            is_gabor = False
            
        return is_gabor, segments_bad_properties
    

    def is_pinknoise(self) -> tuple[bool, np.ndarray]:
        """Evaluate whether segments match the PinkNoise pattern.

        Returns
        -------
        tuple[bool, numpy.ndarray]
            Classification decision and per-segment boolean mask of failures.
        """
        
        # parameters
        n_segments = 12
        n_segments_tolerance = 2
        duration = 27
        duration_tolerance = 2  
        min_margin_size = 5
        max_segments_not_maching = 2 # maximun number of segments not matching the requierments

        # compute the difference between the expected number of segments and the obtained
        n_segments_diff = np.abs(self.n_segments-n_segments)
    
        # check if the segments have the good duration
        good_duration = np.logical_and( self.segments['duration']>=(duration-duration_tolerance),  self.segments['duration']<=(duration+duration_tolerance))
        bad_duration = good_duration==False
        
        # check it does not have margin (it can be mixed with Gabor that does have it)
        marg = ['margin_left','margin_right','margin_top','margin_bottom']
        has_marg = np.full(self.n_segments,True)
        for margi in marg:
            has_marg = np.logical_and(has_marg, self.segments[margi]>=min_margin_size)
        bad_marg = has_marg==True

        # count how many segments do not match the requierments
        segments_bad_properties = (bad_marg | bad_duration )
        n_segments_bad_properties = np.sum(segments_bad_properties)
        
        # decide whether it is a PinkNoise
        if (n_segments_diff<=n_segments_tolerance) & (n_segments_bad_properties<=max_segments_not_maching):
            is_pinknoise = True
        else:
            is_pinknoise = False

        return is_pinknoise, segments_bad_properties
    

    def is_randomdots(self) -> tuple[bool, np.ndarray]:
        """Evaluate whether segments match the RandomDots pattern.

        Returns
        -------
        tuple[bool, numpy.ndarray]
            Classification decision and per-segment boolean mask of failures.
        """
        
        # parameters
        n_segments = 4
        n_segments_tolerance = 1
        duration = 60
        duration_tolerance = 2 
        thresh_background_proportion = 0.5
        max_segments_not_maching = 1 # maximun number of segments not matching the requierments

        # compute the difference between the expected number of segments and the obtained
        n_segments_diff = np.abs(self.n_segments-n_segments)
    
        # check if the segments have the good duration
        good_duration = np.logical_and( self.segments['duration']>=(duration-duration_tolerance),  self.segments['duration']<=(duration+duration_tolerance))
        bad_duration = good_duration==False

        # check if all have background
        has_background = self.segments['background_proportion']>=thresh_background_proportion
        bad_background = has_background==False

        # count how many segments do not match the requierments
        segments_bad_properties = (bad_background | bad_duration )
        n_segments_bad_properties = np.sum(segments_bad_properties)
        
        # decide whether it is a RandomDots
        if (n_segments_diff<=n_segments_tolerance) & (n_segments_bad_properties<=max_segments_not_maching):
            is_randomdot = True
        else:
            is_randomdot = False

        return is_randomdot, segments_bad_properties
        

    def is_naturalvideo(self, lim_segments: int = 3, min_var_duration: float | int = 1) -> tuple[bool, np.ndarray]:
        """Evaluate whether segments match the NaturalVideo pattern.

        Parameters
        ----------
        lim_segments : int, default=3
            Maximum segment count accepted without duration-variance checks.
        min_var_duration : float or int, default=1
            Minimum standard deviation in durations when many segments exist.

        Returns
        -------
        tuple[bool, numpy.ndarray]
            Classification decision and per-segment boolean mask of failures.
        """
        max_segments_not_maching = 1 # maximun number of segments not matching the requierments

        is_uniform = self.segments['is_uniform']
        segments_bad_properties = (is_uniform)
        n_segments_bad_properties = np.sum(segments_bad_properties)


        if self.n_segments<=lim_segments:
            no_pattern = True
        else:
            if np.std(self.segments['duration'])>=min_var_duration:
                no_pattern = True
            else:
                no_pattern = False

        if (n_segments_bad_properties<=max_segments_not_maching) & (no_pattern):
            is_naturalvideo = True
        else:
            is_naturalvideo = False

        return is_naturalvideo, segments_bad_properties
    

    def get_possible_labels(self) -> list[str]:
        """Return the list of supported video class labels.

        Returns
        -------
        list[str]
            Allowed class labels used by rule-based classification.
        """
        return ['NaturalVideo','NaturalImages','GaussianDot','Gabor','PinkNoise','RandomDots']
    

    def classify(self) -> None:
        """Assign the most plausible label from rule-based class checks.

        Notes
        -----
        Candidate non-natural classes are evaluated first. If exactly one class
        matches, it is assigned. Otherwise, ``NaturalVideo`` is tested; if that
        also fails, label is set to ``'unknown'``.
        """

        # labels to check first, if noone of this check if it can be a Natural Video
        labels_ = { 'GaussianDot':'is_gaussiandot',
                    'NaturalImages':'is_naturalimages',
                    'Gabor':'is_gabor',
                    'PinkNoise':'is_pinknoise',
                    'RandomDots':'is_randomdots',
                    }
        
        # check whether it could be the labels in labels_
        res = {}
        n_all = {}
        for labi,methodi in labels_.items():
            output = getattr(self, methodi)()
            res[labi] = output[0]
            n_all[labi] = output[1]
        
        # check for how many classes it matched, if more than one has True values, set as unknownw, if none check if it can be a NaturalVideo
        k = np.array(list(res.keys()))
        v = np.array(list(res.values()))
        n = np.array(list(n_all.values()))
        if len(np.where(v)[0])==1:
            label = k[np.where(v)[0][0]]
            segments_bad_properties = n[np.where(v)[0][0]]
        else:
            v, n = self.is_naturalvideo()
            if v:
                label = 'NaturalVideo'
                segments_bad_properties = n
            else:
                label = 'unknown'
                segments_bad_properties = np.full(self.n_segments, False)

        self.label = str(label)
        self.segments["bad_properties"] = segments_bad_properties
        

    def run_all(self) -> tuple[list[str], list[dict]]:
        """Execute the full segmentation and classification pipeline.

        Returns
        -------
        tuple[list[str], list[dict]]
            ``labels`` is the sequence of labels produced across passes.
            ``segments`` contains snapshots of segment dictionaries for each pass.
        """
        labels = []
        segments = []
        
        # find peaks
        self.find_peaks()

        # define the segments
        self.define_segments()

        # describe the segments
        self.apply_method_to_segments('describe')
        segments.append(self.segments)

        # classify
        self.classify()
        labels.append(self.label)

        # re-define the peaks with parameters set for that category and run classification again
        if self.label in ['NaturalVideo', 'RandomDots', 'NaturalImages', 'Gabor', 'GaussianDot', 'PinkNoise']:
            self.find_peaks(label=self.label)
            self.define_segments()
            segments.append(self.segments)
            self.apply_method_to_segments('describe')
            self.classify()
            labels.append(self.label)

        # recompute the segments when missing peaks are plausible
        self.split_long_segments_by_label(self.label)
        self.apply_method_to_segments('describe')
        segments.append(self.segments)
        self.classify()
        labels.append(self.label)

        return labels, segments


    def pick_key_frames(self, frames_per_segment: int | None = None, distance_between_key_frames: int | None = None) -> np.ndarray:
        """Select representative frame indices for this video.

        Parameters
        ----------
        frames_per_segment : int or None, optional
            Number of uniformly sampled frames in each segment.
        distance_between_key_frames : int or None, optional
            Step size (in frames) between selected key frames.

        Returns
        -------
        numpy.ndarray
            One-dimensional array with selected frame indices.
        """
        
        # select the parameters to pick some representative frames for each video based on the label
        if distance_between_key_frames is None and frames_per_segment is None:
            if self.label in ['GaussianDot', 'NaturalImages', 'Background']:  # static segments
                frames_per_segment = 1
                distance_between_key_frames = None
            elif self.label in ['RandomDots','Gabor','PinkNoise']:
                frames_per_segment = None
                distance_between_key_frames = 8
            elif self.label in ['NaturalVideo']:
                frames_per_segment = 1
                distance_between_key_frames = 10
            else:
                frames_per_segment = 1
                distance_between_key_frames = 10

        # pick the frames
        if ((hasattr(self,'segments')) 
            and ('frame_start' in self.segments.keys()) 
            and ('frame_end' in self.segments.keys())
            ):
            frame_start = self.segments['frame_start']
            frame_end = self.segments['frame_end']
        else:
            frame_start = np.asarray([0])
            frame_end = np.asarray([self.valid_frames])
        
        key_frames = pick_key_frames(frame_start, frame_end, 
                        frames_per_segment=frames_per_segment, 
                        distance_between_key_frames=distance_between_key_frames)
                
        return key_frames


    def plot_changes(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot frame-to-frame changes and detected peaks.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        # compute change
        change = self.compute_time_change()
        # find the peaks
        if not hasattr(self, 'peaks'):
            self.find_peaks()
        # plot
        fig, ax = plt.subplots(1,1)
        ax.plot(change[:self.valid_frames-1], 'k.', linestyle='-')
        ax.set_xlabel('Frame [t]')
        ax.set_ylabel('MSE[t+1, t]')
        if hasattr(self, 'peaks') and self.n_peaks>0:
            ax.plot(self.peaks, change[self.peaks],'rx', linestyle='none')
        if hasattr(self, 'segments'):
            for x in self.segments['frame_start'][1:]:
                ax.axvline(x-1, color='b', linestyle=':')
        return fig, ax
        

    def plot_intensity_hist_all(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot intensity histogram over all video frames.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        img = self.data
        fig, ax = plt.subplots(1,1)
        ax.hist(img.flatten(), range=(0, 255), bins=255)
        return fig, ax


    def plot_intensity_hist(self, segment) -> tuple[plt.Figure, plt.Axes]:
        """Plot intensity histogram for one segment.

        Parameters
        ----------
        segment : int
            Segment index.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        ki = self.segments['frame_start'][segment]+self.segments['transition_start'][segment]
        kf = self.segments['frame_end'][segment]-self.segments['transition_end'][segment]
        img = self.data[:,:,ki:kf]
        fig, ax = plt.subplots(1,1)
        ax.hist(img.flatten(), range=(0, 255), bins=255)
        return fig, ax


    def plot_frame(self, frame) -> tuple[plt.Figure, plt.Axes]:
        """Plot one frame from the video.

        Parameters
        ----------
        frame : int
            Frame index to display.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        fig, ax =plt.subplots(1,1)
        ax.imshow(self.data[:,:,frame], cmap='gray')
        return fig, ax


    def plot_frames(self, frames: np.ndarray | list[int], ncol: int = 5) -> tuple[plt.Figure, plt.Axes]:
        """Plot multiple video frames in a grid.

        Parameters
        ----------
        frames : array-like
            Frame indices to display.
        ncol : int, default=5
            Number of columns in the subplot grid.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        n_frames = len(frames)
        nrow = int(n_frames//ncol)
        if (ncol*nrow)<n_frames:
            nrow = (n_frames//ncol)+1
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*3, nrow*2))
        axs = np.array(ax, ndmin=1).ravel()  # robust handling of single Axes / 1D / 2D arrays
        for j, i in enumerate(frames):
            axs[j].imshow(self.data[:,:,i], cmap='gray', vmin=0, vmax=255)
            axs[j].set_title(f"frame {i}")
        plt.tight_layout()
        return fig, ax
    

    def display_video_clip(self) -> HTML:
        """Render the video as an inline animation.

        Returns
        -------
        IPython.display.HTML
            HTML animation object.
        """
        return display_video_clip(self.data[:,:,:self.valid_frames], interval_ms=1/self.sampling_freq*1000)
    

    def convert_to_dict(self, main_fields: list[str], segments_fields: list[str] | str | None, convert_duplicates: bool = True) -> dict:
            """Serialize selected video and segment metadata to a JSON-safe dict.

            Parameters
            ----------
            main_fields : list[str]
                Video-level attributes to include.
            segments_fields : list[str] or 'all' or None
                Segment fields to include. Use ``'all'`` for every available field.
            convert_duplicates : bool, default=True
                Whether to include duplicate recording/trial mappings.

            Returns
            -------
            dict
                JSON-serializable metadata dictionary.
            """
            meta_dict = {}
            
            # main fields
            for ff in main_fields:
                if hasattr(self, ff):
                    meta_dict[ff] = getattr(self, ff)

            # segments
            if not ((segments_fields==None) or (segments_fields=='all') or isinstance(segments_fields,list)):
                raise ValueError("segments_fields can be ''all'', None, or a list of strings")
            if hasattr(self, 'segments') and (segments_fields!=None):
                if isinstance(segments_fields,str) and segments_fields=='all':
                    segments_fields = self.segments.keys()
                if len(segments_fields)>0:
                    meta_dict['segments'] = {}
                    for ff in segments_fields:
                        if ff in self.segments.keys():
                            if isinstance(self.segments[ff], list)==False:
                                try:
                                    meta_dict['segments'][ff] = self.segments[ff].tolist()
                                except AttributeError:
                                    meta_dict['segments'][ff] = self.segments[ff]

            # duplicates
            if convert_duplicates and hasattr(self, 'duplicates'):
                meta_dict['duplicates'] = {}
                for rec in self.duplicates.keys():
                    meta_dict['duplicates'][rec] = {}
                    meta_dict['duplicates'][rec]['trials'] = list(self.duplicates[rec]['trials'])
                    meta_dict['duplicates'][rec]['n'] = self.duplicates[rec]['n']
            
            meta_dict = to_json_safe(meta_dict)
            return meta_dict
    
    
    def save_metadata(self, folder_metadata: str | Path, main_fields: list[str] | None = None, segments_fields: list[str] | str | None = None, save_duplicates: bool | None = None, file_name: str | None = None, metadata_for: str | None = None):
        """Save video metadata as JSON using optional preset configurations.

        Parameters
        ----------
        folder_metadata : str or pathlib.Path
            Output directory for the metadata file.
        main_fields : list[str] or None, optional
            Video-level fields to serialize. Defaults depend on ``metadata_for``.
        segments_fields : list[str] or 'all' or None, optional
            Segment-level fields to serialize.
        save_duplicates : bool or None, optional
            Whether duplicate mappings are included.
        file_name : str or None, optional
            Output filename. If ``None``, inferred from metadata settings.
        metadata_for : {'exemplar', 'videoID', None}, optional
            Preset that selects default fields and naming convention.
        """

        if not metadata_for in [None, 'exemplar', 'videoID']:
            raise ValueError("metadata_for can be 'exemplar' or 'videoID' or None")
        
        if metadata_for=='exemplar':
            if main_fields is None:
                main_fields = ['recording','trial','label','ID','sampling_freq','valid_frames','peaks']
            if segments_fields is None:
                segments_fields = 'all'
            if save_duplicates is None:
                save_duplicates = False
            if file_name is None:
                file_name = self.trial+'.json'
        elif metadata_for=='videoID':
            if main_fields is None:
                main_fields = ['label','ID','valid_frames','sampling_freq']
            if segments_fields is None:
                segments_fields = ['frame_start','frame_end','bad_properties']
            if save_duplicates is None:
                save_duplicates = True
            if file_name is None:
                file_name = self.label+'-'+self.ID+'.json'
        else:
            if main_fields is None:
                main_fields = ['label','ID','valid_frames','sampling_freq']
            if segments_fields is None:
                segments_fields = ['frame_start','frame_end']
            if save_duplicates is None:
                save_duplicates = True
            if file_name is None:
                file_name = self.label+'-'+self.ID+'.json'

        meta_video = self.convert_to_dict(main_fields, segments_fields, convert_duplicates=save_duplicates)   
        
        full_file_name = os.path.join(folder_metadata, file_name)
        with open(full_file_name, "w") as f:
            json.dump(meta_video, f, indent=4)

    def load_metadata(self, path_to_metadata_file: str | Path):
        """Load video metadata from a JSON file into object attributes.

        Parameters
        ----------
        path_to_metadata_file : str or pathlib.Path
            Path to a video metadata JSON file.
        """

        # load metadata
        with open(path_to_metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # set attributes
        for k, v in metadata.items():
            setattr(self, k, v)
        if hasattr(self, 'segments'):
            for k in self.segments.keys():
                self.segments[k] = np.asarray(self.segments[k])

        
    def load_metadata_from_id(self, folder_metadata: Path | str) -> None:
        """Load metadata by ``self.ID`` and update core video attributes.

        Parameters
        ----------
        folder_metadata : str or pathlib.Path
            Directory containing video metadata files.

        Raises
        ------
        ValueError
            If loaded metadata label or ID does not match the current object.
        """

        # load metadata
        metadata, _ = load_metadata_from_id(self.ID, folder_metadata)

        # check it is compatible
        if metadata['label']!=self.label:
            raise ValueError("The metadata file contains a label different from the video")
        if metadata['ID']!=self.ID:
            raise ValueError("The metadata file contains a ID different from the video")
        
        # set attributes
        self.valid_frames=metadata['valid_frames']
        self.segments={}
        for k in metadata['segments'].keys():
            self.segments[k] = np.asarray(metadata['segments'][k])

    
    def add_duplicates(self, recording: str, trials: list[str]) -> None:
        """Register duplicate trials associated with one recording.

        Parameters
        ----------
        recording : str
            Recording name used as duplicate key.
        trials : iterable
            Trial identifiers that map to the same unique video.
        """
        
        if not hasattr(self, 'duplicates'):
            self.duplicates = {}
        
        self.duplicates[recording] = {}
        self.duplicates[recording]['trials'] = set(trials)
        self.duplicates[recording]['n'] = len(self.duplicates[recording]['trials'])

    

class VideoData(Video):
    """Initialize ``Video`` objects directly from in-memory arrays."""

    def __init__(self, data: np.ndarray, recording: str | None = None, trial: str | None = None, sampling_freq: float | int | None = None, label: str | None = None, ID: str | None = None):
        """Initialize a ``Video`` object from an in-memory data array.

        Parameters
        ----------
        data : numpy.ndarray
            Video array with shape ``(height, width, n_frames)``.
        recording : str or None, optional
            Recording identifier.
        trial : str or None, optional
            Trial identifier.
        sampling_freq : float or int or None, optional
            Sampling frequency in Hz.
        label : str or None, optional
            Optional class label.
        ID : str or None, optional
            Optional unique identifier.
        """
        self.recording = recording
        self.trial = trial
        self.data = data
        self.sampling_freq = sampling_freq 
        n_emptyframes = np.sum(np.all(np.all(np.isnan(self.data),axis=0),axis=0))
        self.valid_frames = np.shape(self.data)[2]-n_emptyframes
        self.label = label
        self.ID = ID


class VideoID(Video):
    """Initialize ``Video`` objects from unique-video metadata IDs."""

    def __init__(self, 
                 data_folder: str | Path, 
                 videos_metadata_folder: str | Path, 
                 video_id: str): 
        """Initialize ``Video`` from a unique-video metadata identifier.

        Parameters
        ----------
        data_folder : str or pathlib.Path
            Root directory with raw recording folders.
        videos_metadata_folder : str or pathlib.Path
            Directory with unique-video metadata files.
        video_id : str
            Unique video identifier used to load metadata.
        """
        
        # load metadata
        metavideo, _ = load_metadata_from_id(video_id, videos_metadata_folder)

        # pick and exemplar video
        duplicates = metavideo.get("duplicates", {})
        if not duplicates:
            raise ValueError("No duplicates found in metadata")
        self.recording = random.choice(list(duplicates.keys()))

        self.trial = str(random.choice(list(metavideo["duplicates"][self.recording]['trials'])))
    
        # load an exemplar video
        recording_folder = os.path.join(data_folder, self.recording)
        self.data = np.load(os.path.join(recording_folder, 'data', 'videos', self.trial+'.npy'))
        
        # set other attributes
        n_emptyframes = np.sum(np.all(np.all(np.isnan(self.data),axis=0),axis=0))
        self.valid_frames = np.shape(self.data)[2]-n_emptyframes
        fields = ["label","ID","sampling_freq"]
        for ff in fields:
            if ff in metavideo.keys():
                setattr(self, ff, metavideo[ff])
        
        # set the segments
        self.segments = {}
        for ff in metavideo["segments"].keys():
            self.segments[ff] = np.asarray(metavideo["segments"][ff])

        # set duplicates
        self.duplicates = {}
        for rec in metavideo["duplicates"].keys():
            self.duplicates[rec] = {}
            self.duplicates[rec]['trials'] = set(metavideo["duplicates"][rec]['trials'])
            self.duplicates[rec]['n'] = metavideo["duplicates"][rec]['n']



class VideoSegment():
    """Represent and describe a single video segment."""
    # Parameter limiting the maximum accepted change to define that segments are static
    thresh_no_change = 50

    # Parameter limiting the maximum number of transition frames accepted when defining static segments
    max_transition_frames = 3

    # Parameter for the limit for the intensity range to define something is spatially uniform 
    thresh_intensity_range = 35


    def __init__(self, video: Video | np.ndarray, segment_index: int | None = None, sampling_freq: float | int | None = None, ID: str | None = None, label: str | None = None):
        """Initialize a segment from a ``Video`` object or a raw array.

        Parameters
        ----------
        video : Video or numpy.ndarray
            Source video object or segment array.
        segment_index : int or None, optional
            Segment index to extract when ``video`` is a ``Video`` instance.
        sampling_freq : float or int or None, optional
            Sampling frequency in Hz for raw-array initialization.
        ID : str or None, optional
            Segment identifier.
        label : str or None, optional
            Segment class label.
        """
        if not isinstance(video, (Video, np.ndarray)):
            raise ValueError("video must be either a numpy array or an instance of Video")

        self.parentvideo = {}
        if isinstance(video, Video):
            # get information about the video from which it was taken
            self.parentvideo['recording'] = video.recording
            self.parentvideo['trial'] = video.trial
            self.parentvideo['label'] = video.label
            self.parentvideo['ID'] = video.ID
            if segment_index==None:
                raise ValueError("An index indicating the segment to take must be provided")
            self.parentvideo['segment_index'] = segment_index
            self.parentvideo['frame_start'] = video.segments['frame_start'][segment_index]
            self.parentvideo['frame_end'] = video.segments['frame_end'][segment_index]

            sampling_freq = video.sampling_freq
            data = video.data[:,:,self.parentvideo['frame_start']:self.parentvideo['frame_end']]
        else:
            data = video

        # take the data        
        self.data = data
        
        # take generic info about the segment
        self.ID = ID
        self.sampling_freq = sampling_freq
        n_emptyframes = np.sum(np.all(np.all(np.isnan(self.data),axis=0),axis=0))
        self.valid_frames = np.shape(self.data)[2]-n_emptyframes
        self.label = label
        
        # add a segments properties attribute
        self.properties = {}

        # add an attribute to add duplicates
        self.duplicates = {}
        if isinstance(video, Video):
            self.add_duplicates(self.parentvideo['ID'], segment_index)
            
        
    def __copy__(self) -> Self:
        """Return a shallow copy of the segment object.

        Returns
        -------
        VideoSegment
            Shallow-copied segment instance.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new


    def __deepcopy__(self, memo) -> Self:
        """Return a deep copy of the segment object.

        Parameters
        ----------
        memo : dict
            Internal memoization dictionary used by ``copy.deepcopy``.

        Returns
        -------
        VideoSegment
            Deep-copied segment instance.
        """
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new
    
    def copy(self, deep=False) -> Self:
        """Copy the segment object.

        Parameters
        ----------
        deep : bool, default=False
            If True, return a deep copy. Otherwise return a shallow copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    

    def compute_time_change(self) -> np.ndarray:
        """Compute frame-to-frame changes for this segment.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of consecutive-frame MSE values.
        """
        return compute_videos_time_change(self.data, self.valid_frames)
    
    
    def is_static(self, limit: float | int = thresh_no_change, max_transition_frames: int = max_transition_frames):
        """Determine whether the segment is static under a change threshold.

        Parameters
        ----------
        limit : float or int, default=thresh_no_change
            Maximum allowed change for static classification.
        max_transition_frames : int, default=max_transition_frames
            Number of edge frames that may be ignored as transitions.

        Notes
        -----
        Results are stored in ``self.properties`` as ``transition_start``,
        ``transition_end``, ``max_change``, and ``is_static``.
        """
        
        # compute change
        change = self.compute_time_change()
        
        # find transition frames and maximun change excluding them
        if self.data.shape[2] > (2*max_transition_frames):
            n_first = find_edges(change, max_transition_frames, limit, revert=False)
            n_last = find_edges(change, max_transition_frames, limit, revert=True)
            max_change = np.max(change[n_first:len(change)-n_last])
        else:
            max_change = np.max(change)
            n_first = 0
            n_last = 0
        
        # determin if it is static
        is_static = max_change<=limit

        # store into the properties attribute
        self.properties['transition_start'] = n_first
        self.properties['transition_end'] = n_last
        self.properties['max_change'] = max_change
        self.properties['is_static'] = is_static
                

    def find_intensity_range(self) -> None:
        """Compute intensity range on non-transition frames.

        Notes
        -----
        Stores the result in ``self.properties['intensity_range']``.
        """
        if not 'is_static' in self.properties.keys():
            self.is_static()
        n_first = self.properties['transition_start']
        n_last = self.properties['transition_end']
        d = self.data[:,:,n_first:self.data.shape[2]-n_last]
        self.properties['intensity_range'] = (np.max(d) - np.min(d))
    

    def find_background(self) -> None:
        """Estimate dominant background color on non-transition frames and its proportion.

        Notes
        -----
        Stores ``background_color`` and ``background_proportion`` in
        ``self.properties``.
        """
        if not 'is_static' in self.properties.keys():
            self.is_static()
        n_first = self.properties['transition_start']
        n_last = self.properties['transition_end']
        d = self.data[:,:,n_first:self.data.shape[2]-n_last]
        hist, bin_edges = np.histogram(d, bins=255, range=(0,256), density=True)
        idx  = np.argmax(hist)
        self.properties['background_color'] = (bin_edges[idx]+bin_edges[idx+1])/2
        self.properties['background_proportion'] = hist[idx]  


    def is_uniform(self, thresh_background_proportion: float = 0.8, thresh_intensity_range: float | int = thresh_intensity_range) -> None:
        """Flag whether the segment is a mostly uniform static image.

        Parameters
        ----------
        thresh_background_proportion : float, default=0.8
            Minimum dominant-background proportion.
        thresh_intensity_range : float or int, default=thresh_intensity_range
            Maximum accepted intensity range.

        Notes
        -----
        Stores the boolean result in ``self.properties['is_uniform']``.
        """
        if not 'intensity_range' in self.properties.keys():
            self.find_intensity_range()
        
        if not 'background_proportion' in self.properties.keys():
            self.find_background()
        
        if not 'is_static' in self.properties.keys():
            self.is_static()
        
        self.properties['is_uniform'] = ((self.properties['background_proportion']>=thresh_background_proportion)
                                         and (self.properties['intensity_range']<=thresh_intensity_range)
                                         and (self.properties['is_static']))


    def find_margins(self, limit: float | int = thresh_intensity_range) -> None:
        """Estimate left/right/top/bottom uniform margins for the segment.

        Parameters
        ----------
        limit : float or int, default=thresh_intensity_range
            Maximum accepted intensity range for margin regions.

        Notes
        -----
        Stores margin values in ``self.properties``.
        """
        if not 'intensity_range' in self.properties.keys():
            self.find_intensity_range()
            
        if self.properties['intensity_range']<=limit:
                margin_left = self.data.shape[1]
                margin_right = self.data.shape[1]
                margin_top = self.data.shape[0]
                margin_bottom = self.data.shape[0]
        else:
            margin_left = find_margin(self.data, limit=limit, axis=1, revert=False)
            margin_right = find_margin(self.data, limit=limit, axis=1, revert=True)
            margin_top = find_margin(self.data, limit=limit, axis=0, revert=False)
            margin_bottom = find_margin(self.data, limit=limit, axis=0, revert=True)
        
        # store into the properties attribute
        self.properties['margin_left'] = margin_left
        self.properties['margin_right'] = margin_right
        self.properties['margin_top'] = margin_top
        self.properties['margin_bottom'] = margin_bottom


    def describe(self) -> None:
        """Compute all derived properties for this segment.

        Notes
        -----
        This method populates ``self.properties`` by calling, in order:
        ``is_static``, ``find_intensity_range``, ``find_margins``,
        ``find_background``, and ``is_uniform``.
        """
        self.is_static()
        self.find_intensity_range()
        self.find_margins()
        self.find_background()
        self.is_uniform()
         

    def label_from_parentvideo(self, thresh_intensity_range: float | int = thresh_intensity_range) -> None:
        """Assign a segment label using parent label and segment properties.

        Parameters
        ----------
        thresh_intensity_range : float or int, default=thresh_intensity_range
            Threshold used to split ``NaturalImages`` into image vs background.
        """
        
        # check the parent video has a label key
        if not 'label' in self.parentvideo.keys():
            raise ValueError("label is not define in parentvideo keys")
        
        match self.parentvideo['label']:
            case 'NaturalVideo':
                self.label = 'NaturalVideo'          
            case 'RandomDots':
                self.label = 'RandomDots'          
            case 'Gabor':
                self.label = 'Gabor'          
            case 'PinkNoise':
                self.label = 'PinkNoise'          
            case 'GaussianDot':
                self.label = 'GaussianDot'  
            case 'NaturalImages':
                self.find_intensity_range()
                if self.properties['intensity_range']<=thresh_intensity_range:
                    self.label = 'Background' 
                else:
                    self.label = 'NaturalImages'         
            case _:
                raise ValueError(f"{self.parentvideo['label']} is not a valid label")

    def pick_key_frames(self, frames_per_segment: int | None = None, distance_between_key_frames: int | None = None) -> np.ndarray:
        """Select representative frame indices for this segment.

        Parameters
        ----------
        frames_per_segment : int or None, optional
            Number of key frames sampled uniformly in the segment.
        distance_between_key_frames : int or None, optional
            Step size (in frames) between consecutive key frames.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of selected frame indices.
        """
        
        # select the parameters to pick some representative frames for each video based on the label
        if distance_between_key_frames is None and frames_per_segment is None:
            if self.label in ['GaussianDot', 'NaturalImages', 'Background']:  # static segments
                frames_per_segment = 1
                distance_between_key_frames = None
            elif self.label in ['RandomDots','Gabor','PinkNoise']:
                frames_per_segment = None
                distance_between_key_frames = 8
            elif self.label in ['NaturalVideo']:
                frames_per_segment = 1
                distance_between_key_frames = 10
            else:
                frames_per_segment = 1
                distance_between_key_frames = 10

        # pick the frames
        frame_start = np.asarray([0])
        frame_end = np.asarray([self.valid_frames])
        key_frames = pick_key_frames(frame_start, frame_end, 
                        frames_per_segment=frames_per_segment, 
                        distance_between_key_frames=distance_between_key_frames)
                
        return key_frames


    def convert_to_dict(self, main_fields: list[str], 
                        convert_duplicates: bool = True, convert_parentvideo: bool = False) -> dict:
        """Serialize segment metadata fields to a JSON-safe dictionary.

        Parameters
        ----------
        main_fields : list[str]
            Segment attributes to include at the top level.
        convert_duplicates : bool, default=True
            Whether to include duplicate mappings in the output.
        convert_parentvideo : bool, default=False
            Whether to include parent video information in the output.

        Returns
        -------
        dict
            JSON-serializable dictionary with selected metadata.
        """
        meta_dict = {}

        # main attributes
        for ff in main_fields:
            if hasattr(self, ff):
                meta_dict[ff] = getattr(self, ff)
        meta_dict = {k: int(v) if isinstance(v, np.integer) else v for k, v in meta_dict.items()}

        # parent video attributes
        if convert_parentvideo:
            meta_dict['parentvideo'] = {}
            for att, val in self.parentvideo.items():
                meta_dict['parentvideo'][att] = val

        # duplicates
        if convert_duplicates:
            meta_dict['duplicates'] = {}
            for videoid in self.duplicates.keys():
                meta_dict['duplicates'][videoid] = {}
                meta_dict['duplicates'][videoid]['segment_index'] = list(self.duplicates[videoid]['segment_index'])
                meta_dict['duplicates'][videoid]['n'] = self.duplicates[videoid]['n']

        return to_json_safe(meta_dict)
    

    def save_metadata(self, folder_metadata: str | Path, file_name: str | None = None, 
                      main_fields: list[str] = ['ID','label','sampling_freq','valid_frames'], convert_parentvideo: bool = False, convert_duplicates: bool = True):
        """Save segment metadata as a JSON file.

        Parameters
        ----------
        folder_metadata : str or pathlib.Path
            Output directory where the metadata file is written.
        file_name : str or None, optional
            Output filename. If ``None``, it is inferred from ``label`` and ``ID``.
        main_fields : list[str], optional
            Segment attributes to serialize.
        convert_parentvideo : bool, default=False
            Whether to include parent-video fields.
        convert_duplicates : bool, default=True
            Whether to include duplicate mappings.
        """

        if file_name is None:
            if self.label and self.ID:
                file_name = f"{self.label}-{self.ID}.json"
            else:
                raise ValueError("No label and ID found, please provide explicitely a file name")

        meta_video = self.convert_to_dict(main_fields=main_fields, 
                                          convert_duplicates=convert_duplicates, 
                                          convert_parentvideo=convert_parentvideo)   
        
        full_file_name = os.path.join(folder_metadata, file_name)
        with open(full_file_name, "w") as f:
            json.dump(meta_video, f, indent=4)

    
    def load_metadata(self, path_to_metadata_file: Path | str) -> None:
        """Load segment metadata from a JSON file into object attributes.

        Parameters
        ----------
        path_to_metadata_file : str or pathlib.Path
            Path to the segment metadata JSON file.
        """

        with open(path_to_metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            setattr(self, k, metadata[k])

    def load_metadata_from_id(self, folder_metadata: str | Path) -> None:
        """Load segment metadata by ``self.ID`` from a metadata directory.

        Parameters
        ----------
        folder_metadata : str or pathlib.Path
            Directory containing segment metadata files.
        """

        metadata, _ = load_metadata_from_id(self.ID, folder_metadata)
        for k, v in metadata.items():
            setattr(self, k, metadata[k])
        
        
    def add_duplicates(self, video_id: str, segment_index: int) -> None:
        """Register one duplicate occurrence for this segment.

        Parameters
        ----------
        video_id : str
            ID of a source video containing this segment.
        segment_index : int
            Segment index within the source video.
        """
        
        if video_id in self.duplicates.keys():
            self.duplicates[video_id]['segment_index'].add(segment_index)
        else:
            self.duplicates[video_id] = {}
            self.duplicates[video_id]['segment_index'] = {segment_index}
        self.duplicates[video_id]['n'] = len(self.duplicates[video_id]['segment_index'])


    def plot_frame(self, frame: int) -> tuple:
        """Plot one frame from the segment.

        Parameters
        ----------
        frame : int
            Frame index to display.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        fig, ax =plt.subplots(1,1)
        ax.imshow(self.data[:,:,frame], cmap='gray')
        return fig, ax


    def plot_frames(self, frames: list[int], ncol: int = 5) -> tuple:
        """Plot multiple segment frames in a grid.

        Parameters
        ----------
        frames : array-like
            Frame indices to display.
        ncol : int, default=5
            Number of columns in the subplot grid.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        n_frames = len(frames)
        nrow = int(n_frames//ncol)
        if (ncol*nrow)<n_frames:
            nrow = (n_frames//ncol)+1
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*3, nrow*2))
        axs = np.array(ax, ndmin=1).ravel()  # robust handling of single Axes / 1D / 2D arrays
        for j, i in enumerate(frames):
            axs[j].imshow(self.data[:,:,i], cmap='gray', vmin=0, vmax=255)
            axs[j].set_title(f"frame {i}")
        plt.tight_layout()
        return fig, ax
    
    def display_video_clip(self) -> HTML:
        """Render the segment as an inline animation.

        Returns
        -------
        IPython.display.HTML
            HTML animation object.
        """
        return display_video_clip(self.data[:,:,:self.valid_frames], interval_ms=1/self.sampling_freq*1000)
     
    
        
class VideoSegmentID(VideoSegment):

    """Initialize ``VideoSegment`` objects from segment metadata IDs."""

    def __init__(self, 
                 data_folder: str | Path, 
                 videos_metadata_folder: str | Path, 
                 segment_metadata_folder: str | Path, 
                 segment_id: str) -> None: 
        """Initialize a ``VideoSegment`` object from a segment metadata ID.

        Parameters
        ----------
        data_folder : str or pathlib.Path
            Root directory with raw video files.
        videos_metadata_folder : str or pathlib.Path
            Directory with unique-video metadata files.
        segment_metadata_folder : str or pathlib.Path
            Directory with segment metadata files.
        segment_id : str
            Segment ID used to locate metadata.
        """
        
        # load metadata
        metasegment, _ = load_metadata_from_id(segment_id, segment_metadata_folder)

        # pick an exemplar video 
        duplicates = metasegment.get("duplicates", {})
        if not duplicates:
            raise ValueError("No duplicates found in metadata")
        videoID = random.choice(list(metasegment["duplicates"].keys()))
        
        segment_indices = duplicates[videoID].get("segment_index", [])
        if not segment_indices:
            raise ValueError(f"No segment indices found for video {videoID}")
        segment_index = random.choice(list(metasegment["duplicates"][videoID]["segment_index"]))
        
        # initialize parent class
        video = VideoID(data_folder, videos_metadata_folder, f"*{videoID}")
        super().__init__(video, segment_index)
        
        # set other properties
        self.ID = metasegment['ID']
        self.label = metasegment['label']




