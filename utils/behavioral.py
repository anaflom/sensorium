import os
import numpy as np
import pandas as pd
import json
import copy
from scipy.signal import detrend
from scipy.signal import welch

import matplotlib.pyplot as plt


def compute_power_spectrum(data, sampling_freq):
    """Compute power spectral density using Welch's method.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional time series.
    sampling_freq : float or int
        Sampling frequency in Hz.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Frequencies and corresponding power spectral density values.
    """
    
    # Remove DC offset (important for pupil data)
    data = data - np.mean(data)

    freqs, psd = welch(
        data,
        fs=sampling_freq,
        window="hann",
        nperseg=128,
        noverlap=64,
        detrend="linear"
    )

    return freqs, psd



class Behavior():

    def __init__(self, recording_folder, trial, behavior_type, indexes=None):
        """Initialize behavior data for one recording/trial.

        Parameters
        ----------
        recording_folder : str or pathlib.Path
            Path to recording folder.
        trial : str
            Trial name or filename.
        behavior_type : {'pupil_center', 'behavior'}
            Behavior data subfolder to load.
        indexes : list[int] or numpy.ndarray or None, optional
            Optional channel indices to select.
        """

        if behavior_type not in ['pupil_center','behavior']:
            raise ValueError("behavior_type must be 'pupil_center' or 'behavior'")
        if indexes is not None: 
            if not isinstance(indexes, list) and not isinstance(indexes, np.ndarray):
                raise ValueError("indexes must be a list or a numpy array")

        trial, ext = os.path.splitext(trial)
        trial = os.path.basename(trial)
        
        self.recording = os.path.basename(recording_folder)
        self.trial = trial
        d = np.load(os.path.join(recording_folder, 'data', behavior_type, trial+'.npy'))
        if indexes is not None:
            d = d[np.asarray(indexes),:]
        self.data = d.squeeze()

        self.sampling_freq = 30
        n_emptyframes = np.sum(np.all(np.isnan(self.data),axis=0))
        self.valid_frames = np.shape(self.data)[-1]-n_emptyframes

        self.label = None
        self.ID = None

    def __copy__(self):
        """Return a shallow copy of the object.

        Returns
        -------
        Behavior
            Shallow copy.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new


    def __deepcopy__(self, memo):
        """Return a deep copy of the object.

        Parameters
        ----------
        memo : dict
            Memo dictionary used by ``copy.deepcopy``.

        Returns
        -------
        Behavior
            Deep copy.
        """
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new
    
    def copy(self, deep=False):
        """Copy the object.

        Parameters
        ----------
        deep : bool, default=False
            If ``True``, return deep copy, otherwise shallow copy.

        Returns
        -------
        Behavior
            Copied object.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)


    def load_metadata_videoid(self, folder_metadata):
        """Load video-level metadata by current ``label`` and ``ID``.

        Parameters
        ----------
        folder_metadata : str or pathlib.Path
            Folder containing global video metadata JSON files.
        """

        file_metadata = self.label+'-'+self.ID+".json"
        path_metavideo = os.path.join(folder_metadata, file_metadata)
        with open(path_metavideo, "r", encoding="utf-8") as f:
            metavideo = json.load(f)

        if metavideo['label']!=self.label:
            raise ValueError("The metadata file contains a label different from the video")
        if metavideo['ID']!=self.ID:
            raise ValueError("The metadata file contains an ID different from the video")
        
        self.valid_frames=metavideo['valid_frames']

        if 'segments' in metavideo.keys():
            self.segments={}
            for k in metavideo['segments'].keys():
                self.segments[k] = np.asarray(metavideo['segments'][k])

    

    

class Gaze(Behavior):

    def __init__(self, recording_folder, trial):
        """Initialize gaze traces for one trial."""
        super().__init__(recording_folder, trial, behavior_type='pupil_center', indexes=[0,1])

    def plot(self):
        """Plot 2D gaze trajectory.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        fig, ax = plt.subplots(ncols=1,nrows=1)
        ax.plot(self.data[0,:], self.data[1,:],'k')
        ax.set_xlabel('gaze x')
        ax.set_ylabel('gaze y')
        return fig, ax

class Pupil(Behavior):

    def __init__(self, recording_folder, trial):
        """Initialize pupil-size trace for one trial."""
        super().__init__(recording_folder, trial, behavior_type='behavior', indexes=[0])

    def detrend(self, type='linear', bp=0):
        """Detrend pupil signal in place.

        Parameters
        ----------
        type : str, default='linear'
            Detrending type passed to ``scipy.signal.detrend``.
        bp : int or array-like, default=0
            Breakpoints for piecewise detrending.

        Returns
        -------
        Pupil
            Self instance after detrending.
        """
        data_detrended = detrend(self.data[:self.valid_frames], axis=-1, type=type, bp=bp)
        self.data[:self.valid_frames] = data_detrended
        return self
    
    def compute_power_spectrum(self):
        """Compute power spectrum of the valid pupil samples.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequencies and PSD values.
        """
        return compute_power_spectrum(self.data[:self.valid_frames], self.sampling_freq)

    def plot_power_spectrum(self):
        """Plot pupil power spectrum.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        freqs, psd = compute_power_spectrum(self.data[:self.valid_frames], self.sampling_freq)
        fig, ax = plt.subplots(ncols=1,nrows=1)
        ax.semilogy(freqs, psd)
        ax.set_xlim(0, 2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title("Pupil size power spectrum")
        plt.show()
        return fig, ax

    def plot(self):
        """Plot pupil trace over time.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        fig, ax = plt.subplots(ncols=1,nrows=1)
        time = np.arange(len(self.data))/self.sampling_freq
        ax.plot(time, self.data,'k')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('pupil size')
        return fig, ax
    
class Locomotion(Behavior):

    def __init__(self, recording_folder, trial):
        """Initialize locomotion trace for one trial."""
        super().__init__(recording_folder, trial, behavior_type='behavior', indexes=[1])

    def plot(self):
        """Plot locomotion speed over time.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        fig, ax = plt.subplots(ncols=1,nrows=1)
        time = np.arange(len(self.data))/self.sampling_freq
        ax.plot(time, self.data,'k')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('locomotion speed')
        return fig, ax
