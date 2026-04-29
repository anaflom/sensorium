# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
import pandas as pd
import json
import copy
from typing import Any, Self
from scipy.signal import detrend
from scipy.signal import welch
from pathlib import Path

import matplotlib.pyplot as plt

from ssdatam.data_handling import load_metadata_json_to_obj


def compute_power_spectrum(
    data: np.ndarray, sampling_freq: float | int
) -> tuple[np.ndarray, np.ndarray]:
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
        detrend="linear",
    )

    return freqs, psd


class Behavior:

    def __init__(
        self,
        recording_folder: str | Path,
        trial: str,
        behavior_type: str,
        sampling_freq: float | int = 30,
        valid_frames: int | None = None,
        label: str | None = None,
        ID: str | None = None,
        indexes: list[int] | np.ndarray | None = None,
    ):
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

        if behavior_type not in ["pupil_center", "behavior"]:
            raise ValueError("behavior_type must be 'pupil_center' or 'behavior'")
        if indexes is not None:
            if not isinstance(indexes, list) and not isinstance(indexes, np.ndarray):
                raise ValueError("indexes must be a list or a numpy array")

        trial, ext = os.path.splitext(trial)
        trial = os.path.basename(trial)

        self.recording = os.path.basename(recording_folder)
        self.trial = trial
        d = np.load(
            os.path.join(recording_folder, "data", behavior_type, trial + ".npy")
        )
        if indexes is not None:
            d = d[np.asarray(indexes), :]
        self.data = d.squeeze()

        self.sampling_freq = sampling_freq
        if valid_frames is None:
            n_emptyframes = np.sum(np.all(np.isnan(self.data), axis=0))
            self.valid_frames = np.shape(self.data)[-1] - n_emptyframes
        else:
            self.valid_frames = valid_frames

        self.label = label
        self.ID = ID

    def __copy__(self) -> Self:
        """Return a shallow copy of the object.

        Returns
        -------
        Behavior
            Shallow copy.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
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

    def copy(self, deep: bool = False) -> Self:
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

    def get_data(self) -> np.ndarray:
        """Return the behavior data array.

        Returns
        -------
        numpy.ndarray
            Behavior data with shape ``(n_channels, valid_frames)`` or ``(valid_frames,)`` if one channel.
        """
        return self.data[:, : self.valid_frames] if self.data.ndim > 1 else self.data[: self.valid_frames]
    
    def load_metadata(self, 
                      path_to_metadata_file: str | Path,
                      attributes_to_check_match: list[str] = ["label", "ID", "valid_frames","sampling_freq"],
                      attributes_to_add: list[str] = ["segments","duplicates"],
                      raise_on_mismatch: bool = True,
                      verbose: bool = True) -> None:
        """Load metadata for current response.

        Parameters
        ----------
        path_to_metadata_file : str or pathlib.Path
            Path to a JSON file with the metadata.
        attributes_to_check_match : list of str, default=["label", "ID", "valid_frames","sampling_freq"]
            List of attributes to check for consistency between the object and the metadata file. If an attribute in this list is not set in the object, it will be set from the metadata file. If it is already set in the object, it will be checked that it matches the value in the metadata file. If there is a mismatch, a ValueError will be raised.
        attributes_to_add : list of str, optional
            List of additional attributes to add from the metadata file. If None, all attributes not in attributes_to_check_match will be added.
        verbose : bool, default=True
            If True, print warnings.
        """

        self = load_metadata_json_to_obj(self, 
                                  path_to_metadata_file, 
                                  attributes_to_check_match=attributes_to_check_match,
                                  attributes_to_add=attributes_to_add,
                                  raise_on_mismatch=raise_on_mismatch,
                                  verbose=verbose)
            

class Gaze(Behavior):

    def __init__(self, recording_folder: str | Path, 
                 trial: str, ID: str | None = None, 
                 label: str | None = None, 
                 valid_frames: int | None = None, 
                 sampling_freq: float | int = 30):
        """Initialize gaze traces for one trial."""
        super().__init__(recording_folder, trial, behavior_type="pupil_center", indexes=[0, 1],
                         ID=ID, sampling_freq=sampling_freq, valid_frames=valid_frames, label=label)

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot 2D gaze trajectory.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.plot(self.data[0, :], self.data[1, :], "k")
        ax.set_xlabel("gaze x")
        ax.set_ylabel("gaze y")
        return fig, ax


class Pupil(Behavior):

    def __init__(self, recording_folder: str | Path, trial: str, 
                 ID: str | None = None, 
                 label: str | None = None, 
                 valid_frames: int | None = None, 
                 sampling_freq: float | int = 30):
        """Initialize pupil-size trace for one trial."""
        super().__init__(recording_folder, trial, behavior_type="behavior", indexes=[0],
                         ID=ID, sampling_freq=sampling_freq, valid_frames=valid_frames, label=label)

    def detrend(self, type: str = "linear", bp: int | list[int] = 0) -> "Pupil":
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
        data_detrended = detrend(
            self.data[: self.valid_frames], axis=-1, type=type, bp=bp
        )
        self.data[: self.valid_frames] = data_detrended
        return self

    def compute_power_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum of the valid pupil samples.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequencies and PSD values.
        """
        return compute_power_spectrum(
            self.data[: self.valid_frames], self.sampling_freq
        )

    def plot_power_spectrum(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot pupil power spectrum.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        freqs, psd = compute_power_spectrum(
            self.data[: self.valid_frames], self.sampling_freq
        )
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.semilogy(freqs, psd)
        ax.set_xlim(0, 2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title("Pupil size power spectrum")
        plt.show()
        return fig, ax

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot pupil trace over time.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        fig, ax = plt.subplots(ncols=1, nrows=1)
        time = np.arange(len(self.data)) / self.sampling_freq
        ax.plot(time, self.data, "k")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("pupil size")
        return fig, ax


class Locomotion(Behavior):

    def __init__(self, recording_folder: str | Path, 
                 trial: str, ID: str | None = None, 
                 label: str | None = None, 
                 valid_frames: int | None = None, 
                 sampling_freq: float | int = 30):
        """Initialize locomotion trace for one trial."""
        super().__init__(recording_folder, trial, behavior_type="behavior", indexes=[1],
                         ID=ID, sampling_freq=sampling_freq, valid_frames=valid_frames, label=label)

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot locomotion speed over time.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib objects.
        """
        fig, ax = plt.subplots(ncols=1, nrows=1)
        time = np.arange(len(self.data)) / self.sampling_freq
        ax.plot(time, self.data, "k")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("locomotion speed")
        return fig, ax
