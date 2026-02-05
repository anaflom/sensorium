import os
import numpy as np
import pandas as pd
import json
import copy
import matplotlib.pyplot as plt

class Neurons():

    def __init__(self, folder_metadata, recording):

        file = os.path.join(folder_metadata, recording, f"meta-neurons_{recording}.csv")
        df_neurons = pd.read_csv(file)
        self.coord_xyz = df_neurons[['coord_x', 'coord_y','coord_z']].copy().to_numpy()
        self.IDs = df_neurons['ID'].copy().to_numpy()
        self.stats_activity = {k:df_neurons[k].copy().to_numpy() for k in ['mean','std','median','min','max']}


    def plot_coordinates(self):

        # Plot the neural coordinates of the recorded neurons 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with color mapped to z
        sc = ax.scatter(self.coord_xyz[:,0], 
                        self.coord_xyz[:,1], 
                        self.coord_xyz[:,2], 
                        c=self.coord_xyz[:,2], 
                        cmap='viridis', s=10, alpha=0.5)

        # Add a colorbar, labels
        plt.colorbar(sc, ax=ax, label='Z value')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])   # equal scaling
        plt.show()
        return fig, ax

class Responses():

    def __init__(self, recording_folder, trial):
        trial, ext = os.path.splitext(trial)
        trial = os.path.basename(trial)
        
        self.recording = os.path.basename(recording_folder)
        self.trial = trial
        self.data = np.load(os.path.join(recording_folder, 'data', 'responses', trial+'.npy'))
        self.sampling_freq = 30

        n_emptyframes = np.sum(np.all(np.isnan(self.data),axis=0))
        self.valid_frames = np.shape(self.data)[-1]-n_emptyframes

        self.label = None
        self.ID = None
        self.neurons = None

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new


    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new
    
    def copy(self, deep=False):
        return copy.deepcopy(self) if deep else copy.copy(self)


    def load_metadata_videoid(self, folder_metadata):

        file_metadata = self.label+'-'+self.ID+".json"
        path_metavideo = os.path.join(folder_metadata, file_metadata)
        with open(path_metavideo, "r", encoding="utf-8") as f:
            metavideo = json.load(f)

        if metavideo['label']!=self.label:
            raise ValueError("The metadata file contains a label different from the video")
        if metavideo['ID']!=self.ID:
            raise ValueError("The metadata file contains a ID different from the video")
        
        self.valid_frames=metavideo['valid_frames']

        self.segments={}
        for k in metavideo['segments'].keys():
            self.segments[k] = np.asarray(metavideo['segments'][k])


    def load_metadata_neurons(self, folder_metadata):

        self.neurons = Neurons(folder_metadata, self.recording)


    def get_data(self, normalization=None):
        if normalization is None:
            data = self.data[:,:self.valid_frames].copy()
        elif normalization=='by_std':
            std = self.neurons.stats_activity['std']
            data = np.divide(self.data[:,:self.valid_frames], std[:,None])
        elif normalization=='by_mean':
            mu = self.neurons.stats_activity['mean']
            data = np.divide(self.data[:,:self.valid_frames]-mu[:,None], mu[:,None])
        else:
            raise ValueError("Normalization can have values: None, by_std, or by_mean")

        return data


    def plot_responses_raster(self, neurons_idx, normalization=None, plot_segments=None):

        if plot_segments is None:
            if self.label=='NaturalVideo':
                plot_segments=False
            else:
                plot_segments=True

        # get the data
        data = self.get_data(normalization=normalization)

        if normalization is None or normalization=='by_std':
            vmin = 0
        elif normalization=='by_mean':
            vmin = -1
        vmax = np.percentile(data.flatten(),99)

        # plot some neurons (rester plot)
        n = len(neurons_idx)
        fig, ax = plt.subplots(1, 1, figsize=(8, 0.05 * n))
        ax.imshow(data[neurons_idx,:], cmap='gray_r', vmin=vmin, vmax=vmax)
        if plot_segments and hasattr(self, 'segments'):
            for x in self.segments['frame_start'][1:]:
                ax.axvline(x-0.5, color='b', linestyle=':',linewidth=0.5)
        ax.set_xlabel("samples")

        return fig, ax
    
    def plot_active_raster(self, neurons_idx, thresh, normalization=None, plot_segments=None):

        if plot_segments is None:
            if self.label=='NaturalVideo':
                plot_segments=False
            else:
                plot_segments=True

        # get the data
        data = self.get_data(normalization=normalization)
        data_act = data>thresh

        # plot some neurons (rester plot)
        n = len(neurons_idx)
        fig, ax = plt.subplots(1, 1, figsize=(8, 0.05 * n))
        ax.imshow(data_act[neurons_idx,:], cmap='gray_r', vmin=0, vmax=1)
        if plot_segments and hasattr(self, 'segments'):
            for x in self.segments['frame_start'][1:]:
                ax.axvline(x-0.5, color='b', linestyle=':',linewidth=0.5)
        ax.set_xlabel("samples")

        return fig, ax    
    

    def plot_responses(self, neurons_idx, normalization=None, plot_segments=None):

        if plot_segments is None:
            if self.label=='NaturalVideo':
                plot_segments=False
            else:
                plot_segments=True

        # get the data
        data = self.get_data(normalization=normalization)

        # plot some neurons (rester plot)
        n = len(neurons_idx)
        time = np.arange(data.shape[1])/self.sampling_freq
        
        fig, axs = plt.subplots(n, 1, sharex=True, figsize=(8, 1 * n), gridspec_kw={"hspace": 0.05})
        if n == 1:
            axs = [axs]
        for ax, neuron_idx in zip(axs, neurons_idx):
            ax.plot(time, data[neuron_idx,:],'k')
            ax.text(-0.1, 0.7, f"{neuron_idx}", transform=ax.transAxes, fontsize=10, color='black', weight='bold')
            if plot_segments and hasattr(self, 'segments'):
                for x in self.segments['frame_start'][1:]:
                    ax.axvline(time[x-1], color='b', linestyle=':',linewidth=0.5)
        axs[-1].set_xlabel("time (s)")
        plt.tight_layout()

        return fig, axs    
    

