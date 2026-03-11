import json
import os
from typing import Self, Any

import assign_grid as ag
import numpy as np
import pandas as pd
from pathlib import Path
import copy


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from utils.data_handling import get_file_with_pattern
from utils.dataset import DataSet
from utils.responses import Responses



def get_ranges_from_positions(positions):
    x_range = (positions[:, 0].min(), positions[:, 0].max())
    y_range = (positions[:, 1].min(), positions[:, 1].max())
    z_pos = np.unique(positions[:, 2])
    layers_distance = np.unique(np.diff(z_pos))
    if len(layers_distance) != 1:
        raise ValueError("Z positions are not evenly spaced.")
    z_step = layers_distance[0]
    z_range = (positions[:, 2].min()-z_step/2, positions[:, 2].max()+z_step/2)
    return (x_range, y_range, z_range)

def create_grid_3D(xyz_ranges, num_grid):
    
    x_range = xyz_ranges[0]
    x_step = (x_range[1] - x_range[0]) / num_grid[0]
    x_lin = np.linspace(x_range[0], x_range[1], num_grid[0]+1)
    x_grid_coordinates = np.array([x+x_step/2 for x in x_lin[:-1]])
    
    y_range = xyz_ranges[1]
    y_step = (y_range[1] - y_range[0]) / num_grid[1]
    y_lin = np.linspace(y_range[0], y_range[1], num_grid[1]+1)
    y_grid_coordinates = np.array([y+y_step/2 for y in y_lin[:-1]])
    
    z_range = xyz_ranges[2]
    z_step = (z_range[1] - z_range[0]) / num_grid[2]
    z_lin = np.linspace(z_range[0], z_range[1], num_grid[2]+1)
    z_grid_coordinates = np.array([z+z_step/2 for z in z_lin[:-1]])

    X, Y, Z = np.meshgrid(x_grid_coordinates, y_grid_coordinates, z_grid_coordinates, indexing='ij')

    return (X,Y,Z), (x_lin, y_lin, z_lin)


def compute_grid_activity_3D(positions, activities, num_grid, xyz_ranges):
    num_frames = activities.shape[1]
    grid_activity = np.zeros((num_grid[0], num_grid[1], num_grid[2], num_frames))
    for t in range(num_frames):
        grid_activity_t = ag.pcs_assign_3d(
            positions[:, 0], positions[:, 1], positions[:, 2], activities[:, t],len(positions),
            num_grid[0], num_grid[1], num_grid[2],
            xyz_ranges[0], xyz_ranges[1], xyz_ranges[2]
            )
        grid_activity[:,:,:,t] = np.transpose(grid_activity_t, (2, 1, 0))
    return grid_activity



class Grid3D:

    def __init__(self, xyz_ranges, num_grid):
        self.num_grid = num_grid
        self.xyz_ranges = xyz_ranges
        self.xyz_coordinates, self.xyz_lines = create_grid_3D(xyz_ranges, num_grid)

    def compute_grid_activity(self, positions, activities):
        return compute_grid_activity_3D(positions, activities, self.num_grid, self.xyz_ranges)
    
    def neurons_in_plane(self, positions, axis, plane_index):
        
        if axis==2:
            x_limits = (self.xyz_ranges[0][0], self.xyz_ranges[0][1])
            y_limits = (self.xyz_ranges[1][0], self.xyz_ranges[1][1])
            z_limits = (self.xyz_lines[2][plane_index], self.xyz_lines[2][plane_index+1])
            lines_plane = (self.xyz_lines[0], self.xyz_lines[1])
            plane_axis_idx = (0, 1)
        elif axis==1:
            x_limits = (self.xyz_ranges[0][0], self.xyz_ranges[0][1])
            y_limits = (self.xyz_lines[1][plane_index], self.xyz_lines[1][plane_index+1])
            z_limits = (self.xyz_ranges[2][0], self.xyz_ranges[2][1])
            lines_plane = (self.xyz_lines[0], self.xyz_lines[2])
            plane_axis_idx = (0, 2)
        elif axis==0:
            x_limits = (self.xyz_lines[0][plane_index], self.xyz_lines[0][plane_index+1])
            y_limits = (self.xyz_ranges[1][0], self.xyz_ranges[1][1])
            z_limits = (self.xyz_ranges[2][0], self.xyz_ranges[2][1])
            lines_plane = (self.xyz_lines[1], self.xyz_lines[2])
            plane_axis_idx = (1, 2)

        mask = (positions[:,0]>= x_limits[0]) & (positions[:,0]< x_limits[1]) & \
               (positions[:,1]>= y_limits[0]) & (positions[:,1]< y_limits[1]) & \
               (positions[:,2]>= z_limits[0]) & (positions[:,2]< z_limits[1])

        return np.where(mask)[0], lines_plane, plane_axis_idx
    
    def neurons_in_cell(self, positions, cell_index):
        x_limits = (self.xyz_lines[0][cell_index[0]], self.xyz_lines[0][cell_index[0]+1])
        y_limits = (self.xyz_lines[1][cell_index[1]], self.xyz_lines[1][cell_index[1]+1])
        z_limits = (self.xyz_lines[2][cell_index[2]], self.xyz_lines[2][cell_index[2]+1])

        mask = (positions[:,0]>= x_limits[0]) & (positions[:,0]< x_limits[1]) & \
               (positions[:,1]>= y_limits[0]) & (positions[:,1]< y_limits[1]) & \
               (positions[:,2]>= z_limits[0]) & (positions[:,2]< z_limits[1])
        
        return np.where(mask)[0]
   
    def save(self, folder):
        np.save(folder / 'num_grid.npy', self.num_grid)
        np.save(folder / 'xyz_ranges.npy', self.xyz_ranges)
    
    def plot_bar_neurons_activity(self, axis, plane_idx, frame, positions, activities, ax = None):
        
        neurons_idx, lines_plane, plane_axis_idx = self.neurons_in_plane(positions, axis, plane_idx)

        coord1_flat = positions[neurons_idx, plane_axis_idx[0]]
        coord2_flat = positions[neurons_idx, plane_axis_idx[1]]
        act_flat = activities[neurons_idx, frame]

        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

        d_x = 10   # bar width/depth
        d_y = 10   # bar width/depth
        d_z = act_flat     # bar height = z value

        ax.bar3d(
            coord1_flat,  # center the bars on x
            coord2_flat,  # center the bars on y
            np.zeros_like(act_flat),  # bars start at z=0
            d_x, d_y, d_z,
            shade=True,
            color=plt.cm.viridis(d_z / d_z.max()),  # color by height
            alpha=0.9
        )

        coord_names = ['X', 'Y', 'Z']
        ax.set_xlim(self.xyz_ranges[plane_axis_idx[0]])
        ax.set_ylim(self.xyz_ranges[plane_axis_idx[1]])
        ax.set_xticks(np.linspace(self.xyz_ranges[plane_axis_idx[0]][0], self.xyz_ranges[plane_axis_idx[0]][1], self.num_grid[plane_axis_idx[0]]+1))
        ax.set_yticks(np.linspace(self.xyz_ranges[plane_axis_idx[1]][0], self.xyz_ranges[plane_axis_idx[1]][1], self.num_grid[plane_axis_idx[1]]+1))
        ax.set_xlabel(coord_names[plane_axis_idx[0]])
        ax.set_ylabel(coord_names[plane_axis_idx[1]])
        ax.set_zlabel('Neurons activity')
        ax.set_title(f'Neurons activity at {coord_names[axis]} plane {plane_idx}, frame {frame}')

        plt.tight_layout()
        plt.show()
        return ax

    def plot_bar_grid_activity(self, axis, plane_idx, frame, grid_activity, ax = None):

        if axis==2:
            coord1 = self.xyz_coordinates[0][:,:,plane_idx]
            coord2 = self.xyz_coordinates[1][:,:,plane_idx]
            act = grid_activity[:,:,plane_idx,frame]
            plane_axis_idx = (0, 1)
        elif axis==1:
            coord1 = self.xyz_coordinates[0][:,plane_idx,:]
            coord2 = self.xyz_coordinates[2][:,plane_idx,:]
            act = grid_activity[:,plane_idx,:,frame]
            plane_axis_idx = (0, 2)
        elif axis==0:
            coord1 = self.xyz_coordinates[1][plane_idx,:,:]
            coord2 = self.xyz_coordinates[2][plane_idx,:,:]
            act = grid_activity[plane_idx,:,:,frame]
            plane_axis_idx = (1, 2)

        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

        coord1_flat = coord1.ravel()
        coord2_flat = coord2.ravel()
        act_flat = act.ravel()

        dx = 0.6 * (coord1[1,0] - coord1[0,0])   # bar width/depth
        dy = 0.6 * (coord2[0,1] - coord2[0,0])   # bar width/depth
        dz = act_flat     # bar height = z value

        ax.bar3d(
            coord1_flat,  # center the bars on x
            coord2_flat,  # center the bars on y
            np.zeros_like(dz),  # bars start at z=0
            dx, dy, dz,
            shade=True,
            color=plt.cm.viridis(dz / dz.max()),  # color by height
            alpha=0.9
        )

        coord_names = ['X', 'Y', 'Z']
        ax.set_xlabel(coord_names[plane_axis_idx[0]])
        ax.set_ylabel(coord_names[plane_axis_idx[1]])
        ax.set_zlabel('Grid activity')
        ax.set_title(f'Grid activity at {coord_names[axis]} plane {plane_idx}, frame {frame}')

        plt.tight_layout()
        plt.show()
        return ax

    def plot_scatter_neurons_activity(self, axis, plane_idx, frame, positions, activities, ax = None):
        
        neurons_idx, lines_plane, plane_axis_idx = self.neurons_in_plane(positions, axis, plane_idx)

        coord1_flat = positions[neurons_idx, plane_axis_idx[0]]
        coord2_flat = positions[neurons_idx, plane_axis_idx[1]]
        act_flat = activities[neurons_idx, frame]

        if ax is None:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)

        scatter = ax.scatter(coord1_flat, coord2_flat, c=act_flat, cmap='viridis')
        fig.colorbar(scatter, ax=ax, label='Activity Level')

        coord_names = ['X', 'Y', 'Z']
        ax.set_xlim(self.xyz_ranges[plane_axis_idx[0]])
        ax.set_ylim(self.xyz_ranges[plane_axis_idx[1]])
        ax.set_xticks(np.linspace(self.xyz_ranges[plane_axis_idx[0]][0], self.xyz_ranges[plane_axis_idx[0]][1], self.num_grid[plane_axis_idx[0]]+1))
        ax.set_yticks(np.linspace(self.xyz_ranges[plane_axis_idx[1]][0], self.xyz_ranges[plane_axis_idx[1]][1], self.num_grid[plane_axis_idx[1]]+1))
        ax.set_box_aspect(1)
        ax.set_xlabel(coord_names[plane_axis_idx[0]])
        ax.set_ylabel(coord_names[plane_axis_idx[1]])
        ax.set_title(f'Neuron activity at {coord_names[axis]} plane {plane_idx}, frame {frame}')
        ax.grid(visible=True, axis='both', color='gray', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()
        return ax

    def plot_colormesh_grid_activity(self, axis, plane_idx, frame, grid_activity, ax=None):

        if axis==2:
            coord1 = self.xyz_lines[0]
            coord2 = self.xyz_lines[1]
            act = grid_activity[:,:,plane_idx,frame]
            plane_axis_idx = (0, 1)
        elif axis==1:
            coord1 = self.xyz_lines[0]
            coord2 = self.xyz_lines[2]
            act = grid_activity[:,plane_idx,:,frame]
            plane_axis_idx = (0, 2)
        elif axis==0:
            coord1 = self.xyz_lines[1]
            coord2 = self.xyz_lines[2]
            act = grid_activity[plane_idx,:,:,frame]
            plane_axis_idx = (1, 2)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        coord_names = ['X', 'Y', 'Z']
        c = ax.pcolormesh(coord1, coord2, act.T, shading='auto', cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel(coord_names[plane_axis_idx[0]])
        ax.set_ylabel(coord_names[plane_axis_idx[1]])
        ax.set_box_aspect(1)
        ax.set_title(f'Grid activity at {coord_names[axis]} plane {plane_idx}, frame {frame}')

        plt.tight_layout()
        plt.show()
        return ax
    

class GridActivity:

    def __init__(self, recording_folder: str | Path, trial: str) -> None:
        """Load responses for one recording/trial.

        Parameters
        ----------
        recording_folder : str or Path
            Path to recording folder.
        trial : str
            Trial name or filename.
        """
        trial, ext = os.path.splitext(trial)
        trial = os.path.basename(trial)

        self.recording = os.path.basename(recording_folder)
        self.trial = trial
        self.data = np.load(
            os.path.join(recording_folder, "trials", trial + ".npy")
        )

        self.sampling_freq = 30
        self.valid_frames = np.shape(self.data)[-1]

        self.label = None
        self.ID = None
        self.grid = None

    def __copy__(self) -> Self:
        """Return a shallow copy of responses object.

        Returns
        -------
        Responses
            Shallow copy.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Return a deep copy of responses object.

        Parameters
        ----------
        memo : dict
            Memo dictionary used by ``copy.deepcopy``.

        Returns
        -------
        Responses
            Deep copy.
        """
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new

    def copy(self, deep: bool = False) -> Self:
        """Copy the responses object.

        Parameters
        ----------
        deep : bool, default=False
            If ``True``, return deep copy, otherwise shallow copy.

        Returns
        -------
        Responses
            Copied object.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def set_grid(self, num_grid, xyz_ranges):
        """Load grid metadata for current data.

        Parameters
        ----------
        num_grid : array-like
            Number of grid points in each dimension.
        xyz_ranges : array-like
            Ranges for each dimension.
        """
        self.grid = Grid3D(xyz_ranges, num_grid)
        
    def load_metadata(self, file_metadata: str | Path, verbose=True) -> None:
        """Load metadata for current response.

        Parameters
        ----------
        file_metadata : str or Path
            Path to a JSON file with the metadata.
        """

        try:
            with open(file_metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            if verbose:
                print(f"Warning. load_metadata: Could not load metadata: {e}")

        # check the metadata
        if self.label is not None and metadata["label"] != self.label:
            raise ValueError(
                "The metadata file contains a label different from the video"
            )
        if self.ID is not None and metadata["ID"] != self.ID:
            raise ValueError(
                "The metadata file contains an ID different from the video"
            )

        # add some other metadata
        if "valid_frames" in metadata.keys():
            self.valid_frames = metadata["valid_frames"]
        if "segments" in metadata.keys():
            self.segments = {}
            for k in metadata["segments"].keys():
                self.segments[k] = np.asarray(metadata["segments"][k])

    def get_data(self, normalization: str | None = None) -> np.ndarray:
        """Return response matrix with optional normalization.

        Parameters
        ----------
        normalization : {None, 'by_std', 'by_mean'}, optional
            Normalization strategy.

        Returns
        -------
        numpy.ndarray
            GridActivity array of shape ``(grid.num_grid, valid_frames)``.
        """
        if normalization is None:
            data = self.data.copy()
        else:
            raise ValueError("Normalization can have values: None, by_std, by_mean, or by_minmax")

        return data
            
    def plot_bar(self, axis, plane_idx, frame, ax=None):
        if self.grid is None:
            raise ValueError("Grid metadata not set. Call set_grid() first.")
        self.grid.plot_bar_grid_activity(axis, plane_idx, frame, self.get_data(), ax = ax)

    def plot_colormesh(self, axis, plane_idx, frame, ax=None):
        if self.grid is None:
            raise ValueError("Grid metadata not set. Call set_grid() first.")
        self.grid.plot_colormesh_grid_activity(axis, plane_idx, frame, self.get_data(), ax = ax)

    
class GridActivityData(GridActivity):
    def __init__(self, data, recording: str, trial: str) -> None:
        self.recording = recording
        self.trial = trial
        self.data = data
        self.sampling_freq = 30
        self.valid_frames = np.shape(self.data)[-1]
        self.label = None
        self.ID = None
        self.grid = None

class DataSetGrid(DataSet):

    def __init__(
        self,
        folder_data: str | Path,
        folder_metadata: str | Path | None = None,
        folder_derivatives: str | Path | None = None,
        recording: list[str] | str | None = None,
        check_data: bool = False,
        check_metadata: bool = False,
        check: bool = False,
        verbose: bool = True,            ):
        super().__init__(folder_data, 
                         folder_metadata=folder_metadata, 
                         recording=recording, 
                         check=check, 
                         check_data=check_data, 
                         check_metadata=check_metadata, 
                         verbose=verbose)
        
        # path to the output folder for the grids activity per trial
        self.folder_derivatives = Path(folder_derivatives) if folder_derivatives is not None else None
        
        # load the grids for each recording
        self.load_grids()
        
    def load_grids(self, recording: str | list[str] | None = None, verbose: bool = True):

        if recording is None:
            recording = self.recording
        elif isinstance(recording, str):
            recording = [recording]
        
        if self.folder_derivatives is None:
            raise ValueError("folder_derivatives must be set to load the grids.")

        for rec in self.recording:
            folder_output_rec = self.folder_derivatives / rec / 'grid'
            num_grid = np.load(folder_output_rec / 'num_grid.npy')
            xyz_ranges = np.load(folder_output_rec / 'xyz_ranges.npy')
            self.info[rec]["grid"] = Grid3D(xyz_ranges, num_grid)


    def load_gridactivity_by_trial(
        self,
        recording: str,
        trial: str,
        verbose: bool = True,
        try_loading_globalmetadata: bool = False,
        ) -> GridActivity:
        """Load one trial grid activity object and attach metadata.

        Returns
        -------
        GridActivity
            Loaded activity object.
        """

        if self.folder_derivatives is None:
            raise ValueError("folder_derivatives must be set to load the grid.")

        # load the data
        recording_folder = os.path.join(self.folder_derivatives, recording)
        gridactivity = GridActivity(recording_folder, trial)

        # lookup trial metadata
        trials_meta = self.filter_trials(recording=recording, trial=trial)
        if len(trials_meta) != 1:
            raise Exception(f"{len(trials_meta)} trials found, instead of only 1 ")
        if "ID" in trials_meta.columns:
            gridactivity.ID = trials_meta["ID"].iloc[0]
        if "label" in trials_meta.columns:
            gridactivity.label = trials_meta["label"].iloc[0]

        # get the grid
        gridactivity.grid = self.info[recording]["grid"]

        # try loading metadata from global metadata folder (if configured)
        if self.folder_globalmetadata_videos is not None and try_loading_globalmetadata:
            try:
                file = get_file_with_pattern(
                    f"*-{gridactivity.ID}.json", self.folder_globalmetadata_videos
                )
                gridactivity.load_metadata(file)
                return gridactivity
            except Exception as e:
                if verbose:
                    print(
                        f"Loading metadata from {self.folder_globalmetadata_videos} failed: {e}"
                    )

        return gridactivity
    
    def load_gridactivity_by(
        self,
        verbose: bool = True,
        query: str | None = None,
        **conditions,
    ) -> tuple[list[GridActivity], pd.DataFrame]:
        """Load responses for all trials matching filters.

        Returns
        -------
        tuple[list[GridActivity], pandas.DataFrame]
            Loaded objects and the filtered trial table.
        """

        trials_df = self.filter_trials(**conditions, query=query)
        gridactivities = []
        for index, row in trials_df.iterrows():
            act = self.load_gridactivity_by_trial(
                recording=row["recording"], trial=row["trial"], verbose=verbose
            )
            gridactivities.append(act)

        return gridactivities, trials_df
