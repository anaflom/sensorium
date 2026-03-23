import json
import os
from typing import Self, Any
from tqdm import tqdm

import assign_grid as ag
import numpy as np
import pandas as pd
from pathlib import Path
import copy


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.animation as animation
from IPython.display import HTML


from utils.data_handling import get_file_with_pattern
from utils.responses import Responses
from tda.dataset_derivatives import DataSetDerivatives



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
    # x_grid_coordinates = np.array([x for x in x_lin[:-1]])
    
    y_range = xyz_ranges[1]
    y_step = (y_range[1] - y_range[0]) / num_grid[1]
    y_lin = np.linspace(y_range[0], y_range[1], num_grid[1]+1)
    y_grid_coordinates = np.array([y+y_step/2 for y in y_lin[:-1]])
    # y_grid_coordinates = np.array([y for y in y_lin[:-1]])
    
    z_range = xyz_ranges[2]
    z_step = (z_range[1] - z_range[0]) / num_grid[2]
    z_lin = np.linspace(z_range[0], z_range[1], num_grid[2]+1)
    z_grid_coordinates = np.array([z+z_step/2 for z in z_lin[:-1]])
    # z_grid_coordinates = np.array([z for z in z_lin[:-1]])

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


# --- Plotting functions ---

def plot_colormesh_grid_activity(xyz_ranges, num_grid, axis, plane_idx, frame, grid_activity, ax=None):

    if axis==2:
        plane_axis_idx = (0, 1)
        act = grid_activity[:,:,plane_idx,frame]
    elif axis==1:
        plane_axis_idx = (0, 2)
        act = grid_activity[:,plane_idx,:,frame]
    elif axis==0:
        act = grid_activity[plane_idx,:,:,frame]
        plane_axis_idx = (1, 2)

    coord1 = np.linspace(xyz_ranges[plane_axis_idx[0]][0], xyz_ranges[plane_axis_idx[0]][1], num_grid[plane_axis_idx[0]]+1)
    coord2 = np.linspace(xyz_ranges[plane_axis_idx[1]][0], xyz_ranges[plane_axis_idx[1]][1], num_grid[plane_axis_idx[1]]+1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    coord_names = ['X', 'Y', 'Z']
    c = ax.pcolormesh(coord1, coord2, act.T, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_xticks(coord1)
    ax.set_yticks(coord2)
    ax.set_xticklabels(np.linspace(0, num_grid[plane_axis_idx[0]], num_grid[plane_axis_idx[0]]+1).astype(int))
    ax.set_yticklabels(np.linspace(0, num_grid[plane_axis_idx[1]], num_grid[plane_axis_idx[1]]+1).astype(int))
    ax.set_xlabel(coord_names[plane_axis_idx[0]])
    ax.set_ylabel(coord_names[plane_axis_idx[1]])
    ax.set_box_aspect(1)
    ax.set_title(f'Grid activity at {coord_names[axis]} plane {plane_idx}, frame {frame}')

    fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.15)
    fig.tight_layout()
    plt.show()

    return ax


def plot_bar_grid_activity(xyz_ranges, num_grid, axis, plane_idx, frame, grid_activity, ax = None):

    if axis==2:
        act = grid_activity[:,:,plane_idx,frame]
        plane_axis_idx = (0, 1)
        step_coord1 = (xyz_ranges[0][1] - xyz_ranges[0][0]) / num_grid[0]
        step_coord2 = (xyz_ranges[1][1] - xyz_ranges[1][0]) / num_grid[1]
    elif axis==1:
        act = grid_activity[:,plane_idx,:,frame]
        plane_axis_idx = (0, 2)
        step_coord1 = (xyz_ranges[0][1] - xyz_ranges[0][0]) / num_grid[0]
        step_coord2 = (xyz_ranges[2][1] - xyz_ranges[2][0]) / num_grid[2]
    elif axis==0:
        act = grid_activity[plane_idx,:,:,frame]
        plane_axis_idx = (1, 2)
        step_coord1 = (xyz_ranges[1][1] - xyz_ranges[1][0]) / num_grid[1]
        step_coord2 = (xyz_ranges[2][1] - xyz_ranges[2][0]) / num_grid[2]

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
  
    dx = 0.5 * step_coord1   # bar width/depth
    dy = 0.5 * step_coord2   # bar width/depth
    coord1_lines = np.linspace(xyz_ranges[plane_axis_idx[0]][0], xyz_ranges[plane_axis_idx[0]][1], num_grid[plane_axis_idx[0]]+1)
    coord2_lines = np.linspace(xyz_ranges[plane_axis_idx[1]][0], xyz_ranges[plane_axis_idx[1]][1], num_grid[plane_axis_idx[1]]+1)
    coord1_lines_ = coord1_lines[:-1] + step_coord1/2
    coord2_lines_ = coord2_lines[:-1] + step_coord2/2
    coord1, coord2 = np.meshgrid(coord1_lines_, coord2_lines_, indexing="ij")
    
    # coord1 = coord1.ravel() - dx/2
    # coord2 = coord2.ravel() - dy/2
    coord1 = coord1.ravel()
    coord2 = coord2.ravel() - dy
    dz = act.ravel() # bar height = z value

    ax.bar3d(
        coord1,  
        coord2,  
        np.zeros_like(dz), 
        dx, dy, dz,
        shade=True,
        color=plt.cm.viridis(dz / dz.max()),  
        alpha=0.9
    )

    coord_names = ['X', 'Y', 'Z']
    ax.set_xlim(xyz_ranges[plane_axis_idx[0]][0]-dx, xyz_ranges[plane_axis_idx[0]][1]+dx)
    ax.set_ylim(xyz_ranges[plane_axis_idx[1]][0]-dy, xyz_ranges[plane_axis_idx[1]][1]+dy)
    ax.set_xticks(coord1_lines)
    ax.set_yticks(coord2_lines)
    ax.set_xticklabels(np.linspace(0, num_grid[plane_axis_idx[0]], num_grid[plane_axis_idx[0]]+1).astype(int))
    ax.set_yticklabels(np.linspace(0, num_grid[plane_axis_idx[1]], num_grid[plane_axis_idx[1]]+1).astype(int))
    ax.set_zlabel('Grid activity')
    ax.set_xlabel(coord_names[plane_axis_idx[0]])
    ax.set_ylabel(coord_names[plane_axis_idx[1]])
    ax.set_title(f'Grid activity at {coord_names[axis]} plane {plane_idx}, frame {frame}')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.tight_layout()
    plt.show()
    return ax


def plot_bar_neurons_activity_in_plane(plane_ranges, plane_num_grid, plane_positions, plane_activities, coord_names, ax = None):
    
    if ax is None:
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    dx = 10   # bar width/depth
    dy = 10   # bar width/depth
    dz = plane_activities     # bar height = z value

    coord1 = plane_positions[:,0]
    coord2 = plane_positions[:,1]
    # coord1 = coord1 - dx/2
    # coord2 = coord2 - dy/2
    coord1 = coord1
    coord2 = coord2 - dy


    ax.bar3d(
        coord1,  
        coord2,  
        np.zeros_like(dz),  
        dx, dy, dz,
        shade=True,
        color=plt.cm.viridis(dz / dz.max()),  # color by height
        alpha=0.9
    )

    ax.set_xlim(plane_ranges[0][0]-dx, plane_ranges[0][1]+dx)
    ax.set_ylim(plane_ranges[1][0]-dy, plane_ranges[1][1]+dy)
    ax.set_xticks(np.linspace(plane_ranges[0][0], plane_ranges[0][1], plane_num_grid[0]+1))
    ax.set_yticks(np.linspace(plane_ranges[1][0], plane_ranges[1][1], plane_num_grid[1]+1))
    ax.set_xticklabels(np.linspace(0, plane_num_grid[0], plane_num_grid[0]+1).astype(int))
    ax.set_yticklabels(np.linspace(0, plane_num_grid[1], plane_num_grid[1]+1).astype(int))
    ax.set_xlabel(coord_names[0])
    ax.set_ylabel(coord_names[1])
    ax.set_zlabel('Neurons activity')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.tight_layout()
    plt.show()
    return ax
    

def plot_scatter_neurons_activity_in_plane(plane_ranges, plane_num_grid, plane_positions, plane_activities, coord_names, vmin=None, vmax=None, ax = None):
    

    coord1_flat = plane_positions[:, 0]
    coord2_flat = plane_positions[:, 1]
    act_flat = plane_activities

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    scatter = ax.scatter(coord1_flat, coord2_flat, c=act_flat, cmap='viridis',vmin=vmin, vmax=vmax)
    fig.colorbar(scatter, ax=ax, label='Activity Level')

    coord_names = ['X', 'Y', 'Z']
    ax.set_xlim(plane_ranges[0])
    ax.set_ylim(plane_ranges[1])
    ax.set_xticks(np.linspace(plane_ranges[0][0], plane_ranges[0][1], plane_num_grid[0]+1))
    ax.set_yticks(np.linspace(plane_ranges[1][0], plane_ranges[1][1], plane_num_grid[1]+1))
    ax.set_xticklabels(np.linspace(0, plane_num_grid[0], plane_num_grid[0]+1).astype(int))
    ax.set_yticklabels(np.linspace(0, plane_num_grid[1], plane_num_grid[1]+1).astype(int))
    ax.set_box_aspect(1)
    ax.set_xlabel(coord_names[0])
    ax.set_ylabel(coord_names[1])
    ax.grid(visible=True, axis='both', color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()
    return ax, scatter
    

# --- Classes ---

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

    def plot_activity_in_cell(self, cell_index, positions=None, activities=None, grid_activity=None):

        if positions is None and activities is None and grid_activity is None:
            raise ValueError("At least one of positions, activities, or grid_activity must be provided.")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        if positions is not None and activities is not None:
            neurons_idx = self.neurons_in_cell(positions, cell_index)
            ax.plot(activities[neurons_idx,:].T, linewidth=1, label=[f'neuron {i}' for i in neurons_idx])
        if grid_activity is not None:
            ax.plot(grid_activity[cell_index[0], cell_index[1], cell_index[2], :].flatten(), color='k', linewidth=2, label='grid activity')
        ax.set_xlabel('frames')
        ax.set_ylabel('activity')
        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.show()
        return fig, ax
    
    def plot_bar_neurons_activity(self, axis, plane_idx, frame, positions, activities, ax = None):
        
        neurons_idx, lines_plane, plane_axis_idx = self.neurons_in_plane(positions, axis, plane_idx)
        plane_axis_idx = np.asarray(plane_axis_idx)
        plane_ranges = self.xyz_ranges[plane_axis_idx]
        plane_num_grid = self.num_grid[plane_axis_idx]
        plane_positions = positions[neurons_idx, :][:, plane_axis_idx]
        plane_activities = activities[neurons_idx, frame]
        coord_names_all = ['X', 'Y', 'Z']
        coord_names = [coord_names_all[i] for i in plane_axis_idx]
        ax = plot_bar_neurons_activity_in_plane(plane_ranges, plane_num_grid, plane_positions, plane_activities, coord_names, ax = ax)
        ax.set_title(f'Neurons activity at {coord_names_all[axis]} plane {plane_idx}, frame {frame}')

        return ax


    def plot_scatter_neurons_activity(self, axis, plane_idx, frame, positions, activities, ax = None):

        neurons_idx, lines_plane, plane_axis_idx = self.neurons_in_plane(positions, axis, plane_idx)
        plane_axis_idx = np.asarray(plane_axis_idx)
        plane_ranges = self.xyz_ranges[plane_axis_idx]
        plane_num_grid = self.num_grid[plane_axis_idx]
        plane_positions = positions[neurons_idx, :][:, plane_axis_idx]
        plane_activities = activities[neurons_idx, frame]
        coord_names_all = ['X', 'Y', 'Z']
        coord_names = [coord_names_all[i] for i in plane_axis_idx]
        ax, scatter = plot_scatter_neurons_activity_in_plane(plane_ranges, plane_num_grid, plane_positions, plane_activities, coord_names, ax = ax)
        ax.set_title(f'Neuron activity at {coord_names_all[axis]} plane {plane_idx}, frame {frame}')
        
        return ax, scatter


    def plot_bar_grid_activity(self, axis, plane_idx, frame, grid_activity, ax = None):

        return plot_bar_grid_activity(self.xyz_ranges, self.num_grid, axis, plane_idx, frame, grid_activity, ax=ax)


    def plot_colormesh_grid_activity(self, axis, plane_idx, frame, grid_activity, ax=None):

        return plot_colormesh_grid_activity(self.xyz_ranges, self.num_grid, axis, plane_idx, frame, grid_activity, ax=ax)
    

    def animate_neurons_activity(self, axis, plane_idx, positions, activities,
                                interval_ms: int = 33, save_path: None | str | Path =None, display: bool = True):
        """
        Animate plot_bar_neurons_activity across all frames.
        """
        Nframes = activities.shape[-1]
        neurons_idx, lines_plane, plane_axis_idx = self.neurons_in_plane(positions, axis, plane_idx)
        plane_axis_idx = np.asarray(plane_axis_idx)
        plane_ranges = self.xyz_ranges[plane_axis_idx]
        plane_num_grid = self.num_grid[plane_axis_idx]
        plane_positions = positions[neurons_idx, :][:, plane_axis_idx]
        coord_names_all = ['X', 'Y', 'Z']
        coord_names = [coord_names_all[i] for i in plane_axis_idx]
        
        # compute global color limits for consistency across frames
        vmin, vmax = activities.min(), activities.max()

        # --- draw first frame ---
        def get_act(frame):
            return activities[neurons_idx, frame]      
        act0 = get_act(0)
        ax, scatter = plot_scatter_neurons_activity_in_plane(plane_ranges, plane_num_grid, plane_positions, act0, coord_names, vmin=vmin, vmax=vmax)
        fig = ax.get_figure()
        title = ax.set_title('')

        # --- update function ---
        def update(frame):
            scatter.set_array(get_act(frame))
            title.set_text(
                f'Neurons activity at {coord_names_all[axis]} plane {plane_idx}, frame {frame}'
            )
            return scatter, title

        anim = animation.FuncAnimation(fig, update, frames=Nframes,
                                    interval=interval_ms, blit=True)

        if save_path is not None:
            save_path = Path(save_path)
            writer = 'ffmpeg' if save_path.suffix == '.mp4' else 'pillow'
            anim.save(save_path, writer=writer)

        plt.close(fig)  # prevent duplicate static plot output

        if display:
            return HTML(anim.to_jshtml())
        else:
            return None


    def animate_grid_activity(self, axis, plane_idx, grid_activity, interval_ms: int = 33, 
                              save_path: None | str | Path =None, display: bool = True):
        """
        Animate plot_colormesh_grid_activity across all frames.

        Parameters
        ----------
        axis : int
            Axis perpendicular to the plane (0=X, 1=Y, 2=Z).
        plane_idx : int
            Index of the plane along the given axis.
        grid_activity : np.ndarray
            Array of shape (Nx, Ny, Nz, Nframes).
        interval_ms : int
            Delay between frames in milliseconds (default 33).
        save_path : str, optional
            If provided, save the animation to this path (e.g. 'anim.mp4' or 'anim.gif').
        display : bool, default=True
            If True, display the animation inline (e.g. in a Jupyter notebook).


        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
        """

        Nframes = grid_activity.shape[-1]
        if axis == 2:
            coord1, coord2 = self.xyz_lines[0], self.xyz_lines[1]
            plane_axis_idx = (0, 1)
        elif axis == 1:
            coord1, coord2 = self.xyz_lines[0], self.xyz_lines[2]
            plane_axis_idx = (0, 2)
        else:  # axis == 0
            coord1, coord2 = self.xyz_lines[1], self.xyz_lines[2]
            plane_axis_idx = (1, 2)
        coord_names = ['X', 'Y', 'Z']

        # compute global color limits for consistency across frames
        vmin, vmax = grid_activity.min(), grid_activity.max()

        # initialize figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # --- draw first frame ---
        def get_act(frame):
            if axis == 2:
                return grid_activity[:, :, plane_idx, frame]
            elif axis == 1:
                return grid_activity[:, plane_idx, :, frame]
            else:
                return grid_activity[plane_idx, :, :, frame]

        act0 = get_act(0)
        
        c = ax.pcolormesh(coord1, coord2, act0.T, shading='auto', 
                        cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(c, ax=ax)

        ax.set_xticks(np.linspace(self.xyz_ranges[plane_axis_idx[0]][0], self.xyz_ranges[plane_axis_idx[0]][1], self.num_grid[plane_axis_idx[0]]+1))
        ax.set_yticks(np.linspace(self.xyz_ranges[plane_axis_idx[1]][0], self.xyz_ranges[plane_axis_idx[1]][1], self.num_grid[plane_axis_idx[1]]+1))
        ax.set_xticklabels(np.linspace(0, self.num_grid[plane_axis_idx[0]], self.num_grid[plane_axis_idx[0]]+1).astype(int))
        ax.set_yticklabels(np.linspace(0, self.num_grid[plane_axis_idx[1]], self.num_grid[plane_axis_idx[1]]+1).astype(int))
        ax.set_xlabel(coord_names[plane_axis_idx[0]])
        ax.set_ylabel(coord_names[plane_axis_idx[1]])
        ax.set_box_aspect(1)

        title = ax.set_title('')
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.15)

        # --- update function ---
        def update(frame):
            c.set_array(get_act(frame).T.ravel())
            title.set_text(
                f'Grid activity at {coord_names[axis]} plane {plane_idx}, frame {frame}'
            )
            return c, title

        anim = animation.FuncAnimation(fig, update, frames=Nframes,
                                    interval=interval_ms, blit=True)
        
        if save_path is not None:
            save_path = Path(save_path)
            writer = 'ffmpeg' if save_path.suffix == '.mp4' else 'pillow'
            anim.save(save_path, writer=writer)

        plt.close(fig)  # prevent duplicate static plot output

        if display:
            return HTML(anim.to_jshtml())
        else:
            return None
        
        

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
        file_pattern = f"*rec-{self.recording}_trial-{self.trial}.npy"
        files = list(Path(recording_folder, "trials").glob(file_pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No file found for pattern {file_pattern} in {recording_folder}")
        elif len(files) > 1:
            raise ValueError(f"Multiple files found for pattern {file_pattern} in {recording_folder}: {[str(f) for f in files]}")
        self.data = np.load(files[0])

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

    def plot_activity_in_cell(self, cell_idx, positions=None, activities=None, normalization=None):
        if self.grid is None:
            raise ValueError("Grid metadata not set. Call set_grid() first.")
        fig, ax = self.grid.plot_activity_in_cell(cell_idx, positions=positions, activities=activities, grid_activity=self.get_data(normalization=normalization))

    def animate_grid_activity(self, axis, plane_idx, interval_ms: int = 33, save_path=None):
        if self.grid is None:
            raise ValueError("Grid metadata not set. Call set_grid() first.")
        return self.grid.animate_grid_activity(axis, plane_idx, self.get_data(), interval_ms=interval_ms, save_path=save_path)

    
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

class DataSetGrid(DataSetDerivatives):

    def __init__(
        self,
        folder_data: str | Path | None = None,
        folder_metadata: str | Path | None = None,
        folder_derivatives: str | Path | None = None,
        recording: list[str] | str | None = None,
        check_data: bool = False,
        check_metadata: bool = False,
        check: bool = False,
        verbose: bool = True,            ):
        super().__init__(folder_data=folder_data, 
                         folder_metadata=folder_metadata, 
                         folder_derivatives=folder_derivatives, 
                         recording=recording, 
                         check=check, 
                         check_data=check_data, 
                         check_metadata=check_metadata, 
                         verbose=verbose)
        
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


    def compute_grid_stats(self, recording: str, trials_for_stats: list | None = None, save: bool = False) -> dict:
        """Compute statistics of the grid activity for a given recording and trials.

        Parameters
        ----------
        recording : str
            Recording name.
        trials_for_stats : list or None, optional
            List of trial indices to include in the statistics. If None, include all trials.

        Returns
        -------
        dict
            Dictionary containing the computed statistics.
        """
        if self.folder_derivatives is None:
            raise ValueError("folder_derivatives must be set to compute grid stats.")

        if trials_for_stats is None:
            trials_for_stats = self.info[recording]["trials"]
        print(f"Computing grid stats for recording {recording}")
        stats = {}
        val_sum = np.zeros(self.info[recording]["grid"].num_grid)
        n = 0
        val_min = np.full(self.info[recording]["grid"].num_grid, np.inf)
        val_max = np.full(self.info[recording]["grid"].num_grid, -np.inf)
        trials_included = []
        for trial in tqdm(trials_for_stats, desc="MEAN, MAX, MIN computation", unit="trial",total=len(trials_for_stats), disable=False):
            try:
                grid_activity = self.load_gridactivity_by_trial(recording=recording, trial=trial, verbose=False)
                trials_included.append(trial)
            except Exception as e:
                print(f"Could not load grid activity for trial {trial}: {e}")
                continue
            data = grid_activity.get_data()
            val_sum += data.sum(axis=-1)
            n += data.shape[-1]
            val_min = np.minimum(val_min, data.min(axis=-1))
            val_max = np.maximum(val_max, data.max(axis=-1))
        stats["mean_activation"] = val_sum / n
        stats["min_activation"] = val_min
        stats["max_activation"] = val_max
        stats["trials_in_stats"] = trials_included

        val_sum_squared = np.zeros(self.info[recording]["grid"].num_grid)
        for trial in tqdm(trials_included, desc="STD computation", unit="trial",total=len(trials_included), disable=False):
            try:
                grid_activity = self.load_gridactivity_by_trial(recording=recording, trial=trial, verbose=False)
            except Exception as e:
                print(f"Could not load grid activity for trial {trial}: {e}")
                continue
            data = grid_activity.get_data()
            val_sum_squared += ((data - stats["mean_activation"][:,:,:,np.newaxis]) ** 2).sum(axis=-1)
        stats["std_activation"] = np.sqrt(val_sum_squared / n)

        if save:
            output_folder = self.folder_derivatives / recording / "gridactivity_stats"
            output_folder.mkdir(parents=True, exist_ok=True)
            for stat_name, stat_value in stats.items():
                np.save(output_folder / f"{stat_name}.npy", stat_value)

        return stats
    
    def load_grid_stats(self, 
                        stat_name: str | list[str] = ["mean_activation", "std_activation", "min_activation", "max_activation", "trials_in_stats"], 
                        recording: str | list[str] | None = None,  
                        keep_as_attribute: bool = True,
                        verbose: bool = True,
                        ) -> np.ndarray:
        """Load precomputed grid statistics.

        Parameters
        ----------
        stat_name : str | list[str]
            Name(s) of the statistic(s) to load (e.g., 'mean_activation').
        recording : str | list[str] | None, optional
            Recording name.
        keep_as_attribute : bool, optional
            If True, keep the loaded statistics as attributes of the DataSetGrid object.
        verbose : bool, optional
            If True, print warnings if loading fails.

        Returns
        -------
        numpy.ndarray
            Loaded statistic array.
        """

        if recording is None:
            recording = self.recording
        elif isinstance(recording, str):
            recording = [recording]
        if isinstance(stat_name, str):
            stat_name = [stat_name]

        if self.folder_derivatives is None:
            raise ValueError("folder_derivatives must be set to load grid stats.")

        stats = {}
        if keep_as_attribute:
            if not hasattr(self, "stats"):
                self.stats = {}
        for rec in recording:
            stats[rec] = {}
            for stat in stat_name:
                files = list((self.folder_derivatives / rec / "gridactivity_stats").glob(f"*{stat}.npy"))
                if len(files) == 0:
                    raise FileNotFoundError(f"No file found for statistic '{stat}' in {self.folder_derivatives / rec / 'grid_stats'}")
                elif len(files) > 1:
                    raise ValueError(f"Multiple files found for statistic '{stat}' in {self.folder_derivatives / rec / 'grid_stats'}: {[str(f) for f in files]}")
                stat_array = np.load(files[0])
                stats[rec][stat] = stat_array
                if keep_as_attribute:
                    if rec not in self.stats:
                        self.stats[rec] = {}
                    self.stats[rec][stat] = stat_array
        
        return stats

        
