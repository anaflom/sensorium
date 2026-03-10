import assign_grid as ag
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator



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
    
    def neurons_in_plane_z(self, positions, plane_index):
        limits = (self.xyz_lines[2][plane_index], self.xyz_lines[2][plane_index+1])
        idx = np.where((positions[:, 2] >= limits[0]) & (positions[:, 2] < limits[1]))[0]
        return idx
    
    def save(self, folder):
        np.save(folder / 'num_grid.npy', self.num_grid)
        np.save(folder / 'xyz_ranges.npy', self.xyz_ranges)
    
    def plot_bar_neurons_activity(self, plane, frame, positions, activities, ax = None):
        
        neurons_idx = self.neurons_in_plane_z(positions, plane)

        x_flat = positions[neurons_idx, 0]
        y_flat = positions[neurons_idx, 1]
        z_flat = activities[neurons_idx, frame]

        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

        dx = 10   # bar width/depth
        dy = 10   # bar width/depth
        dz = z_flat     # bar height = z value

        ax.bar3d(
            x_flat,  # center the bars on x
            y_flat,  # center the bars on y
            np.zeros_like(z_flat),  # bars start at z=0
            dx, dy, dz,
            shade=True,
            color=plt.cm.viridis(z_flat / z_flat.max()),  # color by height
            alpha=0.9
        )

        ax.set_xlim(self.xyz_ranges[0])
        ax.set_ylim(self.xyz_ranges[1])
        ax.set_xticks(np.linspace(self.xyz_ranges[0][0], self.xyz_ranges[0][1], self.num_grid[0]+1))
        ax.set_yticks(np.linspace(self.xyz_ranges[1][0], self.xyz_ranges[1][1], self.num_grid[1]+1))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Neurons activity at plane {plane}, frame {frame}')

        plt.tight_layout()
        plt.show()
        return ax

    def plot_bar_grid_activity(self, plane, frame, grid_activity, ax = None):
        x = self.xyz_coordinates[0][:,:,plane]
        y = self.xyz_coordinates[1][:,:,plane]
        z = grid_activity[:,:,plane,frame]

        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

        x_flat = x.ravel()
        y_flat = y.ravel()
        z_flat = z.ravel()

        dx = 0.6 * (x[1,0] - x[0,0])   # bar width/depth
        dy = 0.6 * (y[0,1] - y[0,0])   # bar width/depth
        dz = z_flat     # bar height = z value

        ax.bar3d(
            x_flat,  # center the bars on x
            y_flat,  # center the bars on y
            np.zeros_like(z_flat),  # bars start at z=0
            dx, dy, dz,
            shade=True,
            color=plt.cm.viridis(z_flat / z_flat.max()),  # color by height
            alpha=0.9
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Grid activity at plane {plane}, frame {frame}')

        plt.tight_layout()
        plt.show()
        return ax

    def plot_scatter_neurons_activity(self, plane, frame, positions, activities, ax = None):
        
        neurons_idx = self.neurons_in_plane_z(positions, plane)

        x_flat = positions[neurons_idx, 0]
        y_flat = positions[neurons_idx, 1]
        z_flat = activities[neurons_idx, frame]

        if ax is None:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)

        scatter = ax.scatter(x_flat, y_flat, c=z_flat, cmap='viridis')
        fig.colorbar(scatter, ax=ax, label='Activity Level')

        ax.set_xlim(self.xyz_ranges[0])
        ax.set_ylim(self.xyz_ranges[1])
        ax.set_xticks(np.linspace(self.xyz_ranges[0][0], self.xyz_ranges[0][1], self.num_grid[0]+1))
        ax.set_yticks(np.linspace(self.xyz_ranges[1][0], self.xyz_ranges[1][1], self.num_grid[1]+1))
        ax.set_box_aspect(1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Neuron activity at plane {plane}, frame {frame}')
        ax.grid(visible=True, axis='both', color='gray', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()
        return ax

    def plot_colormesh_grid_activity(self, plane, frame, grid_activity, ax=None):
        x = self.xyz_lines[0]
        y = self.xyz_lines[1]
        z = grid_activity[:,:,plane,frame]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        c = ax.pcolormesh(x, y, z, shading='auto', cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_box_aspect(1)
        ax.set_title(f'Grid activity at plane {plane}, frame {frame}')

        plt.tight_layout()
        plt.show()
        return ax