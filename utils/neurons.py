import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Neurons():

    def __init__(self, folder_metadata, recording):

        file = os.path.join(folder_metadata, recording, f"meta-neurons_{recording}.csv")
        try:
            df_neurons = pd.read_csv(file)
            self.coord_xyz = df_neurons[['coord_x', 'coord_y','coord_z']].copy().to_numpy()
            self.IDs = df_neurons['ID'].copy().to_numpy()
            self.stats_activity = {k:df_neurons[k].copy().to_numpy() for k in ['mean_activation', 'std_activation', 'median_activation', 'min_activation', 'max_activation'] if k in df_neurons.columns}
        except Exception as e:
            print(f"Error loading neurons for recording {recording}: {e}")


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


class NeuronsData(Neurons):

    def __init__(self, coord_xyz, IDs, stats_activity=None):

        self.coord_xyz = coord_xyz
        self.IDs = IDs
        self.stats_activity = stats_activity
