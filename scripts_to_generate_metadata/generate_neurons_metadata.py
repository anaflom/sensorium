from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.dataset import DataSet


def main(folder_data, folder_meta):

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta)

    # create a folder for the outputs if it doesn't exists
    dataset.create_folders_metadata(what_global_data=[])

    # generate the metadata for all recordings
    dataset.generates_neurons_metadata()

    
if __name__ == "__main__":

    # path to the folder with the data as downloaded
    folder_data = repo_root / 'data'
    # path to the metadata folder
    folder_meta = repo_root / 'metadata'

    try:
        main(folder_data, folder_meta)
        print("\nNeurons metadata generation completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")