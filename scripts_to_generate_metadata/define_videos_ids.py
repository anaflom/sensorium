from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.dataset import DataSet

def main(folder_data, folder_meta, folder_results):

    # parameters
    limit_dissimilarity = 5 # set the limit to decide that two videos are the same or different

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta, folder_metadata_per_trial=folder_results)

    # define IDs
    dataset.define_videos_id(limit_dissimilarity=limit_dissimilarity)


if __name__ == "__main__":

    # data folder
    folder_data = repo_root / 'data'
    # metadata folder
    folder_meta = repo_root / 'metadata'
    # results folder
    folder_results = repo_root / 'intermediate_results'

    try:
        main(folder_data, folder_meta, folder_results)
        print("\nVideo classification completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")