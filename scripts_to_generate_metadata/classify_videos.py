from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.dataset import DataSet

def main(folder_data, folder_results):

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=None, folder_metadata_per_trial=folder_results)
    
    # clasiffy the videos for all recordings
    dataset.classify_videos()
    
    

if __name__ == "__main__":

    # base folder
    folder_data = repo_root / 'data'
    # results folder
    folder_results = repo_root / 'intermediate_results'

    try:
        main(folder_data, folder_results)
        print("\nVideo classification completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")