
from managedata.dataset import DataSet

def main(folder_data, folder_results):

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=None, folder_metadata_per_trial=folder_results)
    
    # clasiffy the videos for all recordings
    dataset.clasiffy_videos()
    
    

if __name__ == "__main__":

    # base folder
    folder_data = '/home/anaflo/MDMC/thesis/sensorium/data/'
    # results folder
    folder_results = '/home/anaflo/MDMC/thesis/sensorium/intermediate_results/'

    try:
        main(folder_data, folder_results)
        print("\nVideo classification completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")