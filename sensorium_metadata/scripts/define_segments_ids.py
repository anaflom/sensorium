import os
from pathlib import Path

from managedata.videos import VideoSegment
from managedata.videos_duplicates import (compute_dissimilarity_video_list, find_equal_sets_scipy, generate_new_id)
from managedata.handle_dataset import DataSet

def main(folder_data, folder_meta):

    # the labels to check
    labels = ['NaturalImages','GaussianDot','Gabor','PinkNoise','RandomDots']

    # set the limit to decide that two segments are the same or different
    limit_dissimilarity = 20

    # initialize a datset object to handle the data and metadata
    dataset = DataSet(folder_data, folder_metadata=folder_meta)

    # validate required folders
    if dataset.folder_globalmetadata_videos is None or dataset.folder_globalmetadata_segments is None:
        raise ValueError("folder_globalmetadata_videos and folder_globalmetadata_segments must be set")

    all_used_ids = []

    for lab in labels:

        print(f"\nFinding idenitcal segments for label {lab}")
        print("--------------------------------------------------")
        
        all_segments = []
        folder = Path(dataset.folder_globalmetadata_videos)
        json_files = list(folder.glob(f"{lab}*.json"))
        print(f"- {len(json_files)} distint videos found")

        if len(json_files) == 0:
            print(f"Warning: No videos found for label {lab}")
            continue

        # load all segments
        for file_videoID in json_files:
            try:
                video_id = Path(file_videoID).stem.split('-')[1]
                video = dataset.load_video_by_id(video_id)

                if not hasattr(video, 'segments') or 'frame_start' not in video.segments:
                        print(f"Warning: Video {video_id} has no valid segments")
                        continue

                for seg_idx in range(len(video.segments['frame_start'])):
                    try: 
                        segment = VideoSegment(video, seg_idx)
                        segment.label_from_parentvideo()
                        all_segments.append(segment)
                    except Exception as e:
                        print(f"Warning: Could not load segment {seg_idx} from video {video_id}: {e}")
                        continue

            except Exception as e:
                print(f"Warning: Could not load video {video_id}: {e}")
                continue

        print(f"- {len(all_segments)} segments were found and loaded")

        if len(all_segments) == 0:
            print(f"Warning: No segments found for label {lab}")
            continue

        # compute dissimilarity
        try:
            print('Computing dissimilarity between segments...')
            dissimilarity = compute_dissimilarity_video_list(all_segments, dissimilarity_measure='mse', check_edges_first=False)
        except Exception as e:
            print(f"Error computing dissimilarity for label {lab}: {e}")
            continue

        # extract sets of identical segments
        mask = dissimilarity<=limit_dissimilarity
        list_identical = find_equal_sets_scipy(mask)
        print(f"- {len(list_identical)} different segments were found")

        # loop over identical segments and save metadata 
        print("Saving metadata...")
        for setiden in list_identical:
            try:
                # generate a new id
                the_id = generate_new_id(all_used_ids, prefix='s')
                all_used_ids.append(the_id)

                # generate a SegmentID object from the exemplar segment and add the duplicates
                segment_i_id = all_segments[next(iter(setiden))].copy(deep=True)
                segment_i_id.ID = the_id
                for k in setiden:
                    segment_i_id.add_duplicates( all_segments[k].parentvideo['ID'], all_segments[k].parentvideo['segment_index'])

                # save a json file with the video metadata
                segment_i_id.save_metadata(dataset.folder_globalmetadata_segments)
            
            except Exception as e:
                print(f"Error processing segment set: {e}")
                continue


if __name__ == "__main__":

    # path to the folder with the data as downloaded
    folder_data = '/home/anaflo/MDMC/thesis/sensorium/data/'
    # path to the metadata folder
    folder_meta = os.path.join('..','metadata')

    try:
        main(folder_data, folder_meta)
        print("\nSegment ID definition completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")