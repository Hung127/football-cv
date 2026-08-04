from trackers import Tracker
import cv2
from utils import read_video, read_video_batch

def main():
    # read video
    print("INITIALIZING...")
    video_path = "./data/input/08fd33_4.mp4"
    tracker = Tracker("./models/best.pt")
    # save video
    saved_video_path = "./data/output/output_video.avi"
    start = 0

    print("TRACKING...")
    tracks = tracker.get_obj_tracks([], read_from_stub=True, stub_path="./stubs/track_stubs.pkl")

    batch_size = 50
    writer = None

    # READING batches
    for frames_batch in read_video_batch(video_path, batch_size):
        end = start + len(frames_batch)

        # PROCESS
        print( f"Processing frames " f"{start}–{end - 1}")
        track_batch = {
            object_type: tracks[object_type][start:end]
            for object_type in tracks
        }

        annotated_batch = tracker.draw_annotation(frames_batch, track_batch)

        if not annotated_batch:
            continue

        # WRITE
        if writer is None:
            fourcc = cv2.VideoWriter.fourcc(*"XVID")
            height, width = annotated_batch[0].shape[:2]
            writer = cv2.VideoWriter(saved_video_path,
                                     fourcc,
                                     24,
                                     (width, height))
        if not writer.isOpened():
            raise ValueError("Invalid output path")

        print( f"Writing frames " f"{start}–{end - 1}")
        for frame in annotated_batch:
            writer.write(frame)

        # free up memory
        del annotated_batch
        del track_batch

        # update frame num
        start = end

    if writer is not None:
        writer.release()

if __name__ == '__main__':
    main()
