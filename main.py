from trackers import Tracker
import cv2
from utils import read_video, save_video, read_video_batch

def main():
    # read video
    print("READING...")
    video_path = "./data/input/08fd33_4.mp4"
    tracker = Tracker("./models/best.pt")
    # save video
    saved_video_path = "./data/output/output_video.avi"
    start = 0

    # frames_info = tracker.detect_frames(video_frames) # type: ignore
    print("TRACKING...")
    tracks = tracker.get_obj_tracks([], read_from_stub=True, stub_path="./stubs/track_stubs.pkl")

    batch_size = 50
    writer = None

    for frames_batch in read_video_batch(video_path, batch_size):
        end = start + len(frames_batch)

        print( f"Processing frames " f"{start}–{end - 1}")
        track_batch = {
            object_type: tracks[object_type][start:end]
            for object_type in tracks
        }

        annotated_batch = tracker.draw_annotation(frames_batch, track_batch)

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

        del annotated_batch
        del track_batch

        start = end

    if writer is not None:
        writer.release()
    # save_video(processed_frames, saved_video_path)

if __name__ == '__main__':
    main()
