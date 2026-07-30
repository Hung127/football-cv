from trackers import Tracker
from utils import read_video, save_video

def main():
    # read video
    print("READING...")
    video_path = "./data/input/08fd33_4.mp4"
    video_frames = read_video(video_path=video_path)

    tracker = Tracker("./models/best.pt")

    # frames_info = tracker.detect_frames(video_frames) # type: ignore
    print("TRACKING...")
    tracks = tracker.get_obj_tracks(video_frames, read_from_stub=True, stub_path="./stubs/track_stubs.pkl")

    print("DRAWING...")
    processed_frames = tracker.draw_annotation(video_frames, tracks)

    # save video
    print("SAVING...")
    saved_video_path = "./data/output/output_video.avi"
    save_video(processed_frames, saved_video_path)

if __name__ == '__main__':
    main()
