from trackers import Tracker
from utils import read_video, save_video

def main():
    # read video
    video_path = "./data/input/08fd33_4.mp4"
    video_frames = read_video(video_path=video_path)

    tracker = Tracker("./models/best.pt")

    frames_info = tracker.detect_frames(video_frames) # type: ignore

    # print(video_frames[0].shape)

    # save video
    # saved_video_path = "./data/output/output_video.avi"
    # save_video(video_frames, saved_video_path)

if __name__ == '__main__':
    main()
