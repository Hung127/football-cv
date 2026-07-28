import cv2

def read_video(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path) 

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    return frames

def save_video(frames: list, path: str):
    if not frames:
        raise ValueError("Frames cannot be empty")

    fourcc = cv2.VideoWriter.fourcc(*"XVID")

    height, width = frames[0].shape[:2]

    writer = cv2.VideoWriter(path,
                             fourcc,
                             24,
                             (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    for frame in frames:
        writer.write(frame)

    writer.release()
