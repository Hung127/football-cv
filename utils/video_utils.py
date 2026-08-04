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

    cap.release()

    return frames

def read_video_batch(video_path: str, batch: int):
    cap = cv2.VideoCapture(video_path) 

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    frames = []
    ret = None

    while cap.isOpened():
        frames = []
        for i in range(batch):
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        yield frames
        if not ret:
            break
        del frames

    cap.release()


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

def save_frame(frames: list, writer: cv2.VideoWriter):
    if not frames:
        raise ValueError("Frames cannot be empty")

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer")

    for frame in frames:
        writer.write(frame)
