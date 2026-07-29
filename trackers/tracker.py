from ultralytics.models import YOLO
from ultralytics.engine.results import Results

class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = None

    def _partition(self, frames_info: list[Results]):
        """
        A function that helps me understand return format of YOLO prediction
        """
        if not frames_info:
            return

        class_names = frames_info[0].names
        class_names_inv = {class_names[k]:k for k in class_names}

        # print(class_names) # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        # print(class_names_inv) # {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}

        for idx, frame_info in enumerate(frames_info):
            data = frame_info.boxes.data # type: ignore -- format: [x1, y1, x2, y2, confidence, class_id]
            for obj in data:
                print(f"Frame: {idx}\nType: {class_names[int(obj[-1])]}\nBOX: {obj[:4]}\n")

    def detect_frames(self, frames: list):
        batch_size = 20
        detections = []
        length = len(frames)
        length = 10
        for i in range(0, length, batch_size):
            detection = self.model.predict(frames[i: i + batch_size], conf=0.1)
            detections += detection
        self._partition([detections[5]])
        return detections
