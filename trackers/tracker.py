from ultralytics.models import YOLO
from ultralytics.engine.results import Results
import supervision as sv
import pickle
import os
import cv2

class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def get_obj_tracks(self, frames: list, read_from_stub = False, stub_path: str | None = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        tracks = {
            # frame_idx -> {track_id: {"bbox": [x1, y1, x2, y2]}}
            "player": [],

            # frame_idx -> {track_id: {"bbox": [x1, y1, x2, y2]}}
            "referee": [],

            # frame_idx -> {1: {"bbox": [x1, y1, x2, y2]}}  # single ball
            "ball": [],
        }

        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v:k for k, v in class_names.items()}

            # print(class_names) # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            # print(class_names_inv) # {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert all goalkeeper to player
            for idx, class_id in enumerate(detection_supervision.class_id): # type: ignore
                if class_id == class_names_inv["goalkeeper"]:
                    detection_supervision.class_id[idx] = class_names_inv["player"] # type: ignore

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for i in range(len(detection_with_tracks)):
                bbox = detection_with_tracks.xyxy[i].tolist()
                class_id = detection_with_tracks.class_id[i]
                track_id = detection_with_tracks.tracker_id[i]

                if track_id is None:
                    continue

                if class_id == class_names_inv["player"]:
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}

                elif class_id == class_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

            # Ball: use raw YOLO detections
            for i in range(len(detection_supervision)):
                bbox = detection_supervision.xyxy[i].tolist()
                class_id = detection_supervision.class_id[i] # type: ignore

                if class_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def detect_frames(self, frames: list):
        batch_size = 20
        detections = []
        length = len(frames)
        for i in range(0, length, batch_size):
            detection = self.model.predict(frames[i: i + batch_size], conf=0.1)
            detections += detection
        return detections

    def draw_annotation(self, frames: list, tracks: dict) -> list:
        # # frame_idx -> {track_id: {"bbox": [x1, y1, x2, y2]}}
        # "player": [],
        #
        # # frame_idx -> {track_id: {"bbox": [x1, y1, x2, y2]}}
        # "referee": [],
        #
        # # frame_idx -> {1: {"bbox": [x1, y1, x2, y2]}}  # single ball
        # "ball": [],
        result_frames = []
        for frame_num, frame in enumerate(frames):
            result_frame = frame.copy()
            for k in tracks:
                for track_id in tracks[k][frame_num]:
                    bbox = tracks[k][frame_num][track_id]["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            result_frames.append(result_frame)
        return result_frames

    # NOTE: test function
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

