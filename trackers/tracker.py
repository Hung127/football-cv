from ultralytics.models import YOLO
import supervision as sv
import pickle
import os
import sys
import numpy as np
import cv2

sys.path.append("../") # get access to utils

from utils import get_center, get_width, get_height

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
                confidence = float(detection_with_tracks.confidence[i])

                if track_id is None:
                    continue

                if class_id == class_names_inv["player"]:
                    tracks["player"][frame_num][track_id] = {"bbox": bbox, "confidence": confidence}

                elif class_id == class_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox, "confidence": confidence}

            # Ball: use raw YOLO detections
            best_bbox = None
            best_conf = -1.0
            for i in range(len(detection_supervision)):
                class_id = detection_supervision.class_id[i]  # type: ignore

                if class_id != class_names_inv["ball"]:
                    continue

                confidence = float(detection_supervision.confidence[i])  # type: ignore

                if confidence > best_conf:
                    best_conf = confidence
                    best_bbox = detection_supervision.xyxy[i].tolist()

            if best_bbox is not None:
                tracks["ball"][frame_num][1] = {
                    "bbox": best_bbox,
                    "confidence": best_conf,
                }

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

    def draw_ellipse(self, frame, bbox: list):
        x_center, y_center = get_center(bbox)
        x1, y1, x2, y2 = map(int, bbox)
        width = get_width(bbox)
        center = (x_center, y2)
        axes = (width + 1, int(0.35 * width) + 1)
        cv2.ellipse(img=frame, center=center, axes=axes,
                    angle=0, startAngle=-45, endAngle=235,
                    color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

    def draw_triangle(self, frame, bbox: list):
        x_center, y_center = get_center(bbox)
        x1, y1, x2, y2 = map(int, bbox)
        width = get_width(bbox)
        center = (x_center, y1 - 10)
        top_left = (x1 - 10, y1 - 10 - width*2 - 1)
        top_right = (x2 + 10, y1 - 10 - width*2 - 1)
        points = np.array([center, top_left, top_right], np.int32)
        cv2.drawContours(frame, [points], -1, (0, 255, 0), thickness=-1)



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
                    if k == "player":
                        self.draw_ellipse(result_frame, bbox)
                    elif k == "ball":
                        self.draw_triangle(result_frame, bbox)
                    else:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            result_frames.append(result_frame)
        return result_frames
