from typing import Tuple

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import Config


LEFT_EYE_IDXS = (33, 133, 160, 159, 158, 144, 145, 153)
RIGHT_EYE_IDXS = (362, 263, 387, 386, 385, 373, 374, 380)
NOSE_TIP_IDX = 1
LEFT_EAR_IDX = 234
RIGHT_EAR_IDX = 454
CHIN_IDX = 152
TOP_HEAD_IDX = 10


def create_detector(model_path: str) -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_eye_bbox(landmarks, indices, width: int, height: int) -> Tuple[int, int, int, int]:
    x_coords = [int(landmarks[i].x * width) for i in indices]
    y_coords = [int(landmarks[i].y * height) for i in indices]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def analyze_focus(landmarks, blendshapes, config: Config) -> Tuple[bool, str, Tuple[int, int, int]]:
    bs_dict = {cat.category_name: cat.score for cat in blendshapes}

    look_down_score = (
        bs_dict.get("eyeLookDownLeft", 0) + bs_dict.get("eyeLookDownRight", 0)
    ) / 2
    look_up_score = (
        bs_dict.get("eyeLookUpLeft", 0) + bs_dict.get("eyeLookUpRight", 0)
    ) / 2
    look_side_score = max(
        bs_dict.get("eyeLookInLeft", 0),
        bs_dict.get("eyeLookOutLeft", 0),
        bs_dict.get("eyeLookInRight", 0),
        bs_dict.get("eyeLookOutRight", 0),
    )

    nose_y = landmarks[NOSE_TIP_IDX].y
    left_ear_y = landmarks[LEFT_EAR_IDX].y
    right_ear_y = landmarks[RIGHT_EAR_IDX].y
    chin_y = landmarks[CHIN_IDX].y
    top_head_y = landmarks[TOP_HEAD_IDX].y

    face_height = chin_y - top_head_y
    ears_y_avg = (left_ear_y + right_ear_y) / 2
    head_pitch_ratio = (nose_y - ears_y_avg) / face_height

    if look_down_score > config.threshold_down:
        status_text = f"LOOKING DOWN ({look_down_score:.2f})"
    elif look_side_score > config.threshold_side:
        status_text = f"LOOKING SIDE ({look_side_score:.2f})"
    elif look_up_score > config.threshold_up:
        status_text = f"LOOKING UP ({look_up_score:.2f})"
    elif head_pitch_ratio > config.threshold_head_pitch:
        status_text = f"HEAD DOWN ({head_pitch_ratio:.2f})"
    else:
        return True, "OK - you are focused", (0, 255, 0)

    return False, status_text, (0, 0, 255)


def draw_eye_boxes(frame, landmarks, color: Tuple[int, int, int]) -> None:
    h, w, _ = frame.shape
    lx, ly, lw, lh = get_eye_bbox(landmarks, LEFT_EYE_IDXS, w, h)
    cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), color, 2)

    rx, ry, rw, rh = get_eye_bbox(landmarks, RIGHT_EYE_IDXS, w, h)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
