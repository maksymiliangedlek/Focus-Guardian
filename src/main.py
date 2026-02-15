import time

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "data/face_landmarker.task"
VIDEO_PATH = "data/video.mp4"
MUSIC_PATH = "data/video.wav"
LOOK_AWAY_LIMIT = 0.5
THRESHOLD_DOWN = 0.5
THRESHOLD_SIDE = 0.6
THRESHOLD_UP = 0.5
THRESHOLD_HEAD_PITCH = 0.30

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=True,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
penalty_cap = None
look_away_start = None
penalty_active = False
pygame.mixer.init()
pygame.mixer.music.load(MUSIC_PATH)
LEFT_EYE_IDXS = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_IDXS = [362, 263, 387, 386, 385, 373, 374, 380]
NOSE_TIP_IDX = 1
LEFT_EAR_IDX = 234
RIGHT_EAR_IDX = 454
CHIN_IDX = 152
TOP_HEAD_IDX = 10


def get_eye_bbox(landmarks, indices, width, height):
    x_coords = [int(landmarks[i].x * width) for i in indices]
    y_coords = [int(landmarks[i].y * height) for i in indices]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


print("FocusGuardian is active...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    is_looking_at_screen = False
    color = (0, 0, 255)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        blendshapes = detection_result.face_blendshapes[0]
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

        if look_down_score > THRESHOLD_DOWN:
            status_text = f"LOOKING DOWN ({look_down_score:.2f})"
        elif look_side_score > THRESHOLD_SIDE:
            status_text = f"LOOKING SIDE ({look_side_score:.2f})"
        elif look_up_score > THRESHOLD_UP:
            status_text = f"LOOKING UP ({look_up_score:.2f})"
        elif head_pitch_ratio > THRESHOLD_HEAD_PITCH:
            status_text = f"HEAD DOWN ({head_pitch_ratio:.2f})"
        else:
            is_looking_at_screen = True
            status_text = "OK - you are focused"
            color = (0, 255, 0)

        lx, ly, lw, lh = get_eye_bbox(landmarks, LEFT_EYE_IDXS, w, h)
        cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), color, 2)

        rx, ry, rw, rh = get_eye_bbox(landmarks, RIGHT_EYE_IDXS, w, h)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)

        # ny = int(nose_y * h)
        # nx = int(landmarks[NOSE_TIP_IDX].x * w)
        # ey = int(ears_y_avg * h)
        # cv2.line(frame, (0, ey), (w, ey), (255, 255, 0), 1) # ear line
        # cv2.circle(frame, (nx, ny), 5, (255, 0, 255), -1)   # nose dot

    else:
        status_text = "NO FACE!"

    if is_looking_at_screen:
        look_away_start = None
        pygame.mixer.music.stop()
        if penalty_active:
            penalty_active = False
            if penalty_cap:
                penalty_cap.release()
                penalty_cap = None
            try:
                cv2.destroyWindow("VIDEO PENALTY")
            except:
                pass
            print("You cameback! Video stopped")
    else:
        if look_away_start is None:
            look_away_start = time.time()

        elapsed = time.time() - look_away_start
        status_text += f" | {round(elapsed, 1)}s"

        if elapsed > LOOK_AWAY_LIMIT:
            penalty_active = True

    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Anty-Doom-Scrolling", frame)

    if penalty_active:
        if penalty_cap is None:
            penalty_cap = cv2.VideoCapture(VIDEO_PATH)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)

        ret, video_frame = penalty_cap.read()

        if not ret:
            penalty_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, video_frame = penalty_cap.read()

        if ret:
            video_frame = cv2.resize(video_frame, (800, 600))
            cv2.imshow("VIDEO PENALTY", video_frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
if penalty_cap:
    penalty_cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
