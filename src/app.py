import time
from pathlib import Path

import cv2
import mediapipe as mp
import pygame

from config import resolve_config
from detector import analyze_focus, create_detector, draw_eye_boxes
from penalty import PenaltyController, init_audio


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config = resolve_config(project_root)

    detector = create_detector(config.model_path)
    cap = cv2.VideoCapture(0)
    init_audio(config.music_path)
    penalty = PenaltyController(
        config.video_path,
        config.penalty_window_title,
        config.penalty_frame_size,
    )

    look_away_start = None
    penalty_active = False

    print("FocusGuardian is active...")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            is_looking_at_screen = False
            status_text = "NO FACE!"
            color = (0, 0, 255)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                blendshapes = detection_result.face_blendshapes[0]
                is_looking_at_screen, status_text, color = analyze_focus(
                    landmarks, blendshapes, config
                )
                draw_eye_boxes(frame, landmarks, color)

            if is_looking_at_screen:
                look_away_start = None
                if penalty_active:
                    penalty_active = False
                    penalty.stop()
                    print("You cameback! Video stopped")
            else:
                if look_away_start is None:
                    look_away_start = time.time()

                elapsed = time.time() - look_away_start
                status_text += f" | {round(elapsed, 1)}s"

                if elapsed > config.look_away_limit:
                    penalty_active = True

            cv2.putText(
                frame,
                status_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.imshow(config.window_title, frame)

            if penalty_active:
                penalty.start()
                video_frame = penalty.read_frame()
                if video_frame is not None:
                    cv2.imshow(config.penalty_window_title, video_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        penalty.stop()
        pygame.mixer.quit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
