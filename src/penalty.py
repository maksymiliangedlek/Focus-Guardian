from typing import Optional, Tuple

import cv2
import pygame


def init_audio(music_path: str) -> None:
    pygame.mixer.init()
    pygame.mixer.music.load(music_path)


class PenaltyController:
    def __init__(self, video_path: str, window_title: str, frame_size: Tuple[int, int]):
        self.video_path = video_path
        self.window_title = window_title
        self.frame_size = frame_size
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)

    def stop(self) -> None:
        pygame.mixer.music.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        try:
            cv2.destroyWindow(self.window_title)
        except Exception:
            pass

    def read_frame(self):
        if self._cap is None:
            return None

        ret, video_frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, video_frame = self._cap.read()

        if not ret:
            return None

        return cv2.resize(video_frame, self.frame_size)

