from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Config:
    model_path: str = "data/face_landmarker.task"
    video_path: str = "data/video.mp4"
    music_path: str = "data/video.wav"
    look_away_limit: float = 0.5
    threshold_down: float = 0.5
    threshold_side: float = 0.6
    threshold_up: float = 0.5
    threshold_head_pitch: float = 0.30
    window_title: str = "Anty-Doom-Scrolling"
    penalty_window_title: str = "VIDEO PENALTY"
    penalty_frame_size: Tuple[int, int] = (800, 600)


def resolve_config(base_dir: Path) -> Config:
    resolved_base = base_dir.resolve()
    data_dir = resolved_base / "data"
    return Config(
        model_path=str(data_dir / "face_landmarker.task"),
        video_path=str(data_dir / "video.mp4"),
        music_path=str(data_dir / "video.wav"),
    )
