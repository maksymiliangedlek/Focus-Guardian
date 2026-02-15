# 👁️ FocusGuardian: Anti-Doom-Scrolling App

FocusGuardian is a Python-based computer vision tool designed to break the habit of "doom-scrolling" or basically to manage your focus. Using MediaPipe's advanced Face Landmarker, the app monitors your focus in real-time. If you look away, check your phone (detected via head pitch), or look to the sides for too long, the app triggers a "penalty" video and audio loop to bring your attention back to your work(literally ;) ).

## ✨ Features

- **Real-Time Gaze Tracking**: Uses MediaPipe Blendshapes to detect if eyes are looking up, down, or sideways.
- **Head Posture Detection**: Analyzes facial geometry to detect when you're tilting your head down (typically to look at a phone).
- **Automated Penalty System**: 
  - Plays a video overlay using OpenCV.
  - Plays a synchronized audio loop using Pygame.
- **Dynamic Feedback**: On-screen bounding boxes for eyes that change color (Green/Red) based on your focus state.
- **Auto-Resume**: The penalty video and audio stop immediately when you return your focus to the screen.

## 🛠️ Tech Stack

- **Python 3.x**
- **MediaPipe Tasks API** (Face Landmarker & Blendshapes)
- **OpenCV** (Video capture and rendering)
- **Pygame** (Audio playback)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/maksymiliangedlek/FocusGuardian.git](https://github.com/maksymiliangedlek/FocusGuardian.git)
   cd FocusGuardian
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install opencv-python mediapipe pygame
   ```

4. **Prepare Data Folder:**
   You can change penalty video any time just by changing video files in a `data/` folder in the root directory:
   - `video.mp4`: Your choice of penalty video.
   - `video.wav`: Your choice of penalty audio/music.

## 🚀 How to Use

Simply run the main script:
```bash
python main.py
```

- **OK - you are focused**: Everything is fine (Green boxes).
- **LOOKING DOWN / SIDE / HEAD DOWN**: The timer starts (Red boxes).
- **Penalty Trigger**: If the state persists for more than `0.5s` (default), the video window pops up.

## ⚙️ Configuration

You can fine-tune the sensitivity in `main.py` to match your camera setup:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `LOOK_AWAY_LIMIT` | `0.5` | Seconds before the penalty starts. |
| `THRESHOLD_DOWN` | `0.5` | Eye gaze down sensitivity (0.0 - 1.0). |
| `THRESHOLD_HEAD_PITCH`| `0.30`| Head tilt sensitivity (Nose vs Ears position). |
| `THRESHOLD_SIDE` | `0.6` | Eye gaze side sensitivity. |

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.