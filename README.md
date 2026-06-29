# CPV — Face Attendance System
**Subject**: Computer Vision | **University**: FPT University

---

## Team Members

| Student ID | Full Name |
|---|---|
| HE190019 | Tran Van Truong |
| HE191357 | Nguyen Huy Son |
| HE200457 | Le Trung Kien |
| HE200629 | Le Trung Hieu |
| HE204132 | Hoang Trung Hieu |
| HE204913 | Nguyen Quang Minh |

---

## Project Structure

```
project-folder/               ← repository root
├── train.py                  Training pipeline entry point
├── main.py                   Inference pipeline entry point
├── self_check.py             Automated sanity test
├── requirements.txt          Python dependencies
├── assignment.md             Assignment specification
├── running-guide.md          Cross-platform running guide (macOS / Linux / Windows)
│
├── app/                      Source package
│   ├── config.py             All shared constants (IMAGE_SIZE, thresholds, …)
│   ├── storage.py            Path constants + file I/O helpers
│   ├── function1_stream.py   F1: threaded RTSP / webcam stream
│   ├── function2_frames.py   F2: video → frame extraction + dataset collection
│   ├── function3_detection.py F3: Haar Cascade face detector
│   ├── function4_recognition.py F4: centroid classifier (train + predict)
│   ├── face.py               Facade combining F3 + F4
│   └── ui.py                 Desktop UI (CustomTkinter)
│
├── dataset/                  Face images per student (gitignored)
├── data/                     Trained model.pkl (gitignored)
├── videos/                   Source .MOV recordings (gitignored)
└── attendance.csv            Output attendance log (gitignored)
```

---

## Quick Start

```bash
# 1. Activate the virtual environment
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

# 2. Install dependencies (first time only)
pip install -r requirements.txt
```

> 📖 For a detailed, step-by-step guide covering all operating systems, see **[running-guide.md](running-guide.md)**.

---

## Running Guide

See **[running-guide.md](running-guide.md)** for the full cross-platform guide, including:

- Environment setup on **macOS**, **Linux**, and **Windows**
- Dependency installation notes and platform-specific fixes
- Detailed training and inference instructions with expected output
- Troubleshooting for common issues (camera, tkinter, FFMPEG, PowerShell, etc.)

---

## Pipeline 1 — Training (`python train.py`)

Converts raw `.MOV` video recordings into a trained recognition model.

```
videos/*.MOV
  → [F2] extract frames  →  dataset/{ID}_{Name}/*.jpg
  → [F3+F4] re-detect + vectorize + centroids  →  data/model.pkl
```

```bash
python train.py                  # full: extract frames + train
python train.py --skip-extract   # retrain on existing dataset only
python train.py --max-images 50  # save up to 50 frames per student
python train.py --interval 10    # denser frame sampling (every 10th frame)
```

**Step 1 — Frame Extraction** (`function2_frames.py`)
- Opens each `.MOV` with OpenCV (FFMPEG backend)
- Samples 1 frame every 15 frames, mirrors, resizes to 640 px wide
- Skips near-duplicate frames (mean pixel diff < 8.0)
- Saves ≤ 30 `.jpg` images per student into `dataset/{ID}_{Name}/`

**Step 2 — Model Training** (`function3_detection.py` + `function4_recognition.py`)
- Re-detects face in each dataset image (Haar Cascade + CLAHE pre-processing)
- Vectorizes each 112×112 face: `equalizeHist → flatten → float32 / 255`
- Computes one centroid (mean vector) per student
- Auto-calibrates rejection threshold from training distance distribution
- Saves model to `data/model.pkl`

---

## Pipeline 2 — Inference (`python main.py`)

Runs real-time face recognition against the trained model.

```
webcam / RTSP
  → [F1] threaded stream
  → [F3] detect face (every 4th frame)
  → [F4] nearest-centroid prediction
  → attendance.csv
```

```bash
python main.py
```

**UI Modes**:

| Mode | Description |
|---|---|
| **Đăng ký** (Register) | Capture face images for a new student + trigger retraining |
| **Điểm danh** (Attendance) | Live recognition — each student logged once per session |
