# Running Guide — CPV Face Attendance System

> Complete step-by-step instructions to set up and run the project on **macOS**, **Linux**, and **Windows**.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Set Up the Python Environment](#3-set-up-the-python-environment)
4. [Install Dependencies](#4-install-dependencies)
5. [Run the Sanity Check](#5-run-the-sanity-check)
6. [Prepare Videos for Training](#6-prepare-videos-for-training)
7. [Pipeline 1 — Training](#7-pipeline-1--training)
8. [Pipeline 2 — Inference (Attendance App)](#8-pipeline-2--inference-attendance-app)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

| Tool | Minimum Version | Check command |
|------|----------------|---------------|
| Python | 3.9 | `python --version` or `python3 --version` |
| pip | 23.0 | `pip --version` |
| Git | any | `git --version` |
| Webcam / RTSP camera | — | required for inference |

> **Windows users:** Install Python from [python.org](https://www.python.org/downloads/).  
> Make sure to check **"Add Python to PATH"** during installation.

> **macOS users:** It is recommended to install Python via [Homebrew](https://brew.sh/):
> ```bash
> brew install python
> ```

> **Linux (Ubuntu/Debian) users:**
> ```bash
> sudo apt update && sudo apt install python3 python3-pip python3-venv -y
> ```

---

## 2. Clone the Repository

```bash
git clone https://github.com/AnhNN04/fptu-mentor-sm09-cpv-minhnguyen.git
cd fptu-mentor-sm09-cpv-minhnguyen
```

---

## 3. Set Up the Python Environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> **PowerShell note:** If you get an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

After activation, your terminal prompt will be prefixed with `(.venv)`.

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.8.0 | Face detection (Haar Cascade), video I/O |
| `numpy` | ≥ 1.24.0 | Numerical operations, feature vectors |
| `Pillow` | ≥ 10.0.0 | Image processing utilities |
| `customtkinter` | ≥ 5.2.0 | Desktop UI framework |

> **Linux extra:** If OpenCV fails to open a display window, install system libraries:
> ```bash
> sudo apt install libgl1 libglib2.0-0 -y
> ```

> **macOS extra (Apple Silicon M1/M2/M3):** If `opencv-python` fails, try:
> ```bash
> pip install opencv-python-headless
> ```

---

## 5. Run the Sanity Check

Verify that the installation and core logic work correctly **before** using real data:

```bash
python self_check.py
```

Expected output:
```
self-check ok
```

If this passes, the environment is ready to use.

---

## 6. Prepare Videos for Training

Place your recorded `.MOV` video files inside the `videos/` directory.  
Each filename must match exactly:

```
videos/
├── HE190019_Tran_Van_Truong.MOV
├── HE191357_Nguyen_Huy_Son.MOV
├── HE200457_Le_Trung_Kien.MOV
├── HE200629_Le_Trung_Hieu.MOV
├── HE204132_Hoang_Trung_Hieu.MOV
└── HE204913_Nguyen_Quang_Minh.MOV
```

> If a video file is not found, that student will be skipped with a `[SKIP]` warning.  
> You can still train with only a subset of students.

---

## 7. Pipeline 1 — Training

### Full training (extract frames → train model)

```bash
python train.py
```

### Retrain on existing dataset (skip frame extraction)

```bash
python train.py --skip-extract
```

### Advanced options

```bash
# Save up to 50 frames per student (default: 30)
python train.py --max-images 50

# Sample every 10th frame instead of every 15th (denser)
python train.py --interval 10

# Use a custom video directory
python train.py --video-dir /path/to/your/videos

# Combine options
python train.py --max-images 50 --interval 10 --video-dir ./my-videos
```

### What happens during training

```
Step 1 — Frame Extraction (function2_frames.py)
  ├── Opens each .MOV with OpenCV (FFMPEG backend)
  ├── Samples 1 frame every 15 frames (default)
  ├── Mirrors + resizes frames to max 640 px wide
  ├── Skips near-duplicate frames (pixel diff < 8.0)
  └── Saves ≤ 30 JPEG images → dataset/{ID}_{Name}/

Step 2 — Model Training (function3_detection.py + function4_recognition.py)
  ├── Re-detects face in each image (Haar Cascade + CLAHE)
  ├── Vectorizes: equalizeHist → flatten → float32 / 255.0
  ├── Computes one centroid (mean vector) per student
  ├── Auto-calibrates the rejection threshold
  └── Saves model → data/model.pkl
```

### Expected output

```
============================================================
STEP 1 — Extract frames from source videos
============================================================
  Processing : HE190019_Tran_Van_Truong.MOV
  Student    : HE190019 — Tran Van Truong
  Saved      : 30 images  →  dataset/HE190019_Tran_Van_Truong
  ...

============================================================
STEP 2 — Train centroid recognition model
============================================================
  Images trained on : 180
  Model saved to    : data/model.pkl

============================================================
Training complete.
Run  python main.py  to start the attendance application.
============================================================
```

---

## 8. Pipeline 2 — Inference (Attendance App)

> **Prerequisite:** A trained model must exist at `data/model.pkl`.  
> Run `python train.py` first if it does not.

```bash
python main.py
```

The desktop UI will open with two modes:

| Mode (Vietnamese) | Description |
|---|---|
| **Đăng ký** (Register) | Capture face images for a new student directly via webcam and trigger retraining |
| **Điểm danh** (Attendance) | Live face recognition — each recognized student is logged once per session |

### Attendance output

Recognized students are logged to `attendance.csv` in the project root:

```
student_id,name,timestamp
HE190019,Tran Van Truong,2026-06-29 14:00:01
HE191357,Nguyen Huy Son,2026-06-29 14:00:15
...
```

Each student is recorded **only once per session**.

---

## 9. Troubleshooting

### Camera not detected

```
Error: Could not open webcam (index 0)
```

- Make sure your webcam is plugged in and not used by another app.
- Try changing the camera index in `app/config.py` or the stream URL.
- **Windows:** Ensure the app has camera permissions in *Settings → Privacy → Camera*.
- **macOS:** Allow camera access when macOS prompts you on first launch.

---

### `ModuleNotFoundError`

```
ModuleNotFoundError: No module named 'cv2'
```

- Make sure the virtual environment is activated (your prompt shows `(.venv)`).
- Reinstall dependencies: `pip install -r requirements.txt`.

---

### `No module named 'tkinter'`

**Linux only** — tkinter is not always bundled with Python:

```bash
sudo apt install python3-tk -y
```

---

### OpenCV cannot open `.MOV` files

- OpenCV requires **FFMPEG** support to read `.MOV` files.
- On Windows, `opencv-python` includes FFMPEG by default.
- On Linux, reinstall with FFMPEG support:
  ```bash
  pip install opencv-python
  ```

---

### Training fails with `[ERROR] No student folders found`

- Check that `dataset/` contains at least one folder with images.
- Run Step 1 first (`python train.py`) or place images manually into `dataset/{ID}_{Name}/`.

---

### Model not found when running `main.py`

```
FileNotFoundError: data/model.pkl not found
```

- Run the training pipeline first:
  ```bash
  python train.py
  ```
- If you have an existing dataset and want to skip extraction:
  ```bash
  python train.py --skip-extract
  ```

---

### PowerShell script execution blocked (Windows)

```
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

Fix:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate environment (macOS/Linux) | `source .venv/bin/activate` |
| Activate environment (Windows CMD) | `.venv\Scripts\activate.bat` |
| Activate environment (Windows PS) | `.venv\Scripts\Activate.ps1` |
| Install dependencies | `pip install -r requirements.txt` |
| Run sanity check | `python self_check.py` |
| Full training | `python train.py` |
| Retrain only | `python train.py --skip-extract` |
| Start attendance app | `python main.py` |
