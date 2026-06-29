"""
main.py — Inference Pipeline
==============================
Run from the detect-face/ directory:

    python main.py

Prerequisite: a trained model must exist at data/model.pkl.
If it does not exist, run the training pipeline first:

    python train.py

INFERENCE PIPELINE:
──────────────────────────────────────────────────────────────────────────────
  webcam (index 0) or RTSP stream (rtsp://...)
       │
       ▼  [F1] CameraStream (function1_stream.py)
          - cv2.VideoCapture with CAP_DSHOW (webcam) or CAP_FFMPEG (RTSP)
          - background thread continuously reads frames into a shared buffer
          - main UI thread calls .read() to get the latest frame instantly
       │
       ▼  every 4th frame:
          [F3] FaceDetector.detect_largest() (function3_detection.py)
          - preprocess: grayscale → CLAHE → GaussianBlur
          - Haar Cascade detectMultiScale
          - crop largest detected face → resize to IMAGE_SIZE (112×112)
       │
       ▼  [F4] FaceRecognizer.predict() (function4_recognition.py)
          - equalizeHist → flatten → float32 / 255.0 → 12 544-dim vector
          - Euclidean distance to each student centroid in model.pkl
          - reject if distance > calibrated threshold (Unknown)
          - reject if gap between best and 2nd-best < AMBIGUITY_MARGIN
          - return Prediction(student_id, name, distance)
       │
       ▼  storage.log_attendance(student_id, name)
          - append row to attendance.csv (student_id, name, timestamp)
          - each student logged only once per UI session
──────────────────────────────────────────────────────────────────────────────
"""
from app.ui import FaceAttendanceApp


def main() -> None:
    app = FaceAttendanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
