from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import IMAGE_SIZE

@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int


class FaceDetector:
    """Function 3: detect and crop the largest face in one frame."""

    def __init__(self, image_size: tuple[int, int] = IMAGE_SIZE) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise ValueError("Không load được Haar Cascade nhận diện khuôn mặt")
        self.image_size = image_size

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def detect_largest(self, frame: np.ndarray) -> tuple[FaceBox | None, np.ndarray | None]:
        gray = self.preprocess(frame)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(70, 70))
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        face_crop = frame[y : y + h, x : x + w]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        return FaceBox(int(x), int(y), int(w), int(h)), self.normalize(face_gray)

    def normalize(self, face_gray: np.ndarray) -> np.ndarray:
        return cv2.resize(face_gray, self.image_size, interpolation=cv2.INTER_AREA)


def scale_box(box: FaceBox, scale: float) -> FaceBox:
    if scale == 1.0:
        return box
    inv = 1.0 / scale
    return FaceBox(int(box.x * inv), int(box.y * inv), int(box.w * inv), int(box.h * inv))
