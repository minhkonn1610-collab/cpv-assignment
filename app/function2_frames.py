from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from . import storage
from .config import MAX_IMAGES_PER_STUDENT, MIN_FRAME_DIFFERENCE, get_camera_backend


def mirror_frame(frame):
    return cv2.flip(frame, 1)


def resize_for_detection(frame, max_width: int = 640):
    if frame is None:
        raise ValueError("Frame đang bị None, không thể resize.")
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / width
    size = (max_width, int(height * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA), scale


def fit_preview(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Kích thước ảnh không hợp lệ.")
    scale = min(max_width / image.width, max_height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


class DatasetCollector:
    def __init__(
        self,
        root_dir: str | Path | None = None,
        max_images: int = MAX_IMAGES_PER_STUDENT,
        delay_seconds: float = 0.5,
        resize_width: Optional[int] = 640,
        mirror: bool = True,
        min_difference: float = MIN_FRAME_DIFFERENCE,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else storage.DATASET_DIR
        self.max_images = max_images
        self.delay_seconds = delay_seconds
        self.resize_width = resize_width
        self.mirror = mirror
        self.min_difference = min_difference
        self.cap: Optional[cv2.VideoCapture] = None
        self.saved_count = 0
        self.last_save_time = 0.0
        self.last_saved_gray: Optional[np.ndarray] = None
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_student_folder(self, student_id: str, student_name: str) -> Path:
        student_id = student_id.strip()
        student_name = student_name.strip().replace(" ", "_")
        if not student_id:
            raise ValueError("Mã sinh viên không được để trống.")
        if not student_name:
            raise ValueError("Họ tên sinh viên không được để trống.")
        folder_path = self.root_dir / f"{student_id}_{student_name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def count_images(self, folder_path: Path) -> int:
        return sum(1 for file_path in folder_path.iterdir() if file_path.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def next_image_path(self, student_id: str, student_name: str, prefix: str = "frame") -> Path:
        folder_path = self.create_student_folder(student_id, student_name)
        image_index = self.count_images(folder_path) + 1
        return folder_path / f"{prefix}_{image_index:03d}.jpg"

    def process_frame(self, frame):
        if frame is None:
            return None
        if self.mirror:
            frame = mirror_frame(frame)
        if self.resize_width is not None:
            frame, _ = resize_for_detection(frame, self.resize_width)
        return frame

    def is_duplicate(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        if self.last_saved_gray is None:
            self.last_saved_gray = gray
            return False
        score = float(np.mean(cv2.absdiff(gray, self.last_saved_gray)))
        if score < self.min_difference:
            return True
        self.last_saved_gray = gray
        return False

    def save_frame(self, frame, student_id: str, student_name: str, prefix: str = "frame", avoid_duplicate: bool = True) -> Optional[Path]:
        if frame is None:
            raise ValueError("Không có frame để lưu.")
        if avoid_duplicate and self.is_duplicate(frame):
            return None
        save_path = self.next_image_path(student_id, student_name, prefix)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Không encode được ảnh frame để lưu.")
        encoded.tofile(str(save_path))
        if not save_path.exists():
            raise ValueError(f"Không thể lưu ảnh: {save_path}")
        self.saved_count += 1
        return save_path

    def connect(self, source: str | int) -> bool:
        self.release()
        backend = get_camera_backend(source)
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            self.release()
            return False
        return True

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def auto_capture(self, source: str | int, student_id: str, student_name: str) -> int:
        if not self.connect(source):
            raise ValueError(f"Không mở được nguồn video: {source}")
        self.saved_count = 0
        self.last_save_time = 0.0
        self.last_saved_gray = None
        while self.saved_count < self.max_images:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            frame = self.process_frame(frame)
            now = time.time()
            if now - self.last_save_time >= self.delay_seconds and self.save_frame(frame, student_id, student_name) is not None:
                self.last_save_time = now
        self.release()
        return self.saved_count

    def extract_from_video(self, video_path: str, student_id: str, student_name: str, frame_interval: int = 15, max_images: Optional[int] = None) -> int:
        max_images = self.max_images if max_images is None else max_images
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Không mở được video: {video_path}")
        saved_count = 0
        frame_index = 0
        self.saved_count = 0
        self.last_saved_gray = None
        while saved_count < max_images:
            ret, frame = video.read()
            if not ret or frame is None:
                break
            if frame_index % frame_interval == 0:
                frame = self.process_frame(frame)
                if self.save_frame(frame, student_id, student_name, prefix="video", avoid_duplicate=True) is not None:
                    saved_count += 1
            frame_index += 1
        video.release()
        return saved_count
