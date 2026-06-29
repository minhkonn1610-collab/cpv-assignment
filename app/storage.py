from __future__ import annotations

import csv
import pickle
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_SIZE  # noqa: F401 — re-exported for convenience

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
DATA_DIR = ROOT / "data"
STUDENTS_DIR = DATASET_DIR
MODEL_PATH = DATA_DIR / "model.pkl"
ATTENDANCE_PATH = ROOT / "attendance.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _clean(value: str) -> str:
    value = re.sub(r"[^\w\-. ]+", "", value.strip(), flags=re.UNICODE)
    return re.sub(r"\s+", "_", value)[:80]


def student_dir(student_id: str, name: str) -> Path:
    ensure_dirs()
    sid = _clean(student_id)
    sname = _clean(name)
    if not sid or not sname:
        raise ValueError("Vui lòng nhập mã sinh viên và họ tên")
    path = DATASET_DIR / f"{sid}_{sname}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_samples(student_id: str, name: str) -> int:
    folder = student_dir(student_id, name)
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_student_folder(path: Path) -> tuple[str, str] | None:
    parts = path.name.split("_", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1].replace("_", " ")


def image_paths(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_gray(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def read_color(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_face(face_img, student_id: str, name: str) -> Path:
    folder = student_dir(student_id, name)
    target = folder / f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    ok, encoded = cv2.imencode(".jpg", face_img)
    if not ok:
        raise ValueError("Không encode được ảnh khuôn mặt")
    encoded.tofile(str(target))
    if not target.exists():
        raise ValueError("Không lưu được ảnh khuôn mặt")
    return target


def save_model(model: object) -> None:
    ensure_dirs()
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)


def load_model() -> object | None:
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def log_attendance(student_id: str, name: str) -> None:
    exists = ATTENDANCE_PATH.exists()
    with ATTENDANCE_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["student_id", "name", "timestamp"])
        writer.writerow([student_id, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
