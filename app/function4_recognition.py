from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import storage
from .config import AMBIGUITY_MARGIN, BASE_THRESHOLD, IMAGE_SIZE
from .function3_detection import FaceDetector


@dataclass(frozen=True)
class Prediction:
    student_id: str
    name: str
    distance: float


class FaceRecognizer:
    """Function 4: train and recognize student identity from a cropped face."""

    def __init__(self, image_size: tuple[int, int] = IMAGE_SIZE, threshold: float = BASE_THRESHOLD) -> None:
        self.image_size = image_size
        self.threshold = threshold
        self.detector = FaceDetector(image_size=image_size)
        self.model: dict | None = storage.load_model()

    def normalize(self, face_gray) -> np.ndarray:
        return cv2.resize(face_gray, self.image_size, interpolation=cv2.INTER_AREA)

    def vectorize(self, face_gray) -> np.ndarray:
        face_gray = cv2.equalizeHist(self.normalize(face_gray))
        return face_gray.astype("float32").reshape(-1) / 255.0

    def training_face(self, img_path):
        color = storage.read_color(img_path)
        if color is not None:
            _, face_gray = self.detector.detect_largest(color)
            if face_gray is not None:
                return face_gray
        return storage.read_gray(img_path)

    def train(self) -> int:
        storage.ensure_dirs()
        features_by_label: dict[str, list[np.ndarray]] = {}
        people: dict[str, tuple[str, str]] = {}

        for folder in storage.DATASET_DIR.iterdir():
            meta = storage.parse_student_folder(folder)
            if not folder.is_dir() or not meta:
                continue
            people[folder.name] = meta
            for img_path in storage.image_paths(folder):
                face_gray = self.training_face(img_path)
                if face_gray is not None:
                    features_by_label.setdefault(folder.name, []).append(self.vectorize(face_gray))

        if not features_by_label:
            raise ValueError("Chua co anh khuon mat de huan luyen")

        labels = list(features_by_label)
        centroids = np.asarray([np.mean(features_by_label[label], axis=0) for label in labels], dtype="float32")
        threshold = self.calibrated_threshold(features_by_label, labels, centroids)
        self.model = {"labels": labels, "centroids": centroids, "people": people, "threshold": threshold}
        storage.save_model(self.model)
        return sum(len(items) for items in features_by_label.values())

    def calibrated_threshold(self, features_by_label: dict[str, list[np.ndarray]], labels: list[str], centroids: np.ndarray) -> float:
        own_distances: list[float] = []
        wrong_distances: list[float] = []
        for label_index, label in enumerate(labels):
            for vector in features_by_label[label]:
                distances = np.linalg.norm(centroids - vector, axis=1)
                own_distances.append(float(distances[label_index]))
                if len(distances) > 1:
                    wrong_distances.append(float(np.min(np.delete(distances, label_index))))

        if not own_distances:
            return self.threshold
        own_limit = float(np.percentile(own_distances, 95)) * 1.15
        if not wrong_distances:
            return min(self.threshold, own_limit)
        wrong_floor = float(np.percentile(wrong_distances, 5)) * 0.85
        return max(4.0, min(self.threshold, own_limit, wrong_floor))

    def predict(self, face_gray) -> Prediction | None:
        if not self.model:
            self.model = storage.load_model()
        if not self.model or "centroids" not in self.model:
            raise ValueError("Chua co model moi. Hay bam Huan luyen model truoc")

        vector = self.vectorize(face_gray)
        centroids = self.model["centroids"]
        distances = np.linalg.norm(centroids - vector, axis=1)
        order = np.argsort(distances)
        best = int(order[0])
        distance = float(distances[best])
        threshold = float(self.model.get("threshold", self.threshold))
        if distance > threshold:
            return None
        if len(order) > 1 and float(distances[int(order[1])]) - distance < AMBIGUITY_MARGIN:
            return None
        label = self.model["labels"][best]
        student_id, name = self.model["people"][label]
        return Prediction(student_id, name, distance)
