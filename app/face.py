from .function3_detection import FaceBox, FaceDetector
from .function4_recognition import Prediction, FaceRecognizer


class FaceEngine:
    def __init__(self, image_size: tuple[int, int] = (112, 112), threshold: float = 16.0) -> None:
        self.detector = FaceDetector(image_size=image_size)
        self.recognizer = FaceRecognizer(image_size=image_size, threshold=threshold)

    def detect_largest(self, frame):
        return self.detector.detect_largest(frame)

    def train(self) -> int:
        return self.recognizer.train()

    def predict(self, face_gray):
        return self.recognizer.predict(face_gray)
