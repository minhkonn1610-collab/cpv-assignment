from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app import storage
from app.face import FaceEngine


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        storage.DATASET_DIR = root / "dataset"
        storage.DATA_DIR = root / "data"
        storage.STUDENTS_DIR = storage.DATASET_DIR
        storage.MODEL_PATH = storage.DATA_DIR / "model.pkl"
        storage.ATTENDANCE_PATH = root / "attendance.csv"
        storage.ensure_dirs()

        a = storage.student_dir("HE001", "Nguyen Van A")
        b = storage.student_dir("HE002", "Tran Van B")
        for i in range(3):
            cv2.imwrite(str(a / f"a_{i}.jpg"), np.full((112, 112), 40 + i, dtype=np.uint8))
            cv2.imwrite(str(b / f"b_{i}.jpg"), np.full((112, 112), 210 - i, dtype=np.uint8))

        engine = FaceEngine(threshold=1.0)
        assert engine.train() == 6
        pred = engine.predict(np.full((112, 112), 40, dtype=np.uint8))
        assert pred is not None
        assert pred.student_id == "HE001"
        assert pred.name == "Nguyen Van A"

        storage.log_attendance(pred.student_id, pred.name)
        assert storage.ATTENDANCE_PATH.exists()

    print("self-check ok")


if __name__ == "__main__":
    main()

