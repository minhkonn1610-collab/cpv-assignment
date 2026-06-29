from __future__ import annotations

import threading

import cv2

from .config import get_camera_backend

class CameraStream:
    """Function 1: stream video from webcam or RTSP camera."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self._cap: cv2.VideoCapture | None = None
        self.resolution = ""
        self._target_width = width
        self._target_height = height
        self._target_fps = fps
        self._frame = None
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def connect(self, source: str | int) -> bool:
        self.disconnect()
        backend = get_camera_backend(source)
        self._cap = cv2.VideoCapture(source, backend)
        if not self._cap.isOpened():
            self.disconnect()
            return False

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
            self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.disconnect()
            return False

        h, w = frame.shape[:2]
        self.resolution = f"{w}x{h}"
        self._frame = frame
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return True

    def _update(self) -> None:
        while self._running and self._cap:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                self._running = False

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
        self._cap = None
        self._thread = None
        self.resolution = ""
        self._frame = None

    @property
    def connected(self) -> bool:
        return self._running and bool(self._cap and self._cap.isOpened())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
