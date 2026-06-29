from __future__ import annotations

import time

import cv2
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont, ImageTk

from . import storage
from .face import FaceBox, FaceEngine, Prediction
from .function1_stream import CameraStream
from .function2_frames import fit_preview, mirror_frame, resize_for_detection
from .function3_detection import scale_box

import platform

def load_system_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    sys_name = platform.system()
    paths = []
    if sys_name == "Windows":
        paths = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
    elif sys_name == "Darwin":
        paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf"
        ]
    else:
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ]

    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


class FaceAttendanceApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        storage.ensure_dirs()
        self.title("Face Attendance - Điểm danh khuôn mặt")
        self.geometry("1180x720")
        self.minsize(1000, 640)

        self.camera = CameraStream()
        self.face_engine = FaceEngine()
        self.mode = ctk.StringVar(value="Đăng ký")
        self.current_face = None
        self.last_face_time = 0.0
        self.frame_index = 0
        self.detect_every = 4
        self.cached_box: FaceBox | None = None
        self.cached_label = ""
        self.cached_color = (51, 65, 85)
        self.cached_face_seen = False
        self.logged_ids: set[str] = set()
        self.video_image: ImageTk.PhotoImage | None = None
        self.font = load_system_font(22)

        self.last_status = "Sẵn sàng"
        self.running = True
        self.train_button: ctk.CTkButton | None = None
        self.clear_session_button: ctk.CTkButton | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(30, self.update_frame)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self, corner_radius=0)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="Camera", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=14, pady=12)
        self.url_entry = ctk.CTkEntry(top, placeholder_text="RTSP URL hoặc Index webcam (mặc định 0)")

        self.url_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=12)
        ctk.CTkButton(top, text="Mở webcam", command=self.connect_webcam, width=110).grid(row=0, column=2, padx=6, pady=12)
        ctk.CTkButton(top, text="Kết nối RTSP", command=self.connect_camera, width=120).grid(row=0, column=3, padx=6, pady=12)
        ctk.CTkButton(top, text="Ngắt", command=self.disconnect_camera, width=90, fg_color="#334155").grid(row=0, column=4, padx=(6, 14), pady=12)

        body = ctk.CTkFrame(self, corner_radius=0, fg_color="#0f172a")
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(body, text="Chưa kết nối camera", fg_color="#020617", corner_radius=8)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)

        side = ctk.CTkFrame(body, width=320, corner_radius=8)
        side.grid(row=0, column=1, sticky="ns", padx=(0, 14), pady=14)
        side.grid_propagate(False)

        ctk.CTkLabel(side, text="Bảng điều khiển", font=("Arial", 18, "bold")).pack(anchor="w", padx=16, pady=(16, 8))
        self.mode_button = ctk.CTkSegmentedButton(side, values=["Đăng ký", "Điểm danh"], variable=self.mode, command=self.set_mode)
        self.mode_button.pack(fill="x", padx=16, pady=(0, 14))

        self.form = ctk.CTkFrame(side, fg_color="transparent")
        ctk.CTkLabel(self.form, text="Mã sinh viên").pack(anchor="w")
        self.student_id_entry = ctk.CTkEntry(self.form, placeholder_text="VD: SE123456")
        self.student_id_entry.pack(fill="x", pady=(4, 10))
        ctk.CTkLabel(self.form, text="Họ tên").pack(anchor="w")
        self.name_entry = ctk.CTkEntry(self.form, placeholder_text="VD: Nguyễn Văn A")
        self.name_entry.pack(fill="x", pady=(4, 10))
        ctk.CTkButton(self.form, text="Chụp ảnh khuôn mặt", command=self.capture_face).pack(fill="x", pady=(2, 8))
        self.sample_label = ctk.CTkLabel(self.form, text="Khuyến nghị: 10 ảnh / người")
        self.sample_label.pack(anchor="w")

        self.action_frame = ctk.CTkFrame(side, fg_color="transparent")
        self.train_button = ctk.CTkButton(self.action_frame, text="Huấn luyện model", command=self.train_model, fg_color="#16a34a")
        self.clear_session_button = ctk.CTkButton(self.action_frame, text="Xóa lượt điểm danh phiên này", command=self.clear_session, fg_color="#334155")

        self.status_label = ctk.CTkLabel(side, text="Trạng thái", font=("Arial", 14, "bold"))
        self.status_label.pack(anchor="w", padx=16, pady=(8, 4))
        self.status_box = ctk.CTkTextbox(side, height=150, wrap="word")
        self.status_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.status_box.configure(state="disabled")

        self._sync_mode_ui()
        self.set_status("Sẵn sàng. Bấm Mở webcam để dùng camera laptop.")

    def _sync_mode_ui(self) -> None:
        self.form.pack_forget()
        self.action_frame.pack_forget()
        if self.train_button:
            self.train_button.pack_forget()
        if self.clear_session_button:
            self.clear_session_button.pack_forget()

        if self.mode.get() == "Đăng ký":
            self.form.pack(fill="x", padx=16, after=self.mode_button)
            self.action_frame.pack(fill="x", padx=16, pady=(18, 14), after=self.form)
            if self.train_button:
                self.train_button.pack(fill="x")
            return

        self.action_frame.pack(fill="x", padx=16, pady=(18, 14), after=self.mode_button)
        if self.clear_session_button:
            self.clear_session_button.pack(fill="x")
    def connect_webcam(self) -> None:
        val = self.url_entry.get().strip()
        idx = 0
        if val:
            try:
                idx = int(val)
            except ValueError:
                pass

        self.set_status(f"Đang mở webcam laptop (index {idx})...")
        if self.camera.connect(idx):
            self.set_status(f"Đã mở webcam (index {idx}) ({self.camera.resolution})")
        else:
            self.set_status(f"Lỗi: không mở được webcam (index {idx})")


    def connect_camera(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            self.set_status("Lỗi: chưa nhập RTSP URL")
            return
        if not url.lower().startswith("rtsp://"):
            self.set_status("Lỗi: URL phải bắt đầu bằng rtsp://")
            return
        self.set_status("Đang kết nối camera RTSP...")
        self.set_status("Đã kết nối RTSP stream" if self.camera.connect(url) else "Lỗi: không mở được RTSP stream")

    def disconnect_camera(self) -> None:
        self.camera.disconnect()
        self.current_face = None
        self.last_face_time = 0.0
        self.cached_box = None
        self.cached_label = ""
        self.cached_face_seen = False
        self.video_image = None
        self.video_label.configure(image="", text="Chưa kết nối camera")
        self.set_status("Đã ngắt camera")

    def set_mode(self, value: str) -> None:
        self.current_face = None
        self.last_face_time = 0.0
        self.cached_box = None
        self.cached_label = ""
        self.cached_face_seen = False
        self._sync_mode_ui()
        if value == "Đăng ký":
            self.set_status("Chế độ đăng ký: nhập MSSV, họ tên, rồi chụp ảnh.")
        else:
            self.set_status("Chế độ điểm danh: khuôn mặt đã học sẽ tự ghi CSV.")

    def capture_face(self) -> None:
        if self.current_face is None or time.monotonic() - self.last_face_time > 2:
            self.set_status("Lỗi: chưa phát hiện khuôn mặt để chụp")
            return
        student_id = self.student_id_entry.get().strip()
        name = self.name_entry.get().strip()
        try:
            saved = storage.save_face(self.current_face, student_id, name)
            count = storage.count_samples(student_id, name)
        except ValueError as exc:
            self.set_status(f"Lỗi: {exc}")
            return
        self.sample_label.configure(text=f"Đã lưu {count} ảnh. Khuyến nghị: 10 ảnh / người")
        self.set_status(f"Đã lưu ảnh {count}: {saved.name}")

    def train_model(self) -> None:
        if self.train_button:
            self.train_button.configure(state="disabled")
        self.set_status("Đang huấn luyện model... Vui lòng đợi (không tắt ứng dụng)")

        import threading
        def run_training():
            try:
                count = self.face_engine.train()
                self.after(0, lambda: self.on_training_complete(count))
            except Exception as exc:
                self.after(0, lambda: self.on_training_failed(exc))

        threading.Thread(target=run_training, daemon=True).start()

    def on_training_complete(self, count: int) -> None:
        self.set_status(f"Huấn luyện xong {count} ảnh. Model: data/model.pkl")
        if self.train_button:
            self.train_button.configure(state="normal")

    def on_training_failed(self, exc: Exception) -> None:
        self.set_status(f"Lỗi: {exc}")
        if self.train_button:
            self.train_button.configure(state="normal")


    def clear_session(self) -> None:
        self.logged_ids.clear()
        self.set_status("Đã xóa danh sách điểm danh trong phiên hiện tại")

    def update_frame(self) -> None:
        if not self.running or not self.winfo_exists():
            return
        if self.camera.connected:
            frame = self.camera.read()
            if frame is None:
                self.set_status_once("Lỗi: mất tín hiệu camera")
            else:
                self.render_frame(frame)
        self.after(30, self.update_frame)

    def render_frame(self, frame) -> None:
        frame = mirror_frame(frame)
        self.frame_index += 1
        if self.frame_index % self.detect_every == 0:
            self.update_detection(frame)

        # 1. Resize frame to fit screen preview first
        h_orig, w_orig = frame.shape[:2]
        target_w = max(320, self.video_label.winfo_width() - 8)
        target_h = max(240, self.video_label.winfo_height() - 8)

        scale_w = target_w / w_orig
        scale_h = target_h / h_orig
        scale = min(scale_w, scale_h)

        new_w = max(1, int(w_orig * scale))
        new_h = max(1, int(h_orig * scale))

        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Draw bounding box on the resized frame coordinates
        box = self.cached_box
        if box and self.cached_face_seen:
            bx = int(box.x * scale)
            by = int(box.y * scale)
            bw = int(box.w * scale)
            bh = int(box.h * scale)
            cv2.rectangle(resized_frame, (bx, by), (bx + bw, by + bh), self.cached_color, 2)

        # 3. Convert to RGB PIL Image
        rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # 4. Render text directly on the final image for perfect, crisp text
        if box and self.cached_face_seen:
            bx = int(box.x * scale)
            by = int(box.y * scale)
            ImageDraw.Draw(image).text((bx, max(4, by - 30)), self.cached_label, font=self.font, fill=self.cached_color[::-1])

        self.video_image = ctk.CTkImage(light_image=image, dark_image=image, size=(new_w, new_h))
        self.video_label.configure(image=self.video_image, text="")



    def update_detection(self, frame) -> None:
        detect_frame, scale = resize_for_detection(frame)
        box, face_img = self.face_engine.detect_largest(detect_frame)
        self.cached_box = scale_box(box, scale) if box else None
        self.cached_label = "Không thấy mặt"
        self.cached_color = (51, 65, 85)
        self.cached_face_seen = False
        if not box or face_img is None:
            return

        self.cached_color = (34, 197, 94)
        self.cached_label = "Đã phát hiện mặt"
        self.cached_face_seen = True
        if self.mode.get() == "Đăng ký":
            self.current_face = face_img
            self.last_face_time = time.monotonic()
            return

        prediction = self.try_predict(face_img)
        self.cached_label = f"{prediction.student_id} - {prediction.name}" if prediction else "Unknown"

    def try_predict(self, face_img) -> Prediction | None:
        try:
            prediction = self.face_engine.predict(face_img)
        except ValueError as exc:
            self.set_status_once(f"Lỗi: {exc}")
            return None
        if not prediction:
            return None
        if prediction.student_id not in self.logged_ids:
            storage.log_attendance(prediction.student_id, prediction.name)
            self.logged_ids.add(prediction.student_id)
            self.set_status(f"Đã điểm danh: {prediction.student_id} - {prediction.name}")
        return prediction

    def set_status_once(self, message: str) -> None:
        if message != self.last_status:
            self.set_status(message)

    def set_status(self, message: str) -> None:
        self.last_status = message
        self.status_box.configure(state="normal")
        self.status_box.insert("end", message + "\n")
        self.status_box.see("end")
        self.status_box.configure(state="disabled")

    def on_close(self) -> None:
        self.running = False
        self.camera.disconnect()
        self.destroy()