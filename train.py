"""
train.py — Training Pipeline
==============================
Run from the detect-face/ directory:

    python train.py                        # extract frames from videos/ then train
    python train.py --skip-extract         # skip extraction, retrain on existing dataset
    python train.py --video-dir ./videos --max-images 50 --interval 10

TRAINING PIPELINE:
──────────────────────────────────────────────────────────────────────────────
  videos/*.MOV
       │
       ▼  STEP 1 — [F2] DatasetCollector.extract_from_video()
          - open each .MOV with cv2.VideoCapture (CAP_FFMPEG backend)
          - sample 1 frame every FRAME_INTERVAL frames
          - mirror + resize each frame to max-width 640 px (INTER_AREA)
          - skip frames whose mean pixel diff from previous < MIN_FRAME_DIFFERENCE
          - save up to MAX_IMAGES_PER_STUDENT frames as .jpg into dataset/{ID}_{Name}/
       │
       ▼  dataset/{ID}_{Name}/*.jpg         (up to 30 images / student, deduped)
       │
       ▼  STEP 2 — [F3 + F4] FaceRecognizer.train()
          For each image in dataset/:
            a) load as colour (BGR) → run FaceDetector.detect_largest() [F3]
               • preprocess: grayscale → CLAHE → GaussianBlur
               • detectMultiScale (scaleFactor=1.2, minNeighbors=5, minSize=70)
               • crop largest face from original colour frame → grayscale
               • resize to IMAGE_SIZE (112×112, INTER_AREA)
            b) fallback: if no face detected → read image as grayscale directly
            c) vectorize [F4]:
               • equalizeHist on 112×112 crop
               • flatten → float32 → divide by 255.0
               • result: 12 544-dim unit vector
          For each student label:
            • centroid = mean of all image vectors
          Auto-calibrate threshold:
            • own_distances  = distances from each vector to its own centroid
            • wrong_distances = distances to nearest other centroid
            • threshold = max(4.0, min(BASE_THRESHOLD,
                                       p95(own)×1.15,
                                       p5(wrong)×0.85))
       │
       ▼  data/model.pkl  {labels, centroids, people, threshold}
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app import storage
from app.config import FRAME_INTERVAL, MAX_IMAGES_PER_STUDENT
from app.face import FaceEngine
from app.function2_frames import DatasetCollector


def to_rel(path: Path) -> Path:
    try:
        return path.relative_to(storage.ROOT)
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# Video → student mapping
# Keys are the expected filenames inside --video-dir.
# Values are (student_id, display_name) used to name the dataset folder.
# ---------------------------------------------------------------------------
VIDEO_STUDENT_MAP: dict[str, tuple[str, str]] = {
    "HE190019_Tran_Van_Truong.MOV":   ("HE190019", "Tran Van Truong"),
    "HE191357_Nguyen_Huy_Son.MOV":    ("HE191357", "Nguyen Huy Son"),
    "HE200457_Le_Trung_Kien.MOV":     ("HE200457", "Le Trung Kien"),
    "HE200629_Le_Trung_Hieu.MOV":     ("HE200629", "Le Trung Hieu"),
    "HE204132_Hoang_Trung_Hieu.MOV":  ("HE204132", "Hoang Trung Hieu"),
    "HE204913_Nguyen_Quang_Minh.MOV": ("HE204913", "Nguyen Quang Minh"),
}


# ---------------------------------------------------------------------------
# Step 1 — Extract frames from source videos
# ---------------------------------------------------------------------------

def step1_extract_frames(video_dir: Path, max_images: int, interval: int) -> None:
    """
    Function 2 — DatasetCollector.extract_from_video()

    For every video file listed in VIDEO_STUDENT_MAP:
      1. Open with cv2.VideoCapture (CAP_FFMPEG).
      2. Sample one frame every `interval` frames.
      3. Mirror + resize each frame (max 640 px wide).
      4. Skip frames too similar to the previous saved one.
      5. Save JPEG images to dataset/{student_id}_{student_name}/.
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Extract frames from source videos")
    print("=" * 60)

    collector = DatasetCollector(max_images=max_images)

    any_found = False
    for video_name, (student_id, student_name) in VIDEO_STUDENT_MAP.items():
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"  [SKIP] Not found: {to_rel(video_path)}")
            continue

        any_found = True
        print(f"\n  Processing : {video_name}")
        print(f"  Student    : {student_id} — {student_name}")

        total = collector.extract_from_video(
            video_path=str(video_path),
            student_id=student_id,
            student_name=student_name,
            frame_interval=interval,
            max_images=max_images,
        )
        target_dir = storage.DATASET_DIR / f"{student_id}_{student_name.replace(' ', '_')}"
        print(f"  Saved      : {total} images  →  {to_rel(target_dir)}")

    if not any_found:
        print(f"\n  [WARNING] No video files found in '{to_rel(video_dir)}'.")
        print("  Place source .MOV files there or use --skip-extract to train on existing dataset.")

    print(f"\n  Dataset directory: {to_rel(storage.DATASET_DIR)}")


# ---------------------------------------------------------------------------
# Step 2 — Train recognition model
# ---------------------------------------------------------------------------

def step2_train_model() -> None:
    """
    Functions 3 + 4 — FaceDetector (re-detection) + FaceRecognizer.train()

    1. Iterate all student folders in dataset/.
    2. For each image: re-detect the face crop with Haar Cascade (F3).
    3. Vectorize each crop: equalizeHist → flatten → float32 / 255.
    4. Compute one centroid (mean vector) per student.
    5. Auto-calibrate the rejection threshold from the training distribution.
    6. Save model to data/model.pkl.
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Train centroid recognition model")
    print("=" * 60)

    engine = FaceEngine()
    try:
        count = engine.train()
    except ValueError as exc:
        print(f"\n  [ERROR] {exc}")
        print("  Run Step 1 first or check that dataset/ contains student folders with images.")
        return

    print(f"\n  Images trained on : {count}")
    print(f"  Model saved to    : {to_rel(storage.MODEL_PATH)}")
    print("\n  Model contents:")
    print("    labels    — list of folder names (positional, aligns with centroids)")
    print("    centroids — (N_students × 12 544) float32 matrix")
    print("    people    — folder_name → (student_id, display_name)")
    print("    threshold — auto-calibrated rejection distance")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face attendance — Training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                          # full pipeline (extract + train)
  python train.py --skip-extract           # retrain on existing dataset only
  python train.py --max-images 50          # save up to 50 frames per student
  python train.py --interval 10            # denser frame sampling
  python train.py --video-dir /my/videos   # custom video directory
        """,
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip frame extraction; use existing images in dataset/ only",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("videos"),
        metavar="DIR",
        help="Directory containing source .MOV files (default: videos/)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES_PER_STUDENT,
        metavar="N",
        help=f"Max frames to save per student (default: {MAX_IMAGES_PER_STUDENT})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=FRAME_INTERVAL,
        metavar="N",
        help=f"Sample 1 frame every N video frames (default: {FRAME_INTERVAL})",
    )
    args = parser.parse_args()

    storage.ensure_dirs()

    if not args.skip_extract:
        step1_extract_frames(args.video_dir, args.max_images, args.interval)

    step2_train_model()

    print("\n" + "=" * 60)
    print("Training complete.")
    print("Run  python main.py  to start the attendance application.")
    print("=" * 60)


if __name__ == "__main__":
    main()
