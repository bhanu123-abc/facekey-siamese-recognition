#!/usr/bin/env python3
"""Real-time face enrollment and verification with improved threshold logic."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config_loader import load_config
from ml.model import load_model
from ml.verify import (
    enrollment_paths,
    print_verify_debug,
    save_enrollment_photo,
    verify_frames,
)

LEGACY_ANCHOR = PROJECT_ROOT / "application_data" / "input_image" / "input_image.jpg"
ENROLLMENT_DIR = PROJECT_ROOT / "application_data" / "enrollment"


def crop_webcam_frame(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    crop_size = 600
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(w, cx + crop_size // 2)
    y2 = min(h, cy + crop_size // 2)
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (250, 250))


def migrate_legacy_anchor() -> None:
    """Copy the old notebook anchor into enrollment slot 1 if needed."""
    if not LEGACY_ANCHOR.exists():
        return
    target = ENROLLMENT_DIR / "anchor_1.jpg"
    if target.exists():
        return
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    frame = cv2.imread(str(LEGACY_ANCHOR))
    if frame is not None:
        cv2.imwrite(str(target), frame)
        print(f"Migrated legacy anchor -> {target}")


def capture_live_frames(cap, count: int, delay_ms: int) -> list:
    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(crop_webcam_frame(frame))
        time.sleep(delay_ms / 1000.0)
    return frames


def draw_status(frame, lines: list[str]) -> None:
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28


def main() -> None:
    migrate_legacy_anchor()
    config = load_config()
    num_enrollment = int(config["num_enrollment_photos"])
    num_live_frames = int(config["num_live_frames"])
    frame_delay_ms = int(config["frame_delay_ms"])

    print("Loading model...")
    model = load_model()
    print("Model loaded.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    enrollment_slot = 1
    last_result = "Press e to enroll, v to verify, q to quit"

    print("\nControls:")
    print("  e = save enrollment photo (cycles anchor_1..anchor_3)")
    print("  v = verify against enrolled anchors (does NOT overwrite enrollment)")
    print("  q = quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display = crop_webcam_frame(frame.copy())
        enrolled = len(enrollment_paths())
        draw_status(
            display,
            [
                f"Enrolled: {enrolled}/{num_enrollment}",
                f"Next enroll slot: anchor_{enrollment_slot}",
                f"Threshold: {config['detection_threshold']}",
                last_result,
            ],
        )
        cv2.imshow("Face Verification", display)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("e"):
            path = save_enrollment_photo(display, enrollment_slot)
            print(f"Enrollment saved: {path}")
            enrollment_slot = (enrollment_slot % num_enrollment) + 1
            last_result = f"Saved {path.name}"

        elif key == ord("v"):
            if not enrollment_paths():
                last_result = "Enroll first (press e)"
                print(last_result)
                continue

            print("Capturing live frames for verification...")
            live_frames = capture_live_frames(cap, num_live_frames, frame_delay_ms)
            result = verify_frames(model, live_frames, config)
            print_verify_debug(result)
            last_result = (
                f"VERIFIED ({result.confidence:.3f})"
                if result.verified
                else f"NOT VERIFIED ({result.confidence:.3f})"
            )

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
