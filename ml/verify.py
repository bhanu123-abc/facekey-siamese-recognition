from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ml.config_loader import load_config
from ml.preprocess import preprocess_array, preprocess_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENROLLMENT_DIR = PROJECT_ROOT / "application_data" / "enrollment"
LIVE_DIR = PROJECT_ROOT / "application_data" / "live"


@dataclass
class VerifyResult:
    verified: bool
    confidence: float
    threshold: float
    frame_scores: list[float]
    anchor_scores: list[list[float]]
    method: str
    reason: str


def enrollment_paths(max_photos: int | None = None) -> list[Path]:
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(ENROLLMENT_DIR.glob("anchor_*.jpg"))
    if max_photos is not None:
        return paths[:max_photos]
    return paths


def save_enrollment_photo(frame_bgr, slot: int) -> Path:
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    path = ENROLLMENT_DIR / f"anchor_{slot}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return path


def clear_enrollment() -> int:
    """Delete saved enrollment photos only; training data is untouched."""
    paths = enrollment_paths()
    for path in paths:
        path.unlink()
    return len(paths)


def _predict_pair(model, anchor_img: np.ndarray, live_img: np.ndarray) -> float:
    anchor_batch = np.expand_dims(anchor_img, axis=0)
    live_batch = np.expand_dims(live_img, axis=0)
    score = model.predict([anchor_batch, live_batch], verbose=0)[0][0]
    return float(score)


def verify_frames(
    model,
    live_frames_bgr: list,
    config: dict | None = None,
) -> VerifyResult:
    """Compare live webcam frames against fixed enrollment anchors."""
    config = config or load_config()
    threshold = float(config["detection_threshold"])
    use_face_crop = bool(config.get("use_face_crop", True))

    anchors = enrollment_paths()
    if not anchors:
        return VerifyResult(
            verified=False,
            confidence=0.0,
            threshold=threshold,
            frame_scores=[],
            anchor_scores=[],
            method="median_of_frames_best_anchor",
            reason="No enrollment photos found. Press e to enroll first.",
        )

    anchor_tensors = [
        preprocess_file(path, use_face_crop=use_face_crop) for path in anchors
    ]
    frame_scores: list[float] = []
    anchor_scores: list[list[float]] = []

    for frame in live_frames_bgr:
        live_tensor = preprocess_array(frame, use_face_crop=use_face_crop)
        scores_for_frame = [
            _predict_pair(model, anchor_tensor, live_tensor)
            for anchor_tensor in anchor_tensors
        ]
        anchor_scores.append(scores_for_frame)
        frame_scores.append(max(scores_for_frame))

    # Ignore a single weak camera frame rather than letting it pull down every check.
    confidence = float(np.median(frame_scores)) if frame_scores else 0.0
    verified = confidence >= threshold
    reason = (
        f"Median score {confidence:.4f} >= threshold {threshold:.4f}"
        if verified
        else f"Median score {confidence:.4f} < threshold {threshold:.4f}"
    )

    return VerifyResult(
        verified=verified,
        confidence=confidence,
        threshold=threshold,
        frame_scores=frame_scores,
        anchor_scores=anchor_scores,
        method="median_of_frames_best_anchor",
        reason=reason,
    )


def print_verify_debug(result: VerifyResult) -> None:
    print("\n--- Verification Debug ---")
    print(f"Method:     {result.method}")
    print(f"Threshold:  {result.threshold:.4f}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Verified:   {result.verified}")
    print(f"Reason:     {result.reason}")
    for index, scores in enumerate(result.anchor_scores, start=1):
        print(f"Frame {index} scores vs anchors: {[round(s, 4) for s in scores]}")
    print(f"Frame max scores: {[round(s, 4) for s in result.frame_scores]}")
    print("--------------------------\n")
