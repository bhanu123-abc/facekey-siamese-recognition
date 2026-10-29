"""Calibrate detection threshold from positive and negative image pairs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from ml.config_loader import load_config, save_config
from ml.model import load_model
from ml.preprocess import preprocess_file
from ml.verify import _predict_pair

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENROLLMENT_DIR = PROJECT_ROOT / "application_data" / "enrollment"
NEGATIVE_DIR = PROJECT_ROOT / "data" / "negative"


def _collect_scores(model, anchor_tensors: list[np.ndarray], image_paths: list[Path], use_face_crop: bool) -> list[float]:
    scores: list[float] = []
    for path in image_paths:
        live_tensor = preprocess_file(path, use_face_crop=use_face_crop)
        frame_scores = [_predict_pair(model, anchor, live_tensor) for anchor in anchor_tensors]
        scores.append(max(frame_scores))
    return scores


def calibrate(
    positive_limit: int = 20,
    negative_limit: int = 20,
    use_face_crop: bool = True,
) -> dict:
    model = load_model()
    anchors = sorted(ENROLLMENT_DIR.glob("anchor_*.jpg"))
    if not anchors:
        raise FileNotFoundError(
            f"No enrollment photos in {ENROLLMENT_DIR}. Run run_verify.py and press e first."
        )

    anchor_tensors = [preprocess_file(path, use_face_crop=use_face_crop) for path in anchors]

    # Do not compare an anchor with itself: self-comparisons make calibration
    # unrealistically optimistic and were the main source of misleading scores.
    positive_candidates = list(enumerate(anchors))
    random.shuffle(positive_candidates)

    negative_candidates = list(NEGATIVE_DIR.glob("*.jpg"))
    random.shuffle(negative_candidates)
    negative_paths = negative_candidates[:negative_limit]

    positive_scores: list[float] = []
    for index, path in positive_candidates[:positive_limit]:
        other_anchors = [tensor for other_index, tensor in enumerate(anchor_tensors) if other_index != index]
        if not other_anchors:
            continue
        live_tensor = preprocess_file(path, use_face_crop=use_face_crop)
        positive_scores.append(max(_predict_pair(model, anchor, live_tensor) for anchor in other_anchors))
    negative_scores = _collect_scores(model, anchor_tensors, negative_paths, use_face_crop)

    if not positive_scores or not negative_scores:
        raise RuntimeError("Need both positive and negative samples to calibrate.")

    positive_mean = float(np.mean(positive_scores))
    negative_mean = float(np.mean(negative_scores))
    candidates = sorted(set(positive_scores + negative_scores))
    # Select the score split with the best balanced accuracy rather than using
    # an arbitrary average of means.
    suggested_threshold = max(
        candidates,
        key=lambda threshold: (
            np.mean(np.asarray(positive_scores) >= threshold)
            + np.mean(np.asarray(negative_scores) < threshold)
        ) / 2,
    )

    print("\n--- Calibration Results ---")
    print(f"Positive samples: {len(positive_scores)}")
    print(f"Positive mean:    {positive_mean:.4f}")
    print(f"Positive min/max: {min(positive_scores):.4f} / {max(positive_scores):.4f}")
    print(f"Negative samples: {len(negative_scores)}")
    print(f"Negative mean:    {negative_mean:.4f}")
    print(f"Negative min/max: {min(negative_scores):.4f} / {max(negative_scores):.4f}")
    print(f"Suggested threshold: {suggested_threshold:.4f} (best balanced split)")
    print("---------------------------\n")

    config = load_config()
    config["detection_threshold"] = round(suggested_threshold, 4)
    save_config(config)
    print(f"Updated config/thresholds.json -> detection_threshold = {config['detection_threshold']}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate face verification threshold.")
    parser.add_argument("--positive-limit", type=int, default=20)
    parser.add_argument("--negative-limit", type=int, default=20)
    parser.add_argument("--no-face-crop", action="store_true")
    args = parser.parse_args()

    calibrate(
        positive_limit=args.positive_limit,
        negative_limit=args.negative_limit,
        use_face_crop=not args.no_face_crop,
    )


if __name__ == "__main__":
    main()
