from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_detection_info(frame_bgr: np.ndarray) -> dict:
    """Find the largest face and report how much of the frame it occupies."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
    if len(faces) == 0:
        return {"found": False, "box": None, "relative_area": 0.0}
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return {"found": True, "box": (int(x), int(y), int(w), int(h)), "relative_area": float(w * h) / (frame_bgr.shape[0] * frame_bgr.shape[1])}


def crop_face(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop the detected face with context, or preserve the image if none is found."""
    info = face_detection_info(frame_bgr)
    if not info["found"]:
        return frame_bgr
    x, y, w, h = info["box"]
    margin = int(max(w, h) * 0.24)
    return frame_bgr[max(0, y-margin):min(frame_bgr.shape[0], y+h+margin), max(0, x-margin):min(frame_bgr.shape[1], x+w+margin)]


def image_quality_score(frame_bgr: np.ndarray, box: tuple[int, int, int, int] | None = None) -> dict:
    image = frame_bgr if box is None else frame_bgr[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {"brightness": float(np.mean(gray)), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}


def _to_tensor(img_rgb: np.ndarray) -> np.ndarray:
    tensor = tf.image.resize(img_rgb, (100, 100))
    return (tensor / 255.0).numpy()


def preprocess_file(file_path: str | Path, use_face_crop: bool = True) -> np.ndarray:
    frame_bgr = cv2.imread(str(Path(file_path)))
    if frame_bgr is None:
        raise ValueError(f"Could not read image: {file_path}")
    return preprocess_array(frame_bgr, use_face_crop)


def preprocess_array(frame_bgr: np.ndarray, use_face_crop: bool = True) -> np.ndarray:
    cropped = crop_face(frame_bgr) if use_face_crop else frame_bgr
    return _to_tensor(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
