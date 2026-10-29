"""Browser UI for one-person Siamese face verification."""
from __future__ import annotations

import base64
import binascii
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from ml.config_loader import load_config
from ml.model import load_model
from ml.preprocess import face_detection_info, image_quality_score
from ml.verify import clear_enrollment, enrollment_paths, save_enrollment_photo, verify_frames

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
_model, _lock = None, threading.Lock()


def model():
    global _model
    with _lock:
        if _model is None:
            _model = load_model()
    return _model


def decode(value: str) -> np.ndarray:
    try:
        raw = base64.b64decode(value.split(",", 1)[-1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Camera frame could not be decoded.") from error
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Camera frame is not a valid image.")
    return image


def validate(frame: np.ndarray) -> tuple[bool, str]:
    face = face_detection_info(frame)
    if not face["found"]:
        return False, "No face found. Center your face in the guide."
    if face["relative_area"] < .055:
        return False, "Move closer so your face fills more of the guide."
    quality = image_quality_score(frame, face["box"])
    if quality["sharpness"] < 35:
        return False, "The image is blurry. Hold still and try again."
    if quality["brightness"] < 35:
        return False, "The image is too dark. Add light and retry."
    return True, ""


@app.get("/")
def index():
    return render_template("index.html", config=load_config())


@app.get("/api/status")
def status():
    config = load_config()
    return jsonify(enrolled=len(enrollment_paths()), required=int(config["num_enrollment_photos"]), threshold=float(config["detection_threshold"]))


@app.post("/api/enroll")
def enroll():
    values = (request.get_json(silent=True) or {}).get("frames", [])
    if not isinstance(values, list) or not values:
        return jsonify(error="Send one or more camera frames."), 400
    required, start = int(load_config()["num_enrollment_photos"]), len(enrollment_paths()) + 1
    accepted, errors = 0, []
    for value in values:
        if start + accepted > required:
            break
        try:
            frame = decode(value)
            valid, reason = validate(frame)
            if not valid:
                errors.append(reason); continue
            save_enrollment_photo(frame, start + accepted); accepted += 1
        except ValueError as error:
            errors.append(str(error))
    enrolled = len(enrollment_paths())
    return jsonify(accepted=accepted, enrolled=enrolled, required=required, complete=enrolled >= required,
                   message="Your reference set is ready." if enrolled >= required else f"Saved {accepted} sample(s). Capture {required-enrolled} more.", rejections=errors[:3])


@app.post("/api/reset-enrollment")
def reset():
    return jsonify(removed=clear_enrollment(), message="Reference set cleared.")


@app.post("/api/verify")
def verify():
    values = (request.get_json(silent=True) or {}).get("frames", [])
    if len(enrollment_paths()) < int(load_config()["num_enrollment_photos"]):
        return jsonify(error="Finish enrollment before verifying."), 400
    usable = []
    for value in values if isinstance(values, list) else []:
        try:
            frame = decode(value)
            if validate(frame)[0]: usable.append(frame)
        except ValueError:
            pass
    if not usable:
        return jsonify(error="No clear face was found in the captured frames."), 422
    result = verify_frames(model(), usable, load_config())
    return jsonify(verified=result.verified, score=round(result.confidence * 100, 1), threshold=round(result.threshold * 100, 1),
                   frames_used=len(usable), reason=result.reason, method=result.method)


if __name__ == "__main__":
    # 5000 is frequently occupied by macOS/dev tools; use a less contested port.
    app.run(host="127.0.0.1", port=5055, debug=True)
