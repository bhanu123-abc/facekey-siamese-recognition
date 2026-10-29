from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model as keras_load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "siamesemodel.h5"


class L1Dist(Layer):
    """L1 distance layer used by the Siamese network."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def load_model(model_path: Path | str | None = None):
    """Load the trained Siamese model with required custom objects."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return keras_load_model(
        str(path),
        custom_objects={
            "L1Dist": L1Dist,
            "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
        },
    )
