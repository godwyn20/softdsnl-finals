"""
Feature extraction using InceptionV3 — supports Django UploadedFile and file paths
"""

import numpy as np
import io
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

# Load model once globally
_base_model = InceptionV3(weights="imagenet")
_model = Model(_base_model.input, _base_model.layers[-2].output)


def preprocess_image_file(img_source):
    """
    Accepts either:
    - Django InMemoryUploadedFile
    - File path (string)
    Returns (1, 2048) feature vector.
    """
    # If it's an uploaded file (from Postman)
    if hasattr(img_source, "read"):
        img = Image.open(io.BytesIO(img_source.read())).convert("RGB")
    else:
        # Assume it's a file path
        img = Image.open(img_source).convert("RGB")

    img = img.resize((299, 299))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = _model.predict(x, verbose=0)
    return features
