from PIL import Image
import numpy as np

def preprocess_image_file(fp, target_size=(299,299)):
    """
    Preprocess an uploaded image file (for inference).
    Converts to RGB, resizes, normalizes to range [-1, 1], and expands dimensions.
    """
    img = Image.open(fp).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32")
    arr = (arr / 127.5) - 1.0  # scale to [-1, 1]
    arr = np.expand_dims(arr, axis=0)
    return arr
