"""
Caption generation helper:
- Load caption model and tokenizer from ../result/caption/
- Generate a caption from extracted image features (simplified greedy decoding)
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer


# ---------- CONFIG ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL_DIR = os.path.join(BASE_DIR, "result", "caption")

TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.pkl")
CAPTION_MODEL_FILE = os.path.join(MODEL_DIR, "caption_model.h5")
MAX_LEN = 30
# ----------------------------

tokenizer = None
caption_model = None


# Proper custom NotEqual layer
class NotEqual(Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)

    def call(self, *inputs, **kwargs):
        import tensorflow as tf
        if len(inputs) == 2:
            x, y = inputs
            return tf.not_equal(x, y)
        elif len(inputs) == 1:
            x = inputs[0]
            return tf.not_equal(x, 0)
        else:
            raise ValueError(f"Unexpected inputs for NotEqual: {inputs}")

    def get_config(self):
        base_config = super(NotEqual, self).get_config()
        return base_config


def load_caption_tools():
    """Load tokenizer and trained caption model safely (from result/caption)."""
    global tokenizer, caption_model

    if tokenizer is None:
        print(f" Loading tokenizer from {TOKENIZER_FILE}")
        with open(TOKENIZER_FILE, "rb") as f:
            tokenizer = pickle.load(f)

    if caption_model is None:
        print(f" Loading caption model from {CAPTION_MODEL_FILE}")
        from tensorflow.keras.utils import custom_object_scope
        from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
        with custom_object_scope({'NotEqual': NotEqual}):
            try:
                caption_model = load_model(CAPTION_MODEL_FILE)
            except Exception as e:
                print(" Fallback: loading model in legacy mode due to:", e)
                import h5py
                with h5py.File(CAPTION_MODEL_FILE, "r") as f:
                    caption_model = load_model_from_hdf5(f, custom_objects={'NotEqual': NotEqual})
    return tokenizer, caption_model


def generate_caption(image_feature_vector):
    """
    Generate a caption from an image feature vector.
    Uses greedy decoding for simplicity.
    """
    tokenizer, model = load_caption_tools()
    inv_map = {v: k for k, v in tokenizer.word_index.items()}

    in_text = []
    for i in range(MAX_LEN):
        seq = tokenizer.texts_to_sequences([" ".join(in_text)])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)
        yhat = model.predict([image_feature_vector, seq], verbose=0)
        y_index = np.argmax(yhat)
        word = inv_map.get(y_index)
        if not word:
            break
        in_text.append(word)
        if word == "endseq":
            break

    caption = " ".join(in_text)
    return caption.replace("endseq", "").strip()
