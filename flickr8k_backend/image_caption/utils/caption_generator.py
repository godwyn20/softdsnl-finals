"""
Caption generation helper:
- Load caption model and tokenizer
- Generate a caption from extracted image features (simplified greedy decoding)
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

TOKENIZER_FILE = "data/tokenizer.pkl"
CAPTION_MODEL_FILE = "caption_model.h5"
MAX_LEN = 30

# Global caches
_tokenizer = None
_model = None

def load_caption_tools():
    """Lazy-load tokenizer and trained model."""
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = pickle.load(open(TOKENIZER_FILE, "rb"))
    if _model is None:
        _model = load_model(CAPTION_MODEL_FILE)
    return _tokenizer, _model


def generate_caption(image_feature_vector):
    """
    Generate a caption using greedy search decoding.
    image_feature_vector: numpy array of shape (1, 2048)
    """
    tokenizer, model = load_caption_tools()
    inv_map = {v: k for k, v in tokenizer.word_index.items()}

    in_text = []
    for _ in range(MAX_LEN):
        seq = tokenizer.texts_to_sequences([" ".join(in_text)])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)
        yhat = model.predict([image_feature_vector, seq], verbose=0)
        y_index = np.argmax(yhat)
        word = inv_map.get(y_index)
        if word is None:
            break
        in_text.append(word)
        if word == "endseq":
            break

    caption = " ".join(in_text)
    return caption.replace("endseq", "").strip()
