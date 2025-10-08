from rest_framework.decorators import api_view
from rest_framework.response import Response

#  Import from image_caption utils (correct paths)
from image_caption.utils.preprocess import preprocess_image_file
from image_caption.utils.caption_generator import generate_caption

import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os


# ---------- CONFIG ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SENTIMENT_DIR = os.path.join(BASE_DIR, "result", "sentiment")

SENTIMENT_TOKENIZER_FILE = os.path.join(SENTIMENT_DIR, "sentiment_tokenizer.pkl")
SENTIMENT_MODEL_FILE = os.path.join(SENTIMENT_DIR, "sentiment_model.h5")

_sentiment_tokenizer = None
_sentiment_model = None
MAX_LEN = 50
# ----------------------------


def load_sentiment_tools():
    """Lazy-load sentiment tokenizer and model."""
    global _sentiment_tokenizer, _sentiment_model

    if _sentiment_tokenizer is None:
        print(f" Loading sentiment tokenizer from {SENTIMENT_TOKENIZER_FILE}")
        _sentiment_tokenizer = pickle.load(open(SENTIMENT_TOKENIZER_FILE, "rb"))

    if _sentiment_model is None:
        print(f" Loading sentiment model from {SENTIMENT_MODEL_FILE}")
        _sentiment_model = tf.keras.models.load_model(SENTIMENT_MODEL_FILE)

    return _sentiment_tokenizer, _sentiment_model


@api_view(["POST"])
def predict_image_sentiment(request):
    """
    POST /api/predict_image_sentiment/
    Upload an image → returns generated caption + sentiment prediction
    """
    try:
        #  Validate upload
        if "file" not in request.FILES:
            return Response({"error": "No file uploaded (use key 'file')"}, status=400)

        uploaded_file = request.FILES["file"]
        print(" Received file:", uploaded_file.name)

        #  Step 1: Preprocess image + generate caption
        features = preprocess_image_file(uploaded_file)
        caption = generate_caption(features)
        print(" Generated caption:", caption)

        #  Step 2: Sentiment prediction for caption
        tokenizer, sentiment_model = load_sentiment_tools()
        seq = tokenizer.texts_to_sequences([caption])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
        pred = sentiment_model.predict(padded)[0][0]
        sentiment = "positive" if pred >= 0.5 else "negative"

        print(f" Sentiment: {sentiment} ({pred:.4f})")

        #  Step 3: Return combined result
        return Response({
            "caption": caption,
            "sentiment": sentiment,
            "confidence": float(pred)
        })

    except FileNotFoundError as e:
        print(" Missing required sentiment model/tokenizer file:", e)
        return Response({
            "error": f"Missing required file: {str(e)}",
            "expected_files": {
                "sentiment_tokenizer": os.path.exists(SENTIMENT_TOKENIZER_FILE),
                "sentiment_model": os.path.exists(SENTIMENT_MODEL_FILE)
            }
        }, status=500)

    except Exception as e:
        import traceback
        print(" Error in predict_image_sentiment:", e)
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
