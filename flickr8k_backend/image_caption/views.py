"""
Django view for image caption generation.
"""

import os
import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils.preprocess import preprocess_image_file
from .utils.caption_generator import generate_caption
import tensorflow as tf

# ---------- PATHS ----------
TOKENIZER_FILE = os.path.join("..", "result", "caption", "tokenizer.pkl")
CAPTION_MODEL_FILE = os.path.join("..", "result", "caption", "caption_model.h5")
# ----------------------------

# lazy load (to speed up repeated requests)
_tokenizer = None
_model = None


def load_caption_tools():
    """Loads tokenizer and caption model (once)."""
    global _tokenizer, _model
    if _tokenizer is None:
        print(f" Loading tokenizer from {TOKENIZER_FILE}")
        _tokenizer = pickle.load(open(TOKENIZER_FILE, "rb"))
    if _model is None:
        print(f" Loading caption model from {CAPTION_MODEL_FILE}")
        _model = tf.keras.models.load_model(CAPTION_MODEL_FILE)
    return _tokenizer, _model


@api_view(["POST"])
def predict_caption(request):
    try:
        if "file" not in request.FILES:
            return Response({"error": "No file uploaded (use key 'file')"}, status=400)

        uploaded_file = request.FILES["file"]
        print(" Received file:", uploaded_file.name)

        #  Load tools (same as sentiment)
        tokenizer, model = load_caption_tools()

        #  Preprocess image
        features = preprocess_image_file(uploaded_file)
        print(" Feature shape:", getattr(features, "shape", None))

        # Generate caption
        caption = generate_caption(features)
        print(" Generated caption:", caption)

        return Response({"caption": caption})

    except Exception as e:
        import traceback
        print(" Error in predict_caption:", e)
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
