"""
Train an image captioning model using Flickr8k-style data.
------------------------------------------------------------
 Dual-input model (image features + text sequence)
 Tracks accuracy, val_accuracy, loss, val_loss
 Configurable IMAGE_LIMIT and EPOCHS
 Generates training summary table image
 Saves tokenizer + model to result/caption/
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ---------- CONFIG ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "result", "caption")

CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

TOKENIZER_FILE = os.path.join(RESULT_DIR, "tokenizer.pkl")
FEATURES_FILE = os.path.join(RESULT_DIR, "image_features.npy")
CAPTION_MODEL_FILE = os.path.join(RESULT_DIR, "caption_model.h5")

IMAGE_LIMIT = 500  # lower for light training
EPOCHS = 5
MAX_LEN = 30
EMBED_DIM = 128
LSTM_UNITS = 256
# ----------------------------

os.makedirs(RESULT_DIR, exist_ok=True)


# ---------- STEP 1: LOAD CAPTIONS ----------
def load_captions(filepath):
    captions = {}
    with open(filepath, "r", encoding="utf8") as f:
        header = f.readline()  # skip header if exists
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) != 2:
                continue
            image_id, caption = parts
            caption = "startseq " + caption.lower().strip() + " endseq"
            captions.setdefault(image_id.strip(), []).append(caption)
    total_captions = sum(len(v) for v in captions.values())
    print(f" Loaded {total_captions} captions for {len(captions)} images")
    return captions


# ---------- STEP 2: EXTRACT IMAGE FEATURES ----------
def extract_image_features(image_dir, limit=None):
    print(" Extracting image features with InceptionV3...")
    base_model = InceptionV3(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    image_files = [
        os.path.join(image_dir, img)
        for img in os.listdir(image_dir)
        if img.lower().endswith(".jpg")
    ]
    if limit:
        image_files = image_files[:limit]
        print(f" Limiting to {len(image_files)} images (IMAGE_LIMIT={limit})")

    features = {}
    for img_path in tqdm(image_files, desc="Extracting Features"):
        try:
            img = load_img(img_path, target_size=(299, 299))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feature = model.predict(img, verbose=0)
            features[os.path.basename(img_path)] = feature.flatten()
        except Exception as e:
            print(f" Skipping {img_path}: {e}")

    np.save(FEATURES_FILE, features)
    print(f" Extracted features for {len(features)} images")
    return features


# ---------- STEP 3: TOKENIZER ----------
def build_tokenizer(captions_dict):
    all_captions = [c for caps in captions_dict.values() for c in caps]
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(all_captions)
    pickle.dump(tokenizer, open(TOKENIZER_FILE, "wb"))
    print(f" Vocabulary size: {len(tokenizer.word_index) + 1}")
    return tokenizer


# ---------- STEP 4: CREATE DATASET ----------
def create_training_data(captions_dict, features, tokenizer, max_len=MAX_LEN):
    X1, X2, y = list(), list(), list()
    vocab_size = len(tokenizer.word_index) + 1
    print(" Preparing training sequences...")

    for img, caps in tqdm(captions_dict.items(), desc="Building Sequences"):
        feature = features.get(img)
        if feature is None:
            continue
        for caption in caps:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_word = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                out_word = to_categorical([out_word], num_classes=vocab_size)[0]
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_word)

    print(f" Created {len(X1)} training samples.")
    return np.array(X1), np.array(X2), np.array(y), vocab_size


# ---------- STEP 5: BUILD DUAL-INPUT MODEL ----------
def build_caption_model(vocab_size):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    inputs2 = Input(shape=(MAX_LEN,))
    se1 = Embedding(vocab_size, EMBED_DIM, mask_zero=True)(inputs2)
    se2 = LSTM(LSTM_UNITS)(se1)

    decoder1 = add([fe2, se2])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ---------- STEP 6: SAVE TRAINING TABLE ----------
def save_training_table(history, save_path):
    """Generate a table image showing accuracy, loss, val_accuracy, val_loss."""
    df = pd.DataFrame({
        "Epoch": [f"Epoch {i+1}" for i in range(len(history.history["loss"]))],
        "accuracy": np.round(history.history.get("accuracy", []), 4),
        "val_accuracy": np.round(history.history.get("val_accuracy", []), 4),
        "loss": np.round(history.history.get("loss", []), 4),
        "val_loss": np.round(history.history.get("val_loss", []), 4),
    })

    fig, ax = plt.subplots(figsize=(8, 1 + len(df) * 0.4))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f" Saved training summary table: {save_path}")


# ---------- MAIN ----------
def main():
    print(" Starting caption training pipeline...")
    captions_dict = load_captions(CAPTIONS_FILE)
    features = extract_image_features(IMAGES_DIR, limit=IMAGE_LIMIT)
    tokenizer = build_tokenizer(captions_dict)
    X1, X2, y, vocab_size = create_training_data(captions_dict, features, tokenizer)

    model = build_caption_model(vocab_size)
    print(f" Training caption model ({EPOCHS} epochs)...")
    history = model.fit(
        [X1, X2],
        y,
        epochs=EPOCHS,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    model.save(CAPTION_MODEL_FILE)
    print(f" Saved caption model to {CAPTION_MODEL_FILE}")

    table_path = os.path.join(RESULT_DIR, "caption_training_table.png")
    save_training_table(history, table_path)

    print(f" All outputs saved in {RESULT_DIR}")


if __name__ == "__main__":
    main()
