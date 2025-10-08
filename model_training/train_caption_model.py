"""
Train a simplified image-captioning pipeline (Fixed Version)
-----------------------------------------------------------
✅ Handles mismatched filenames (spaces, lowercase)
✅ Prints progress clearly
✅ Generates training curve and table images for report
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ---------- CONFIG ----------
IMAGES_DIR = "../data/images"      # relative to model_training/
CAPTIONS_FILE = "../data/captions.txt"
FEATURES_FILE = "data/image_features.npy"
TOKENIZER_FILE = "data/tokenizer.pkl"
CAPTION_MODEL_FILE = "caption_model.h5"

IMG_SHAPE = (299, 299)
EMBED_DIM = 256
MAX_WORDS = 10000
MAX_LEN = 30
# ----------------------------

def load_captions(fname):
    captions_dict = {}
    with open(fname, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try different separators: tab, comma, multiple spaces
            if "\t" in line:
                parts = line.split("\t")
            elif "," in line and ".jpg" in line.split(",")[0]:
                parts = line.split(",", 1)
            else:
                parts = line.split(" ", 1)
            if len(parts) != 2:
                continue

            img_id, caption = parts
            img_file = img_id.split("#")[0].strip().lower()
            caption = caption.strip().lower()
            captions_dict.setdefault(img_file, []).append(caption)
    print(f"📄 Loaded {sum(len(v) for v in captions_dict.values())} captions for {len(captions_dict)} images")
    return captions_dict


def extract_image_features(image_paths):
    """Extract 2048-d features per image using InceptionV3 (avg-pooled)"""
    print("🧠 Loading InceptionV3 for feature extraction...")
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    for p in tqdm(image_paths):
        try:
            img = tf.keras.preprocessing.image.load_img(p, target_size=IMG_SHAPE)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = base_model.predict(x, verbose=0)
            fname = os.path.basename(p).strip().lower()
            features[fname] = feat.flatten()
        except Exception as e:
            print(f"⚠️ Skipped {p}: {e}")
    return features

def create_tokenizer(captions_list):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(captions_list)
    return tokenizer

def build_caption_model(vocab_size):
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(EMBED_DIM, activation='relu')(inputs1)

    inputs2 = Input(shape=(MAX_LEN,))
    se1 = Embedding(vocab_size, EMBED_DIM, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)

    decoder1 = add([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_training_results(history):
    """Generate and save training plots and an epoch table image (clean style)."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Create DataFrame from history
    df = pd.DataFrame({
        "Epoch": [f"Epoch {i+1}" for i in range(len(history.history['loss']))],
        "accuracy": np.round(history.history["accuracy"], 4),
        "loss": np.round(history.history["loss"], 4),
        "val_accuracy": np.round(history.history["val_accuracy"], 4),
        "val_loss": np.round(history.history["val_loss"], 4),
    })

    # Plot training curves
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Training Results (Loss & Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_results.png")
    print("📊 Saved training_results.png")

    # Generate table as image
    fig, ax = plt.subplots(figsize=(7, 1 + len(df) * 0.4))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig("training_table.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("📋 Saved training_table.png (epoch summary)")


def main():
    print("🚀 Starting caption model training...\n")

    # Load captions
    caps = load_captions(CAPTIONS_FILE)
    all_captions = [c for caplist in caps.values() for c in caplist]
    print(f"📄 Loaded {len(all_captions)} captions for {len(caps)} images")

    # Tokenizer
    tokenizer = create_tokenizer(all_captions)
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    os.makedirs("data", exist_ok=True)
    pickle.dump(tokenizer, open(TOKENIZER_FILE, "wb"))
    print(f"🔤 Vocabulary size: {vocab_size}")

    # Image list
    image_paths = glob(os.path.join(IMAGES_DIR, "*.jpg"))
    image_paths = image_paths[:1000]
    print(f"🖼️ Found {len(image_paths)} images")

    # Feature extraction
    features = extract_image_features(image_paths)
    print(f"✅ Extracted features for {len(features)} images")

    # Create training samples
    sequences = []
    for img_file, caplist in caps.items():
        img_file_clean = os.path.basename(img_file).strip().lower()
        for cap in caplist:
            encoded = tokenizer.texts_to_sequences([cap])[0]
            if len(encoded) < 2:
                continue
            in_seq = pad_sequences([encoded[:-1]], maxlen=MAX_LEN)[0]
            out_seq = to_categorical([encoded[-1]], num_classes=vocab_size)[0]
            sequences.append((img_file_clean, in_seq, out_seq))
    print(f"🧩 Total raw caption sequences: {len(sequences)}")

    # Filter to existing features
    X1, X2, y = [], [], []
    for img_file, in_seq, out_seq in sequences:
        if img_file in features:
            X1.append(features[img_file])
            X2.append(in_seq)
            y.append(out_seq)
    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
    print(f"📦 Training samples found: {X1.shape[0]}")

    if len(X1) == 0:
        print("❌ ERROR: No matching images found between captions.txt and images folder.")
        return

    # Build and train model
    model = build_caption_model(vocab_size)
    model.summary()

    print("\n🏋️ Training model (5 epochs)...\n")
    history = model.fit(
        [X1, X2], y,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )

    # Save model
    model.save(CAPTION_MODEL_FILE)
    print(f"✅ Saved caption model: {CAPTION_MODEL_FILE}")

    # Generate charts
    plot_training_results(history)
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
