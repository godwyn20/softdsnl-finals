"""
Train a simple sentiment classifier (binary: positive/negative) using Flickr captions.
------------------------------------------------------------
✅ Uses Hugging Face 'datasets' cleanly (optional, not required)
✅ Compatible with Flickr8k captions.txt format
✅ Generates training curve and table images
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset  # from 'datasets' package
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ---------- CONFIG ----------
CAPTIONS_FILE = "../data/captions.txt"
TOKENIZER_FILE = "data/sentiment_tokenizer.pkl"
SENTIMENT_MODEL_FILE = "sentiment_model.h5"
MAX_WORDS = 5000
MAX_LEN = 50
# ----------------------------

def build_dataset_from_captions(captions_filepath, limit=5000):
    """Flexible loader for Flickr8k captions (handles tab, comma, or spaces)."""
    texts, labels = [], []
    print("🧠 Building sentiment dataset from captions:", captions_filepath)

    positive_keywords = {"happy", "smile", "smiling", "beautiful", "love", "lovely", "cute", "fun"}
    negative_keywords = {"sad", "cry", "crying", "angry", "hate", "bad", "ugly", "broken"}

    with open(captions_filepath, "r", encoding="utf8") as f:
        for line in f:
            if len(texts) >= limit:
                break
            line = line.strip()
            if not line:
                continue

            # detect separator
            if "\t" in line:
                parts = line.split("\t")
            elif "," in line and ".jpg" in line.split(",")[0]:
                parts = line.split(",", 1)
            else:
                parts = line.split(" ", 1)

            if len(parts) != 2:
                continue

            caption = parts[1].strip().lower()
            texts.append(caption)

            # rule-based labeling
            if any(w in caption for w in negative_keywords):
                labels.append(0)
            elif any(w in caption for w in positive_keywords):
                labels.append(1)
            else:
                labels.append(1)  # neutral default

    print(f"✅ Loaded {len(texts)} captions for sentiment training.")
    return texts, np.array(labels)


def plot_training_results(history):
    """Plot and save training loss/accuracy curves and epoch table."""
    import pandas as pd

    # Plot training curves
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Sentiment Model Training Results")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sentiment_training_results.png")
    print("📊 Saved sentiment_training_results.png")

    # Create and save epoch summary table (same style you requested)
    df = pd.DataFrame({
        "Epoch": [f"Epoch {i+1}" for i in range(len(history.history["loss"]))],
        "accuracy": np.round(history.history["accuracy"], 4),
        "loss": np.round(history.history["loss"], 4),
        "val_accuracy": np.round(history.history["val_accuracy"], 4),
        "val_loss": np.round(history.history["val_loss"], 4)
    })

    fig, ax = plt.subplots(figsize=(7, 1 + len(df) * 0.4))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig("sentiment_training_table.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("📋 Saved sentiment_training_table.png (epoch summary)")

def main():
    # Step 1: Build dataset
    texts, labels = build_dataset_from_captions(CAPTIONS_FILE, limit=3000)

    # Step 2: Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

    # Step 3: Convert to HuggingFace Dataset (for consistency)
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    print(dataset)

    # Step 4: Build model
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # Step 5: Train model
    print("\n🏋️ Training sentiment model (5 epochs)...\n")
    history = model.fit(padded, labels, epochs=5, batch_size=32, validation_split=0.1)

    # Step 6: Save model + tokenizer
    os.makedirs("data", exist_ok=True)
    pickle.dump(tokenizer, open(TOKENIZER_FILE, "wb"))
    model.save(SENTIMENT_MODEL_FILE)
    print(f"✅ Saved sentiment model: {SENTIMENT_MODEL_FILE}")

    # Step 7: Plot training results
    plot_training_results(history)
    print("\n🎉 Sentiment training completed successfully!")

if __name__ == "__main__":
    main()
