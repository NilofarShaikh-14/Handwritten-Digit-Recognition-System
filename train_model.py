"""
MNIST Handwritten Digit Recognition — Model Training
Loads the raw MNIST binary files already present in the project root,
preprocesses the data, trains a CNN, and saves the model to disk.
"""

import os
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMAGES    = os.path.join(BASE_DIR, "train-images.idx3-ubyte")
TRAIN_LABELS    = os.path.join(BASE_DIR, "train-labels.idx1-ubyte")
TEST_IMAGES     = os.path.join(BASE_DIR, "t10k-images.idx3-ubyte")
TEST_LABELS     = os.path.join(BASE_DIR, "t10k-labels.idx1-ubyte")
MODEL_PATH      = os.path.join(BASE_DIR, "mnist_model.h5")
PLOTS_DIR       = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── MNIST binary reader ────────────────────────────────────────────────────────
def read_images(path: str) -> np.ndarray:
    """Parse the IDX3-ubyte image file format."""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_labels(path: str) -> np.ndarray:
    """Parse the IDX1-ubyte label file format."""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# ── Data loading & preprocessing ──────────────────────────────────────────────
def load_and_preprocess():
    print("Loading MNIST data from binary files …")
    x_train = read_images(TRAIN_IMAGES)
    y_train = read_labels(TRAIN_LABELS)
    x_test  = read_images(TEST_IMAGES)
    y_test  = read_labels(TEST_LABELS)

    print(f"  Train: {x_train.shape}  labels: {y_train.shape}")
    print(f"  Test : {x_test.shape}   labels: {y_test.shape}")

    # Normalise to [0, 1] and add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train[..., np.newaxis]   # (60000, 28, 28, 1)
    x_test  = x_test[...,  np.newaxis]   # (10000, 28, 28, 1)

    return x_train, y_train, x_test, y_test


# ── Model definition ───────────────────────────────────────────────────────────
def build_model() -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(model: keras.Model, x_train, y_train, x_test, y_test):
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    # Light data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    datagen.fit(x_train)

    print("\nTraining …")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=len(x_train) // 128,
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── Evaluation plots ───────────────────────────────────────────────────────────
def save_plots(history, model, x_test, y_test):
    # --- Accuracy / Loss curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=120)
    plt.close()
    print("Saved training_curves.png")

    # --- Confusion matrix ---
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10)
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()
    print("Saved confusion_matrix.png")

    # --- Classification report ---
    report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
    print("\nClassification Report:\n")
    print(report)

    report_path = os.path.join(BASE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    x_train, y_train, x_test, y_test = load_and_preprocess()

    model = build_model()
    model.summary()

    history = train(model, x_train, y_train, x_test, y_test)

    # Reload best checkpoint for evaluation
    best_model = keras.models.load_model(MODEL_PATH)
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy : {test_acc * 100:.2f}%")
    print(f"Test loss     : {test_loss:.4f}")

    save_plots(history, best_model, x_test, y_test)
    print(f"\nModel saved to: {MODEL_PATH}")
