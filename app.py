"""
Flask Backend — MNIST Digit Recognition API
Endpoints
  POST /predict        – accepts a PNG/JPEG upload or a base64-encoded canvas PNG
  GET  /health         – liveness probe
  GET  /               – serves the frontend SPA
"""

import os
import io
import base64
import logging

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, "static")
MODEL_PATH  = os.path.join(BASE_DIR, "mnist_model.h5")
UPLOAD_DIR  = os.path.join(STATIC_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app   = Flask(__name__, static_folder=STATIC_DIR, template_folder=STATIC_DIR)
CORS(app)

# ── Model loading (lazy, once) ─────────────────────────────────────────────────
_model = None

def get_model() -> keras.Model:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run train_model.py first."
            )
        logger.info("Loading model from %s …", MODEL_PATH)
        _model = keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded.")
    return _model


# ── Image preprocessing ────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convert input image to MNIST format:
    - Grayscale
    - White digit on black background
    - 28x28 pixels
    - Normalized to [0, 1]
    """

    # Convert to RGB or grayscale
    if img.mode == "RGBA":
        # Flatten transparent areas onto black
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        background.paste(img, mask=img.getchannel("A"))
        img = background.convert("L")
    else:
        img = img.convert("L")

    # Determine background brightness using image borders
    arr = np.array(img)

    border = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1]
    ])

    background_mean = border.mean()

    # Invert only if the background is light
    if background_mean > 127:
        img = ImageOps.invert(img)

    # Crop around the digit
    bbox = img.getbbox()

    if bbox:
        img = img.crop(bbox)

    # Preserve aspect ratio and add padding
    width, height = img.size

    max_dim = max(width, height)

    canvas_size = max_dim + 20

    canvas = Image.new(
        "L",
        (canvas_size, canvas_size),
        0
    )

    x = (canvas_size - width) // 2
    y = (canvas_size - height) // 2

    canvas.paste(img, (x, y))

    # Resize to MNIST dimensions
    img = canvas.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # Normalize
    arr = np.array(img, dtype="float32") / 255.0

    # Add batch and channel dimensions
    return arr.reshape(1, 28, 28, 1)


def decode_base64_image(data_url: str) -> Image.Image:
    """Decode a data:image/png;base64,… string into a PIL Image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw))


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts either:
      • multipart/form-data  with field "file"  (image upload)
      • application/json     with field "image" (base64 data-URL from canvas)
    Returns JSON: { digit, confidence, probabilities }
    """
    try:
        model = get_model()

        # ── Determine image source ──
        if request.content_type and "multipart/form-data" in request.content_type:
            if "file" not in request.files:
                return jsonify({"error": "No file part in request"}), 400
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            img = Image.open(file.stream)

        elif request.is_json:
            data = request.get_json()
            if not data or "image" not in data:
                return jsonify({"error": "Missing 'image' field in JSON body"}), 400
            img = decode_base64_image(data["image"])

        else:
            return jsonify({"error": "Unsupported content type"}), 415

        # ── Preprocess & infer ──
        tensor = preprocess_image(img)
        probs  = model.predict(tensor, verbose=0)[0]          # shape (10,)
        digit  = int(np.argmax(probs))
        conf   = float(probs[digit])

        logger.info("Prediction: digit=%d  confidence=%.3f", digit, conf)

        return jsonify({
            "digit":         digit,
            "confidence":    round(conf * 100, 2),
            "probabilities": [round(float(p) * 100, 2) for p in probs],
        })

    except FileNotFoundError as exc:
        logger.error(str(exc))
        return jsonify({"error": str(exc)}), 503

    except Exception as exc:        # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify({"error": f"Internal error: {exc}"}), 500


# ── Static assets ──────────────────────────────────────────────────────────────
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-load model at startup so first request is fast
    try:
        get_model()
    except FileNotFoundError as e:
        logger.warning(str(e))

    app.run(host="0.0.0.0", port=5000, debug=False)
