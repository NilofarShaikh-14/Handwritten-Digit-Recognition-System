# MNIST Handwritten Digit Recognizer

A complete end-to-end handwritten digit recognition web application built with:

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras CNN |
| Dataset | MNIST (raw binary files) | https://www.kaggle.com/datasets/hojjatk/mnist-dataset
  
| Backend | Flask REST API |
| Frontend | Vanilla HTML · CSS · JavaScript |
| Image processing | Pillow (PIL) |

---

## Project Structure

```
MNIST_Project/
├── train-images.idx3-ubyte      # MNIST raw binary (training images)
├── train-labels.idx1-ubyte      # MNIST raw binary (training labels)
├── t10k-images.idx3-ubyte       # MNIST raw binary (test images)
├── t10k-labels.idx1-ubyte       # MNIST raw binary (test labels)
│
├── train_model.py               # Data loading, CNN training, plot generation
├── evaluate_model.py            # Standalone evaluation & sample predictions
├── app.py                       # Flask backend API
│
├── static/
│   ├── index.html               # Frontend SPA (draw + upload + model info)
│   ├── plots/                   # Generated PNG plots (created at runtime)
│   └── uploads/                 # Temporary upload folder
│
├── mnist_model.h5               # Saved model (created after training)
├── classification_report.txt    # Text report (created after training)
└── requirements.txt
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train the model

```bash
python train_model.py
```

This will:
- Read the raw MNIST binary files from the project root
- Train a CNN for up to 20 epochs (early stopping enabled)
- Save the best model as `mnist_model.h5`
- Generate `static/plots/training_curves.png` and `static/plots/confusion_matrix.png`
- Print the classification report

Expected test accuracy: **≥ 99%**

### 3 — (Optional) Standalone evaluation

```bash
python evaluate_model.py
```

### 4 — Start the web server

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## API Reference

### `POST /predict`

**Option A — File upload (multipart/form-data)**
```
curl -X POST -F "file=@digit.png" http://localhost:5000/predict
```

**Option B — Base64 canvas image (application/json)**
```json
{ "image": "data:image/png;base64,..." }
```

**Response**
```json
{
  "digit":         7,
  "confidence":    99.83,
  "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.83, 0.0, 0.17]
}
```

### `GET /health`
Returns `{ "status": "ok" }`.

---

## Model Architecture

```
Input  28×28×1
│
├─ Conv2D(32, 3×3) → BatchNorm → ReLU
├─ Conv2D(32, 3×3) → BatchNorm → ReLU
├─ MaxPool(2×2) → Dropout(0.25)
│
├─ Conv2D(64, 3×3) → BatchNorm → ReLU
├─ Conv2D(64, 3×3) → BatchNorm → ReLU
├─ MaxPool(2×2) → Dropout(0.25)
│
├─ Flatten
├─ Dense(256) → BatchNorm → ReLU → Dropout(0.5)
└─ Dense(10)  → Softmax

Optimizer  : Adam (lr=1e-3) with ReduceLROnPlateau
Loss       : Sparse Categorical Cross-Entropy
Augmentation: rotation ±10°, width/height shift ±10%, zoom ±10%
```
