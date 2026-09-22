# 🔢 MNIST Handwritten Digit Recognizer

> End-to-end ML web application that recognises handwritten digits (0–9) using a deep CNN trained on the MNIST dataset — draw on canvas or upload an image and get an instant prediction with confidence scores.

---

## 📋 Table of Contents

- [Project Summary](#-project-summary)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)

---

## 📝 Project Summary

This project implements a **production-ready Handwritten Digit Recognition system** using the MNIST dataset. It combines a deep **Convolutional Neural Network (CNN)** trained with TensorFlow/Keras, a **Flask REST API** backend, and a fully responsive **vanilla HTML/CSS/JavaScript** frontend — packaged as a single deployable web application.

Users interact with the system in two ways:

- ✏️ **Draw** a digit directly on an HTML5 canvas with mouse or touch
- 📁 **Upload** an existing image (PNG, JPEG, BMP, WebP)

The system preprocesses the input, runs it through the trained model, and instantly returns:
- The **predicted digit** (0–9) displayed in a large result badge
- A **confidence percentage** for the prediction
- A **full probability distribution** bar chart across all 10 digit classes

The Model Info tab displays the CNN architecture, training curves, and a confusion matrix heatmap — all rendered inside the browser.

---

## ❗ Problem Statement

Handwritten digit recognition is a fundamental challenge in the field of computer vision and optical character recognition (OCR). Despite the simplicity of the problem domain, handwritten characters exhibit significant variability in style, size, stroke width, and orientation — making it non-trivial to classify them reliably with traditional rule-based methods.

**The goal of this project** is to build a machine learning system that:
1. Learns the visual patterns of handwritten digits from labelled examples
2. Generalises well to new, unseen handwriting from real users
3. Is accessible through an intuitive web interface — no installation needed for the end user

This problem is highly relevant in real-world applications such as postal code reading, bank cheque processing, form digitisation, and assistive technology.

---

## 🎯 Objectives

| # | Objective |
|---|-----------|
| 1 | Parse raw MNIST `.idx` binary files directly without relying on pre-packaged dataset loaders |
| 2 | Normalise pixel values, add channel dimensions, and apply real-time data augmentation |
| 3 | Design and train a deep CNN achieving **≥ 99% test accuracy** on the MNIST test set |
| 4 | Implement smart inference preprocessing: background detection, auto-inversion, tight crop, aspect-preserving resize |
| 5 | Expose the trained model as a **REST API** via Flask — supporting both file uploads and base64 canvas images |
| 6 | Build a fully responsive, dark-themed **Single Page Application** with Draw, Upload, and Model Info tabs |
| 7 | Generate and embed training curves, confusion matrix, and per-class classification report |
| 8 | Save the best model checkpoint automatically using `ModelCheckpoint` callback |

---

## 📊 Dataset

The **MNIST** (Modified National Institute of Standards and Technology) dataset is the standard benchmark for handwritten digit classification.

| Property | Value |
|----------|-------|
| Total Samples | 70,000 |
| Training Samples | 60,000 |
| Test Samples | 10,000 |
| Image Dimensions | 28 × 28 pixels |
| Colour Space | Greyscale (1 channel) |
| Pixel Range | 0 (black background) – 255 (white digit) |
| Classes | 10 (digits 0 – 9) |
| Label Format | Integer (0–9) |
| File Format | IDX binary (`.idx3-ubyte` / `.idx1-ubyte`) |
| Source | [Kaggle — MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) |

The raw binary files are included in the project root and parsed using Python's `struct` module — no external dataset library is required.

---

## 🛠️ Technologies Used

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | All backend scripts and training pipeline |
| **Machine Learning** | TensorFlow / Keras | 2.16.1 | CNN model definition, training, and inference |
| **Image Processing** | Pillow (PIL) | 10.3.0 | Image decoding, greyscale conversion, crop, resize |
| **Data Science** | NumPy | 1.26.4 | Array operations, normalisation, argmax |
| **Evaluation** | scikit-learn | 1.5.0 | Confusion matrix, classification report |
| **Visualisation** | Matplotlib | 3.9.0 | Training curve plots |
| **Visualisation** | Seaborn | 0.13.2 | Confusion matrix heatmap |
| **Backend** | Flask | 3.0.3 | REST API, routing, static file serving |
| **CORS** | Flask-CORS | 4.0.1 | Cross-origin request handling |
| **Model Storage** | HDF5 / h5py | 3.11.0 | Saving and loading trained model weights |
| **Frontend** | HTML5 / CSS3 | — | Responsive SPA layout and styling |
| **Frontend** | Vanilla JavaScript | ES2020 | Canvas drawing, fetch API, DOM manipulation |

---

## 📁 Project Structure

```
MNIST_Project/
│
├── 📄 train_model.py            # CNN training pipeline
│                                #   → reads raw .idx files
│                                #   → preprocesses + augments data
│                                #   → trains CNN (up to 20 epochs)
│                                #   → saves best model as mnist_model.h5
│                                #   → generates training_curves.png & confusion_matrix.png
│
├── 📄 evaluate_model.py         # Standalone evaluation script
│                                #   → loads saved model + test set
│                                #   → prints accuracy & classification report
│                                #   → saves sample_predictions.png (5×10 grid)
│
├── 📄 app.py                    # Flask REST API
│                                #   → GET  /          → serves frontend SPA
│                                #   → POST /predict   → file upload or base64 canvas
│                                #   → GET  /health    → liveness probe
│
├── 📄 requirements.txt          # Pinned Python dependencies
├── 📄 mnist_model.h5            # Saved best model (created after training)
├── 📄 classification_report.txt # Per-class metrics (created after training)
│
├── 🗂️ static/
│   ├── 📄 index.html            # Frontend SPA (722 lines, all CSS + JS inline)
│   ├── 🗂️ plots/               # Generated plots (created at runtime)
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   └── sample_predictions.png
│   └── 🗂️ uploads/             # Temporary uploaded image storage
│
├── 🗂️ (MNIST raw binary files)
│   ├── train-images.idx3-ubyte  # 60,000 training images
│   ├── train-labels.idx1-ubyte  # 60,000 training labels
│   ├── t10k-images.idx3-ubyte   # 10,000 test images
│   └── t10k-labels.idx1-ubyte   # 10,000 test labels
```

---

## 🧠 Model Architecture

The CNN is a two-block feature extractor followed by a dense classifier head.

| # | Layer | Configuration | Output Shape |
|---|-------|--------------|--------------|
| 1 | **Input** | — | `(28, 28, 1)` |
| 2 | Conv2D + BatchNorm | 32 filters, 3×3, same padding, ReLU | `(28, 28, 32)` |
| 3 | Conv2D + BatchNorm | 32 filters, 3×3, same padding, ReLU | `(28, 28, 32)` |
| 4 | MaxPooling2D | pool size (2, 2) | `(14, 14, 32)` |
| 5 | Dropout | rate = 0.25 | `(14, 14, 32)` |
| 6 | Conv2D + BatchNorm | 64 filters, 3×3, same padding, ReLU | `(14, 14, 64)` |
| 7 | Conv2D + BatchNorm | 64 filters, 3×3, same padding, ReLU | `(14, 14, 64)` |
| 8 | MaxPooling2D | pool size (2, 2) | `(7, 7, 64)` |
| 9 | Dropout | rate = 0.25 | `(7, 7, 64)` |
| 10 | Flatten | — | `(3136,)` |
| 11 | Dense + BatchNorm | 256 units, ReLU | `(256,)` |
| 12 | Dropout | rate = 0.5 | `(256,)` |
| 13 | **Dense (Output)** | 10 units, Softmax | `(10,)` |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (`lr = 1e-3`) |
| Loss Function | Sparse Categorical Cross-Entropy |
| Batch Size | 128 |
| Max Epochs | 20 (EarlyStopping enabled) |
| Monitor Metric | Validation Accuracy |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | patience=6, restore best weights |
| Augmentation | Rotation ±10°, shift ±10%, zoom ±10% |
| Target Accuracy | **≥ 99%** on test set |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- The 4 MNIST raw binary files in the project root (included)

---

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Train the model

```bash
python train_model.py
```

This will:
- Read the raw MNIST `.idx` binary files
- Train the CNN for up to 20 epochs with early stopping
- Save the best checkpoint as `mnist_model.h5`
- Generate `static/plots/training_curves.png` and `static/plots/confusion_matrix.png`
- Write `classification_report.txt` with per-digit precision, recall, and F1

> **Expected test accuracy: ≥ 99%**

---

### Step 3 — (Optional) Standalone evaluation

```bash
python evaluate_model.py
```

Prints accuracy + classification report and saves a 5×10 grid of sample predictions to `static/plots/sample_predictions.png`.

---

### Step 4 — Start the web application

```bash
python app.py
```

Open your browser at **[http://localhost:5000](http://localhost:5000)**

---

## 🔌 API Reference

### `POST /predict`

Accepts **either** a multipart file upload or a JSON base64 image.

**Option A — File upload**
```bash
curl -X POST -F "file=@digit.png" http://localhost:5000/predict
```

**Option B — Base64 canvas (application/json)**
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

| Field | Type | Description |
|-------|------|-------------|
| `digit` | `int` | Predicted digit class (0–9) |
| `confidence` | `float` | Confidence % for the top prediction |
| `probabilities` | `float[10]` | Softmax probability % for each digit class |

**Error responses** return `{ "error": "<message>" }` with the appropriate HTTP status code (`400`, `415`, `503`, `500`). A `503` is returned when the model has not been trained yet.

---

### `GET /health`

```bash
curl http://localhost:5000/health
# → { "status": "ok" }
```

---

## 📄 License

This project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) which is freely available for non-commercial use. The source code is open for educational and research purposes.
