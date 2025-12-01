# Deepfake Detection - Video Classification
Deep learning model untuk mendeteksi video deepfake menggunakan EfficientNet-B0 + BiGRU.

Table of Contents

## Overview
Model ini dapat mengklasifikasikan video sebagai REAL (asli) atau FAKE (deepfake) dengan menganalisis:
- 8 frames dari bagian tengah video (mid-clip 2 detik)
- Face detection & cropping menggunakan MTCNN
- Temporal patterns menggunakan Bidirectional GRU

### Quick Start
```
streamlit run app.py
```

### Architecture
Input Video → Frame Extraction → Face Detection → EfficientNet-B0 → BiGRU → Classification

### Dataset
Model di-training pada Deep Fake Detection (DFD) dataset:
- Total videos: 3,431
- FAKE: 3,068 videos (89.4%)
- REAL: 363 videos (10.6%)
- Resolution: 1920×1080, 24 fps
- Sumber: [https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset]

Data Split
- Training: 2,744 videos (80%)
- Validation: 343 videos (10%)
- Test: 344 videos (10%)
- Stratified split untuk maintain class distribution.

### Training Configuration:
- Optimizer: AdamW (lr=2e-3, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss + class weighting
- Epochs: 12 (warmup: 3, fine-tune: 9)
- Batch size: 16
- GPU: Tesla P100 (Kaggle)
- Training time: ~2 hours

Training Strategy
- Warmup (Epoch 1-3): EfficientNet frozen, train BiGRU+FC only
- Fine-tuning (Epoch 4-12): All layers trainable, ReduceLROnPlateau scheduler

### Project Structure
```text
project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── notebooks/                   # Notebooks
│   └── deepfake-detection.ipynb # Training notebook
└── src/                         # Source code
    └── deepfake_inference.py    # Inference pipeline
```
