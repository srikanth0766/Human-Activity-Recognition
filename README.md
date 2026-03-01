# Human Activity Recognition (HAR) Backend API

This repository contains the production-ready FastAPI backend for the Explainable Human Activity Recognition (HAR) deep learning model trained on the UCF50 dataset.

## Features
- **Accurate Action Tracking**: Achieves 99%+ accuracy over 50 real-world human activity classes.
- **Explainability**: Generates Grad-CAM visual heatmaps showcasing structural decision-making. 
- **Production Ready**: Lightweight FastAPI backend mapped against raw PyTorch weights, cleanly decoupling training artifacts from execution pipelines.

## Project Structure
```
har_project/
├── app/
│   ├── main.py             # FastAPI entrypoint and routes
│   ├── inference.py        # Core PyTorch predict loops
│   ├── model_loader.py     # Weights mapping logic
│   ├── video_utils.py      # OpenCV to PyTorch tensor conversions
│   └── gradcam.py          # Heatmap generation and injection
├── models/
│   └── architecture.py     # ResNet50 + BiLSTM model definitions
├── weights/
│   └── best_model.pth      # 50-class UCF PyTorch Model Weights
├── static/
│   └── outputs/            # Generated visual explainability maps
└── requirements.txt        # Explicit python dependency tree
```

## Setup & Running

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

*(Note: Requires valid `best_model.pth` placed in the `weights/` directory before starting the application).*

2. **Start the Server**
Launch the Uvicorn ASGI server bridging the FastAPI instances from the terminal located inside the `har_project` directory:

```bash
uvicorn app.main:app --reload
```

## API Methods

### 1. `POST /predict`
Submit a raw video tracking general action probabilities.

**Response Syntax:**
```json
{
    "prediction": "JumpingJack",
    "confidence": 0.985
}
```

### 2. `POST /predict-with-gradcam`
Calculates label confidences natively, whilst generating structurally overlaid heatmaps of decisive model activations on the original spatial frames.

**Response Syntax:**
```json
{
    "prediction": "JumpingJack",
    "confidence": 0.985,
    "heatmap_path": "/static/outputs/gradcam_xxxx.jpg"
}
```
