import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Import inference methods
from app.inference import predict
from app.gradcam import predict_with_gradcam
from app.explanation import generate_explanation

app = FastAPI(
    title="HAR Prediction API",
    description="Video inference for Human Activity Recognition",
    version="1.0"
)

# Mount outputs directory so Grad-CAM images can be served statically
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Validation for supported formats
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def validate_video_file(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file extension: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
        )

@app.get("/")
def health_check():
    return {"status": "ok", "message": "HAR Model API is running."}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Endpoint 1: Predict Activity
    Input: video file upload
    Returns JSON: {"prediction": str, "confidence": float, "explanation": str}
    """
    validate_video_file(file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        label, confidence = predict(tmp_path)
        explanation = generate_explanation(tmp_path, label)
        return JSONResponse({
            "prediction": label,
            "confidence": round(confidence, 4),
            "explanation": explanation
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/predict-with-gradcam")
async def predict_gradcam_endpoint(file: UploadFile = File(...)):
    """
    Endpoint 2: Predict Activity & Generate Grad-CAM visualization
    Input: video file upload
    Returns JSON: {"prediction": str, "confidence": float, "heatmap_path": str, "explanation": str}
    """
    validate_video_file(file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        label, confidence, out_path = predict_with_gradcam(tmp_path)
        
        # Relative static URL for client access
        relative_url = f"/static/outputs/{os.path.basename(out_path)}"
        explanation = generate_explanation(tmp_path, label, heatmap_path=out_path)
        
        return JSONResponse({
            "prediction": label,
            "confidence": round(confidence, 4),
            "heatmap_path": relative_url,
            "explanation": explanation
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM Inference error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

