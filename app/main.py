# app/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

# Import the build_system function that initializes your IntegratedRealLLMSystem
from .system import build_system

app = FastAPI(
    title="Menstrual Health RAG API",
    description="FastAPI wrapper for Graph + Transformer + RAG system",
    version="0.1"
)

# Config
CSV_PATH = os.environ.get("DATA_CSV", "/app/data/cleaned_dataset.csv")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
# ensure Render's PORT is used by Uvicorn (env in Dockerfile)
SYSTEM = None

@app.on_event("startup")
def startup_event():
    global SYSTEM
    try:
        # Build system (this may take some seconds — sentence-transformers will load models)
        SYSTEM = build_system(csv_path=CSV_PATH, openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        # keep startup going but log exception
        print("❌ Failed to build system on startup:", e)
        traceback.print_exc()

@app.get("/health")
def health():
    return {"status": "ok", "system_loaded": SYSTEM is not None}

class PredictResponse(BaseModel):
    predicted_phase: str
    days_to_menstrual: float
    confidence: str
    confidence_score: float
    explanation: Optional[str] = None

@app.get("/predict/{user_id}", response_model=PredictResponse)
def predict(user_id: int):
    global SYSTEM
    if SYSTEM is None:
        raise HTTPException(status_code=503, detail="System not yet initialized.")
    try:
        result = SYSTEM.predict_with_explanation(user_id)
        if result is None:
            raise HTTPException(status_code=404, detail="No data/prediction for user.")
        
        return {
            "predicted_phase": result["prediction"]["predicted_phase"],
            "days_to_menstrual": float(result["prediction"]["days_to_menstrual"]),
            "confidence": result["prediction"]["confidence"],
            "confidence_score": float(result["prediction"]["confidence_score"]),
            "explanation": result["explanation"]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Optional training endpoint (use with caution)
@app.post("/train")
def train(num_epochs: int = 10):
    """
    Starts a training run synchronously. Note: running long training on a Render web
    instance can time out — better to train offline and upload model artifacts.
    """
    global SYSTEM
    if SYSTEM is None:
        raise HTTPException(status_code=503, detail="System not initialized.")
    try:
        SYSTEM.train_model(num_epochs=num_epochs)
        return {"status": "training_started", "epochs": num_epochs}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
