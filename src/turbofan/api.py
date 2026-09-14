import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Turbofan Predictor API",
    description="Predict aircraft engine failure from sensor readings",
    version="0.1.0"
)

try:
    model = joblib.load('models/turbofan_failure_predictor.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

class PredictionRequest(BaseModel):
    sensor3: float
    sensor4: float

class PredictionResponse(BaseModel):
    status: str
    confidence: float
    failure_probability: float
    recommendation: str

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict engine status from sensor readings"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        X = np.array([[request.sensor3, request.sensor4]])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        
        status = "Failing Soon" if pred == 1 else "Healthy"
        confidence = float(prob[pred])
        failure_prob = float(prob[1])
        recommendation = "Schedule maintenance" if pred == 1 else "Normal operation"
        
        logger.info(f"Prediction: {status} (confidence: {confidence:.2f})")
        
        return PredictionResponse(
            status=status,
            confidence=confidence,
            failure_probability=failure_prob,
            recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    """API documentation"""
    return {
        "message": "Turbofan Engine Predictor",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }
