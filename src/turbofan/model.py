import joblib
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

class TurbofanPredictor:
    """Load and use trained turbofan model"""
    
    def __init__(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        logger.info("Model loaded")
    
    def predict(self, sensor3: float, sensor4: float) -> Tuple[str, float]:
        """Predict engine status"""
        X = np.array([[sensor3, sensor4]])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]
        
        status = "Failing Soon" if pred == 1 else "Healthy"
        confidence = float(prob[pred])
        failure_prob = float(prob[1])
        
        return {
            "status": status,
            "confidence": confidence,
            "failure_probability": failure_prob,
            "recommendation": "Schedule maintenance" if pred == 1 else "Normal operation"
        }
    
    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """Batch predictions"""
        return self.model.predict(data)
