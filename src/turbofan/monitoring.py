import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def detect_drift(train_data: np.ndarray, new_data: np.ndarray, threshold: float = 0.05) -> tuple:
    """Detect data drift using Kolmogorov-Smirnov test"""
    for col in range(train_data.shape[1]):
        stat, p_value = stats.ks_2samp(train_data[:, col], new_data[:, col])
        if p_value < threshold:
            logger.warning(f"Drift detected in feature {col}: p-value={p_value:.4f}")
            return True, f"Drift detected in feature {col}"
    
    return False, "No drift detected"
