import pytest
import numpy as np
from sklearn.datasets import make_classification
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules import correctly"""
    from src.turbofan import train, model, api
    assert train is not None
    assert model is not None
    assert api is not None

def test_model_class():
    """Test TurbofanPredictor class"""
    from src.turbofan.model import TurbofanPredictor
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import tempfile
    
    # Create model with 2 features (sensor3, sensor4)
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        joblib.dump(rf, f.name)
        temp_path = f.name
    
    predictor = TurbofanPredictor(temp_path)
    result = predictor.predict(0.5, 0.5)
    
    assert 'status' in result
    assert 'confidence' in result
    assert result['confidence'] > 0 and result['confidence'] <= 1
    
    os.remove(temp_path)

def test_batch_prediction():
    """Test batch predictions"""
    from src.turbofan.model import TurbofanPredictor
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import tempfile
    
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        joblib.dump(rf, f.name)
        temp_path = f.name
    
    predictor = TurbofanPredictor(temp_path)
    batch = np.array([[0.5, 0.5]])
    preds = predictor.predict_batch(batch)
    
    assert len(preds) == 1
    
    os.remove(temp_path)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
