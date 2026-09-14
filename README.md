# Aircraft Engine Remaining Useful Life Prediction

Production ML system for predicting turbofan engine failures before they occur. Achieves 89% failure detection rate on NASA C-MAPSS benchmark dataset by optimizing for safety-critical recall rather than generic accuracy metrics.

**Status:** Complete research + production deployment ready  
**Model:** Random Forest (optimized for recall)  
**Dataset:** NASA C-MAPSS (20,631 observations, 100 turbofan engines)  
**Performance:** 89% failure detection, 92% overall accuracy

---

## Problem Statement

Aircraft engine maintenance traditionally relies on fixed maintenance schedules, resulting in unnecessary downtime and missed failures. Predictive maintenance can reduce costs and prevent catastrophic failures—but standard ML metrics (accuracy, precision, F1) don't align with safety-critical requirements.

**The Core Challenge:**
- False negative (missing a failure) = potential in-flight emergency, loss of life, massive liability
- False positive (unnecessary maintenance) = extra labor costs, minor operational disruption
- Cost of false negative >> Cost of false positive

**Traditional Approach:** Optimize for accuracy  
**This System:** Optimize for recall (catch failures, accept false alarms)

---

## Approach

### Dataset

**NASA C-MAPSS Turbofan Engine Degradation Dataset**

| Metric | Value |
|--------|-------|
| Training Engines | 100 turbofans |
| Training Cycles | 20,631 |
| Sensors | 21 operational measurements |
| Operational Settings | 3 parameters (altitude, throttle, temperature) |
| Failure Scenarios | 4 subsets (FD001-FD004) |

**Citation:**
```bibtex
@dataset{c_mapss_2008,
  title={Turbofan Engine Degradation Simulation Data Set},
  author={Saxena, A. and Goebel, K.},
  year={2008},
  publisher={NASA Ames Prognostics Center of Excellence},
  url={https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/}
}
```

**Dataset:** https://www.kaggle.com/code/wassimderbel/nasa-predictive-maintenance-rul

### Feature Engineering

**Remaining Useful Life (RUL) Calculation:**
- For each engine, identify the cycle at which degradation accelerates
- RUL at cycle t = (max_cycle - t)
- Create binary target: healthy (RUL > threshold) vs. failing (RUL ≤ threshold)

**Key Sensors Identified:**
- **Sensor 3 (Temperature):** Strongest degradation signal (p < 0.001)
- **Sensor 4 (Pressure):** Secondary degradation indicator
- Used statistical tests (Mann-Whitney U) to rank features

**Class Imbalance Handling:**
- Imbalanced dataset: ~85% healthy engines, ~15% failing
- Solution: Random Forest with `class_weight='balanced'`
- Prevents model from defaulting to "always predict healthy"

### Model Architecture

**Algorithm:** Random Forest Classifier
- Estimators: 100 trees
- Max depth: 15 (prevent overfitting)
- Class weights: Balanced (penalize false negatives)
- Hyperparameter tuning: GridSearchCV on 80/20 train/validation split

**Why Random Forest for safety-critical systems?**
1. Interpretable: Feature importance shows which sensors matter
2. Robust: Ensemble approach reduces variance
3. Handles non-linear relationships: Engine degradation is complex
4. Confidence estimates: Can threshold on prediction probability

---

## Results

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Recall (Failure Detection) | 89% | Catch 554 out of 620 failing engines |
| Precision | 68% | Some false alarms (acceptable for safety) |
| Overall Accuracy | 92% | Strong general performance |
| False Negatives | 66 | 66 failures missed (tolerable risk) |
| False Positives | 264 | 264 unnecessary maintenance alerts (acceptable cost) |

### Cost-Benefit Analysis

**Assumption:** Preventing 1 failure is worth ~100x unnecessary maintenance

| Scenario | Cost |
|----------|------|
| Preventing 554 failures | 554 × prevention benefit |
| 264 false alarms | 264 × maintenance cost |
| 66 missed failures | 66 × catastrophic loss |
| **Net benefit** | Positive (failure prevention >> maintenance cost) |

### Comparison: Standard ML vs. This System

| Approach | Accuracy | Recall | Strategy |
|----------|----------|--------|----------|
| Accuracy-optimized | 94% | 72% | Misses 28% of failures |
| **This system** | **92%** | **89%** | **Catches 89% of failures** |

Trade-off: Lose 2% accuracy, gain 17% better failure detection.

---

## System Architecture

### Directory Structure

```
rul_toolkit/
├── evaluation/
│   ├── decision_cost.py        # Cost-aware metrics
│   ├── physics_violations.py   # Monotonicity checks
│   ├── metrics.py              # RMSE, MAE, PHM score
│   └── __init__.py
├── preprocessing/
│   ├── loaders.py              # NASA C-MAPSS loader
│   ├── cleaner.py              # Normalize, detect anomalies
│   ├── validators.py           # Data quality checks
│   └── __init__.py
├── models/
│   ├── baseline.py             # Simple baseline models
│   ├── gradient_boosting.py    # RF, LightGBM
│   └── __init__.py
└── uncertainty/
    ├── calibration.py          # Confidence estimation
    ├── intervals.py            # Prediction intervals
    └── __init__.py
```

### Training Pipeline

```
1. Data Loading (loaders.py)
   ↓
2. RUL Calculation per Engine
   ↓
3. Feature Preprocessing (cleaner.py)
   → Normalize operational settings
   → Handle anomalies
   → Engineer derived features
   ↓
4. Dataset Splitting (stratified train/test)
   ↓
5. Model Training (gradient_boosting.py)
   → Grid search hyperparameters
   → Class weight balancing
   ↓
6. Model Evaluation (metrics.py)
   → Accuracy, recall, precision
   → Decision cost analysis
   → Uncertainty quantification
   ↓
7. Results & Visualization
```

### API Server (Deployment)

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "sensor_3": 621.2,
  "sensor_4": 1589.3,
  "sensor_6": 1407.6,
  ...
}
```

**Response:**
```json
{
  "prediction": "FAILING",
  "failure_probability": 0.87,
  "confidence": 0.92,
  "maintenance_recommendation": "Schedule immediate inspection",
  "sensors_of_concern": ["sensor_3", "sensor_4"],
  "model_version": "v1.0"
}
```

---

## Deployment

### Docker (Production)

```bash
# Build image
docker build -t aircraft-rul:latest .

# Run container
docker run -p 8000:8000 aircraft-rul:latest
```

**Features:**
- Reproducible environment across dev/staging/production
- Pre-trained model included in image
- Health check endpoint: `GET /health`
- Structured JSON logging for monitoring

### GitHub Actions (CI/CD)

Automated on every push:
- Unit tests (pytest)
- Model prediction accuracy check
- Docker image build
- (Optional) Push to registry

---

## Usage

### Training

```bash
python -m rul_toolkit.train --data-dir ./data --output-model model.pkl
```

**Output:**
- `model.pkl` — Trained Random Forest
- `evaluation_report.json` — Performance metrics
- `feature_importance.csv` — Which sensors matter
- `plots/` — Diagnostic visualizations

### Prediction (Batch)

```python
from rul_toolkit.models import RandomForestRUL

model = RandomForestRUL.load('model.pkl')

# Single prediction
prediction = model.predict(sensor_readings)
print(f"Failure probability: {prediction.failure_prob}")

# Batch predictions
predictions = model.predict_batch(sensor_readings_array)
```

### Prediction (API)

```bash
# Start server
uvicorn app:app --port 8000

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sensor_data.json

# Interactive docs
open http://localhost:8000/docs
```

### Testing

```bash
pytest tests/ -v

# Expected output:
# test_model_loads :: PASSED
# test_prediction_shape :: PASSED
# test_batch_prediction :: PASSED
# ==== 3 passed ====
```

---

## Key Insights

### 1. Safety-Critical Metrics Matter

Standard ML focuses on accuracy. Aerospace maintenance cares about **recall**—catching failures before they happen. This system explicitly prioritizes missing zero failures over generating some false alarms.

**Lesson:** Domain context changes the optimization target.

### 2. Feature Importance Reveals Degradation

The top 3 sensors (temperature, pressure, speed) account for 70% of failure predictability. Other sensors have marginal value but are kept for robustness.

**Lesson:** Simple interpretable features often beat complex engineered features.

### 3. Class Imbalance Requires Intentional Handling

With 85% healthy engines, a naive model predicting "always healthy" achieves 85% accuracy but 0% recall. Class weighting forces the model to learn failure patterns.

**Lesson:** Accuracy is misleading for imbalanced datasets.

### 4. Ensemble Models Provide Confidence

Random Forest output is more trustworthy than single-tree decisions. Confidence scores enable threshold-tuning for different operational risk profiles.

**Lesson:** Ensemble methods improve both accuracy and interpretability.

---

## Limitations & Future Work

### Known Limitations

1. **Trained on Simulation Data**
   - NASA C-MAPSS is synthetic (high-fidelity simulation, not real flights)
   - Real engine data may have different failure modes or sensor patterns
   - Mitigation: Transfer learning with small real-world dataset

2. **Single Failure Mode per Subset**
   - Each FD subset has one dominant failure type
   - Real engines fail in multiple ways simultaneously
   - Solution: Multi-task learning (predict multiple failure modes)

3. **No Temporal Dynamics**
   - Treats each cycle independently
   - Ignores recent degradation trends
   - Improvement: Add LSTM or temporal CNN for sequence modeling

### Future Enhancements

- [ ] **Physics-Informed Models:** Incorporate thermodynamic constraints
- [ ] **Uncertainty Quantification:** Bayesian methods for confidence intervals
- [ ] **Online Learning:** Retrain on streaming sensor data
- [ ] **Explainability:** SHAP values for per-prediction explanations
- [ ] **Multi-Task Learning:** Predict RUL for different engine components
- [ ] **Transfer Learning:** Fine-tune on real-world engine data

---

## Technical Stack

**Data & ML:** pandas, NumPy, scikit-learn, SciPy  
**Deployment:** FastAPI, Docker  
**Testing:** pytest  
**CI/CD:** GitHub Actions  
**Monitoring:** Structured logging, health endpoints  

**Python Version:** 3.8+  
**Dependencies:** See `requirements.txt`

---

## References

1. Saxena, A., & Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set." NASA Ames Prognostics Data Repository.
2. Coble, J. B., et al. (2015). "Identifying and Mitigating Uncertainty in Prognostics." Annual Conference of the Prognostics and Health Management Society.
3. ISO 13379-1: Condition Monitoring — Data Interpretation and Diagnostics

---

## Contact

**Author:** Nia Racheva  
**Email:** niaracheva05@gmail.com  
**GitHub:** github.com/nia05-tch  
**VU Amsterdam:** B.S. Artificial Intelligence (2027)

---

## License

MIT License — See LICENSE file for details

This work is part of research conducted at Vrije Universiteit Amsterdam.
