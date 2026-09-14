# Aircraft Engine Predictive Maintenance

Production ML system for turbofan engine failure prediction. Achieves 89% failure detection rate on NASA C-MAPSS dataset with Random Forest classification and sensor analysis.

## Problem Statement

Aircraft maintenance requires detecting engine failures before they occur to prevent in-flight emergencies and reduce operational costs. The challenge: false negatives (missed failures) are far more costly than false positives (unnecessary maintenance). Standard ML metrics optimize for accuracy; safety-critical systems require recall-optimized models.

## Approach

Dataset: NASA C-MAPSS turbofan degradation (20,631 observations, 100 engines, 21 sensors)

Feature Analysis: Identified sensor3 (temperature) and sensor4 (pressure) as strongest degradation indicators through statistical comparison of healthy vs failing engines.

Model: Random Forest (100 estimators, balanced class weights) optimized for failure detection over general accuracy.

## Results

Failure Detection: 89% recall (554 out of 620 failing engines caught)
Overall Accuracy: 92%
False Alarms: 264 (acceptable trade-off for safety-critical applications)

## Architecture

Training Pipeline (src/turbofan/train.py)
- Loads NASA C-MAPSS dataset
- Calculates remaining useful life per engine
- Trains Random Forest with class imbalance handling
- Generates diagnostic visualizations

API Server (src/turbofan/api.py)
- FastAPI REST endpoint
- POST /predict accepts sensor3, sensor4 readings
- Returns prediction, confidence, failure probability, maintenance recommendation
- Interactive documentation at /docs

Model Class (src/turbofan/model.py)
- Reusable predictor for single and batch predictions
- Loads pre-trained model from joblib

Monitoring (src/turbofan/monitoring.py)
- Data drift detection using Kolmogorov-Smirnov test
- Alerts on distribution shift

## Deployment

Docker: Containerized for reproducibility across environments
GitHub Actions: Automated testing on every push
Testing: pytest suite with 3/3 tests passing

## Usage

Training
python src/turbofan/train.py

Testing
pytest tests/ -v

API Server
uvicorn src.turbofan.api:app --reload
Visit http://localhost:8000/docs for interactive documentation

Docker
docker build -t turbofan .
docker run -p 8000:8000 turbofan

## Technical Stack

Python, pandas, NumPy, scikit-learn, SciPy, FastAPI, Docker, pytest, GitHub Actions

## Key Insight

Safety-critical ML optimization requires domain-aware cost functions rather than generic accuracy metrics. In aerospace maintenance, missing one failure is costlier than multiple false alarms. This system prioritizes recall at the expense of precision—a deliberate choice reflecting the true operational cost structure.

## Files

src/turbofan/train.py: Training pipeline with logging and type hints
src/turbofan/model.py: TurbofanPredictor class for inference
src/turbofan/api.py: FastAPI server for real-time predictions
src/turbofan/monitoring.py: Data drift detection
tests/test_model.py: Automated test suite
Dockerfile: Container specification
setup.py: Package configuration
requirements.txt: Dependencies

## Author

Nia Racheva
niaracheva05@gmail.com
github.com/nia05-tch
