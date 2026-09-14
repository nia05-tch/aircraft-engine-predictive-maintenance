import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)
sns.set_style('darkgrid')

def load_data(data_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}")
    column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=column_names, dtype=np.float64, na_values='NA')
    logger.info(f"Loaded: {df.shape[0]} observations, {df.shape[1]} features")
    return df

def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating RUL...")
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id', how='left')
    df['rul'] = df['max_cycle'] - df['cycle']
    logger.info(f"RUL range: {df['rul'].min():.0f} - {df['rul'].max():.0f} cycles")
    return df

def create_binary_target(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    logger.info(f"Creating binary target (threshold: {threshold} cycles)")
    df['will_fail_soon'] = (df['rul'] <= threshold).astype(int)
    class_dist = df['will_fail_soon'].value_counts()
    logger.info(f"Class distribution: Healthy={class_dist[0]}, Failing={class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")
    return df

def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        max_depth=10,
        n_jobs=-1
    )
    model.fit(X, y)
    logger.info("Model training complete")
    return model

def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.2%}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Healthy', 'Failing Soon'])}")
    
    cm = confusion_matrix(y_test, y_pred)
    caught, missed, false_alarms = cm[1, 1], cm[1, 0], cm[0, 1]
    total_failures = caught + missed
    recall = (caught / total_failures) * 100
    
    logger.info(f"Aerospace metrics:")
    logger.info(f"  Caught: {caught}/{total_failures} ({recall:.1f}%)")
    logger.info(f"  Missed: {missed} ({missed/total_failures*100:.1f}%)")
    logger.info(f"  False alarms: {false_alarms}")

def save_model(model: RandomForestClassifier, path: str) -> None:
    logger.info(f"Saving model to {path}")
    joblib.dump(model, path)
    logger.info("Model saved")

def main() -> None:
    logger.info("Starting training pipeline")
    os.chdir('/Users/niaracheva/Desktop/aircraft-engine-predictive-maintenance')
    
    df = load_data('data/train_FD001.txt')
    df = calculate_rul(df)
    df = create_binary_target(df, threshold=30)
    
    logger.info("Generating visualizations...")
    engine_1 = df[df['engine_id'] == 1]
    plt.figure(figsize=(10, 4))
    plt.plot(engine_1['cycle'], engine_1['rul'], linewidth=2)
    plt.xlabel('Cycle'), plt.ylabel('Remaining Useful Life')
    plt.title('Engine 1: RUL degradation over time'), plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/engine_degradation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    X = df[['sensor3', 'sensor4']]
    y = df['will_fail_soon']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    model = train_model(X_train.values, y_train.values)
    evaluate_model(model, X_test.values, y_test.values)
    
    os.makedirs('models', exist_ok=True)
    save_model(model, 'models/turbofan_failure_predictor.pkl')
    logger.info("Training pipeline complete")

if __name__ == '__main__':
    main()
