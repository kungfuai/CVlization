import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from pyod.models.iforest import IForest
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
CONFIG_PATH = MODEL_DIR / "config.json"
DETECTOR_PATH = MODEL_DIR / "iforest_detector.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
REPORT_PATH = ARTIFACTS_DIR / "classification_report.json"
CONFUSION_PATH = ARTIFACTS_DIR / "confusion_matrix.json"

TARGET_COLUMN = "Class"
MAX_NORMAL_SAMPLES = int(os.getenv("MAX_NORMAL_SAMPLES", "80000"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = 42


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        print(f"Dataset already present at {RAW_DATA_PATH}")
        return

    print(f"Downloading credit card fraud dataset from {DATA_URL} ...")
    with requests.get(DATA_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with RAW_DATA_PATH.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1_048_576):
                if chunk:
                    f.write(chunk)
    print(f"Saved dataset to {RAW_DATA_PATH}")


def load_and_sample() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    fraud_df = df[df[TARGET_COLUMN] == 1]
    normal_df = df[df[TARGET_COLUMN] == 0]
    print(f"Fraudulent rows: {len(fraud_df):,}; Normal rows: {len(normal_df):,}")

    if MAX_NORMAL_SAMPLES > 0 and len(normal_df) > MAX_NORMAL_SAMPLES:
        normal_df = normal_df.sample(MAX_NORMAL_SAMPLES, random_state=RANDOM_STATE)
        print(f"Down-sampled normal transactions to {len(normal_df):,} rows")

    sampled_df = pd.concat([normal_df, fraud_df], axis=0).sample(frac=1.0, random_state=RANDOM_STATE)
    return sampled_df.reset_index(drop=True)


def train_model(
    df: pd.DataFrame,
) -> Tuple[IForest, StandardScaler, Dict[str, float], Dict, list, pd.DataFrame]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    contamination = float(np.mean(y_train))
    print(f"Estimated contamination rate: {contamination:.6f}")

    detector = IForest(
        contamination=max(contamination, 1e-4),
        n_estimators=300,
        max_samples="auto",
        random_state=RANDOM_STATE,
    )
    detector.fit(X_train_scaled)
    print("Finished training Isolation Forest detector")

    scores = detector.decision_function(X_test_scaled)
    preds = detector.predict(X_test_scaled)

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    report = classification_report(y_test, preds, output_dict=True)
    conf = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(report["1"]["f1-score"]),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "contamination": float(contamination),
        "test_samples": int(len(y_test)),
    }

    predictions_df = X_test.copy()
    predictions_df[TARGET_COLUMN] = y_test
    predictions_df["anomaly_score"] = scores
    predictions_df["is_anomaly_pred"] = preds

    return detector, scaler, metrics, report, conf, predictions_df


def save_artifacts(
    detector: IForest,
    scaler: StandardScaler,
    metrics: Dict[str, float],
    report: Dict,
    confusion: list,
    predictions_df: pd.DataFrame,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(detector, DETECTOR_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved detector to {DETECTOR_PATH}")

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {METRICS_PATH}")

    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2)
    with CONFUSION_PATH.open("w") as f:
        json.dump({"matrix": confusion}, f, indent=2)

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")

    sample_inputs = predictions_df.drop(columns=[TARGET_COLUMN, "anomaly_score", "is_anomaly_pred"]).head(5)
    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)
    print(f"Sample inputs saved to {SAMPLE_INPUT_PATH}")

    config = {
        "model": "pyod.iforest",
        "features": predictions_df.drop(columns=[TARGET_COLUMN, "anomaly_score", "is_anomaly_pred"]).columns.tolist(),
        "target": TARGET_COLUMN,
        "contamination": metrics["contamination"],
    }
    with CONFIG_PATH.open("w") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    download_dataset()
    df = load_and_sample()
    detector, scaler, metrics, report, confusion, predictions_df = train_model(df)
    save_artifacts(detector, scaler, metrics, report, confusion, predictions_df)


if __name__ == "__main__":
    main()
