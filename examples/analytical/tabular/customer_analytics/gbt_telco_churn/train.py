import json
import os
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append("/workspace")

from gbt import train as gbt_train  # noqa: E402

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "telco_customer_churn.csv"
ARTIFACTS_DIR = Path("artifacts")
TRAIN_ARTIFACT_DIR = ARTIFACTS_DIR / "training"
MODEL_DIR = ARTIFACTS_DIR / "model"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        print(f"Dataset already present at {RAW_DATA_PATH}")
        return

    print(f"Downloading Telco Customer Churn dataset from {DATA_URL}...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    RAW_DATA_PATH.write_bytes(response.content)
    print(f"Saved dataset to {RAW_DATA_PATH}")


def load_and_preprocess() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list, list]:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    # Drop identifier column
    df = df.drop(columns=["customerID"])  # unique identifier not useful for modeling

    # Coerce TotalCharges to numeric (there are blank strings)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Binary target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "Churn"]
    numerical_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c != "Churn"]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Churn"],
    )
    print(f"Training rows: {len(train_df):,}, Test rows: {len(test_df):,}")

    y_train = train_df["Churn"].copy()
    y_test = test_df["Churn"].copy()

    return train_df, y_train, test_df, y_test, categorical_cols, numerical_cols


def main() -> None:
    download_dataset()

    TRAIN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df, y_train, test_df, y_test, cat_cols, num_cols = load_and_preprocess()

    model = gbt_train(
        df=train_df,
        df_test=test_df,
        model_lib="binary",
        label_column="Churn",
        categorical_feature_columns=cat_cols,
        numerical_feature_columns=num_cols,
        val_size=0.2,
        log_dir=str(TRAIN_ARTIFACT_DIR),
        num_boost_round=200,
        early_stopping_rounds=25,
    )

    model.save(str(MODEL_DIR))
    print(f"Saved model artifacts to {MODEL_DIR}")

    # Evaluate on hold-out set
    y_prob = model.predict(test_df)
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    metrics = {
        "accuracy": report["accuracy"],
        "roc_auc": float(auc),
        "f1_macro": report["macro avg"]["f1-score"],
        "classification_report": report,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {METRICS_PATH}")

    # Save a small batch of sample inputs for inference demo
    sample_rows = test_df.drop(columns=["Churn"]).head(5).copy()
    sample_rows.to_csv(SAMPLE_INPUT_PATH, index=False)
    print(f"Sample inference input saved to {SAMPLE_INPUT_PATH}")


if __name__ == "__main__":
    main()
