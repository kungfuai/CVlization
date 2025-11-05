import json
import zipfile
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

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "bank_marketing.zip"
CSV_NAME = "bank-additional/bank-additional-full.csv"
CSV_PATH = DATA_DIR / "bank_marketing.csv"

ARTIFACTS_DIR = Path("artifacts")
TRAIN_ARTIFACT_DIR = ARTIFACTS_DIR / "training"
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"

TARGET_COLUMN = "y"


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        print(f"Dataset already present at {CSV_PATH}")
        return

    print(f"Downloading marketing dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        with zf.open(CSV_NAME) as source, CSV_PATH.open("wb") as dest:
            dest.write(source.read())
    print(f"Saved dataset to {CSV_PATH}")


def load_and_preprocess() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(CSV_PATH, sep=";")
    print(f"Loaded {len(df):,} rows")

    df[TARGET_COLUMN] = (df[TARGET_COLUMN].str.strip().str.lower() == "yes").astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    return df, df[TARGET_COLUMN]


def main() -> None:
    download_dataset()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df, labels = load_and_preprocess()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=1701,
        stratify=df[TARGET_COLUMN],
    )
    print(f"Training rows: {len(train_df):,}, Test rows: {len(test_df):,}")

    pos_count = int((train_df[TARGET_COLUMN] == 1).sum())
    neg_count = int((train_df[TARGET_COLUMN] == 0).sum())
    scale_pos_weight = float(neg_count / pos_count) if pos_count else 1.0

    categorical_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != TARGET_COLUMN]
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c != TARGET_COLUMN]

    model = gbt_train(
        df=train_df,
        df_test=test_df,
        model_lib="binary",
        label_column=TARGET_COLUMN,
        categorical_feature_columns=categorical_cols,
        numerical_feature_columns=numerical_cols,
        val_size=0.2,
        log_dir=str(TRAIN_ARTIFACT_DIR),
        params_override={
            "scale_pos_weight": scale_pos_weight,
            "learning_rate": 0.05,
            "min_data_in_leaf": 25,
        },
        num_boost_round=400,
        early_stopping_rounds=40,
    )

    model.save(str(MODEL_DIR))
    print(f"Saved model artifacts to {MODEL_DIR}")

    y_prob = model.predict(test_df)
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(test_df[TARGET_COLUMN], y_pred, output_dict=True)
    auc = roc_auc_score(test_df[TARGET_COLUMN], y_prob)
    metrics = {
        "accuracy": report["accuracy"],
        "roc_auc": float(auc),
        "f1_macro": report["macro avg"]["f1-score"],
        "classification_report": report,
    }

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {METRICS_PATH}")

    booster = model.booster
    gains = booster.feature_importance(importance_type="gain")
    features = booster.feature_name()
    if len(gains) and len(features):
        fi_df = (
            pd.DataFrame({"feature": features, "importance_gain": gains})
            .sort_values("importance_gain", ascending=False)
        )
        fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        print(f"Feature importances saved to {FEATURE_IMPORTANCE_PATH}")

    sample_rows = test_df.drop(columns=[TARGET_COLUMN]).head(5).copy()
    sample_rows.to_csv(SAMPLE_INPUT_PATH, index=False)
    print(f"Sample inference input saved to {SAMPLE_INPUT_PATH}")


if __name__ == "__main__":
    main()
