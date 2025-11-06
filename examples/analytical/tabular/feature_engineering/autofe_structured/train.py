import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from featuretools import dfs
from featuretools.entityset import EntitySet
from featuretools.primitives import (Count, Max, Mean, Min, Sum)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "telco_churn.csv"

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
FEATURE_DIR = ARTIFACTS_DIR / "features"
RAW_FEATURES_PATH = FEATURE_DIR / "raw_features.parquet"
ENGINEERED_FEATURES_PATH = FEATURE_DIR / "engineered_features.parquet"
TRAIN_REPORT_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_PATH = MODEL_DIR / "gbm_classifier.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"

TARGET = "Churn"
ID_COL = "customerID"
TEST_SIZE = 0.2
RANDOM_STATE = 1337


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        print(f"Dataset already present at {RAW_DATA_PATH}")
        return

    print(f"Downloading dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    RAW_DATA_PATH.write_bytes(response.content)
    print(f"Saved dataset to {RAW_DATA_PATH}")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} rows")
    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.dropna()

    y = (df[TARGET].str.lower() == "yes").astype(int)
    X = df.drop(columns=[TARGET])

    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    return X, y


def profile_and_save(X: pd.DataFrame) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(RAW_FEATURES_PATH, index=False)
    profile = {
        "rows": int(len(X)),
        "columns": X.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
        "missing": X.isnull().sum().to_dict(),
    }
    (ARTIFACTS_DIR / "profile.json").write_text(json.dumps(profile, indent=2))


def build_featuretools_matrix(X: pd.DataFrame) -> pd.DataFrame:
    es = EntitySet(id="customers")
    es = es.add_dataframe(
        dataframe_name="records",
        dataframe=X.reset_index(drop=True),
        make_index=True,
        index="record_id",
    )

    feature_matrix, _ = dfs(
        entityset=es,
        target_dataframe_name="records",
        agg_primitives=[Sum, Mean, Max, Min, Count],
        trans_primitives=["day", "month", "weekday"],
        max_depth=1,
        verbose=True,
    )

    engineered = pd.concat([X.reset_index(drop=True), feature_matrix.reset_index(drop=True)], axis=1)
    engineered = engineered.loc[:, ~engineered.columns.duplicated()]
    engineered.to_parquet(ENGINEERED_FEATURES_PATH, index=False)
    return engineered


def build_pipeline(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", "passthrough"),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return transformer, numeric_cols, categorical_cols


def train_model(X: pd.DataFrame, y: pd.Series, transformer: ColumnTransformer) -> Pipeline:
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("transform", transformer),
        ("model", model),
    ])
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "roc_auc": float(auc),
        "accuracy": float(report["accuracy"]),
        "precision": report.get("1", {}).get("precision", float("nan")),
        "recall": report.get("1", {}).get("recall", float("nan")),
        "f1": report.get("1", {}).get("f1-score", float("nan")),
    }
    return metrics


def save_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, float],
    feature_cols: Dict[str, List[str]],
    numeric_cols: List[str],
    categorical_cols: List[str],
    sample_inputs: pd.DataFrame,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    with TRAIN_REPORT_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "feature_columns": feature_cols,
        "model": "GradientBoostingClassifier",
        "transformer": "ColumnTransformer[StandardScaler, OneHotEncoder]",
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)


def main() -> None:
    download_dataset()
    df = load_dataset()
    X, y = preprocess(df)
    profile_and_save(X)
    engineered = build_featuretools_matrix(X)

    X_train, X_test, y_train, y_test = train_test_split(
        engineered,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    transformer, numeric_cols, categorical_cols = build_pipeline(X_train)
    pipeline = train_model(X_train, y_train, transformer)
    metrics = evaluate_model(pipeline, X_test, y_test)
    save_artifacts(
        pipeline,
        metrics,
        {
            "raw": X.columns.tolist(),
            "engineered": engineered.columns.tolist(),
        },
        numeric_cols,
        categorical_cols,
        X_test.head(5),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
