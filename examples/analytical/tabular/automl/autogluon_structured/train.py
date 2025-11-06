import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "telco_churn.csv"

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
LEADERBOARD_PATH = ARTIFACTS_DIR / "leaderboard.csv"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"

TARGET = "Churn"
ID_COL = "customerID"
TEST_SIZE = 0.2
RANDOM_STATE = 1337
TIME_LIMIT = 600  # seconds


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
    return X, y


def train_autogluon(X_train: pd.DataFrame, y_train: pd.Series) -> TabularPredictor:
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    predictor = TabularPredictor(label=TARGET, problem_type="binary", eval_metric="roc_auc", path=str(MODEL_DIR))
    predictor.fit(
        train_data=train_df,
        time_limit=TIME_LIMIT,
        presets="best_quality",
        verbosity=2,
    )
    return predictor


def evaluate(predictor: TabularPredictor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    leaderboard = predictor.leaderboard(test_df, silent=True)
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)

    y_proba = predictor.predict_proba(X_test)[1]
    y_pred = predictor.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc": float(auc),
        "accuracy": float(report["accuracy"]),
        "precision": report.get("1", {}).get("precision", float("nan")),
        "recall": report.get("1", {}).get("recall", float("nan")),
        "f1": report.get("1", {}).get("f1-score", float("nan")),
        "best_model": predictor.get_model_best(),
    }
    return metrics


def save_artifacts(predictor: TabularPredictor, metrics: Dict[str, float], X_test: pd.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "autogluon_version": predictor.get_info()["version"],
        "time_limit": TIME_LIMIT,
        "label": TARGET,
        "best_model": metrics["best_model"],
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    sample_inputs = X_test.head(5)
    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)

    preds = predictor.predict_proba(X_test)
    pred_df = X_test.copy()
    pred_df["prob_no"] = preds[0]
    pred_df["prob_yes"] = preds[1]
    pred_df.to_csv(PREDICTIONS_PATH, index=False)


def main() -> None:
    download_dataset()
    df = load_dataset()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    predictor = train_autogluon(X_train, y_train)
    metrics = evaluate(predictor, X_test, y_test)
    save_artifacts(predictor, metrics, X_test)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
