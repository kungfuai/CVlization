import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests
import pycaret
from pycaret.classification import pull, save_model, setup, compare_models, predict_model
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "telco_churn.csv"

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_PATH = MODEL_DIR / "pycaret_best_model"
LEADERBOARD_PATH = ARTIFACTS_DIR / "leaderboard.csv"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"

TARGET = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 1337
TIME_LIMIT = 600


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
    X = df.drop(columns=[TARGET, "customerID"], errors="ignore")
    return X, y


def train_pycaret(X_train: pd.DataFrame, y_train: pd.Series):
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    setup(
        data=train_df,
        target=TARGET,
        session_id=RANDOM_STATE,
        use_gpu=False,
        log_experiment=False,
        fold=5,
        verbose=False,
    )

    best_model = compare_models(budget_time=TIME_LIMIT, sort="AUC")
    leaderboard = pull()

    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(best_model, str(MODEL_PATH))
    return best_model


def evaluate(best_model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    preds = predict_model(best_model, data=test_df)
    if "Score" in preds.columns:
        score_series = preds["Score"]
    elif "prediction_score" in preds.columns:
        score_series = preds["prediction_score"]
    elif "prediction_probability_1" in preds.columns:
        score_series = preds["prediction_probability_1"]
    else:
        score_series = pd.Series(0.5, index=preds.index)

    score_series = pd.to_numeric(score_series, errors="coerce").fillna(0.5)

    label_series = preds.get("Label")
    if label_series is None:
        label_series = preds.get("prediction_label")
    if label_series is None:
        label_series = pd.Series(0, index=preds.index)
    label_series = label_series.astype(str).str.lower()
    label_series = label_series.map({"1": 1, "true": 1, "yes": 1}).fillna(0).astype(int)

    y_prob = score_series.where(label_series == 1, 1 - score_series).values
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc": float(auc),
        "accuracy": float(report["accuracy"]),
        "precision": report.get("1", {}).get("precision", float("nan")),
        "recall": report.get("1", {}).get("recall", float("nan")),
        "f1": report.get("1", {}).get("f1-score", float("nan")),
        "best_model": str(best_model),
    }
    preds.to_csv(PREDICTIONS_PATH, index=False)
    return metrics


def save_artifacts(metrics: Dict[str, float], X_test: pd.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "pycaret_version": pycaret.__version__,
        "time_limit": TIME_LIMIT,
        "label": TARGET,
        "best_model": metrics["best_model"],
    }
    CONFIG_PATH = MODEL_DIR / "config.json"
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    X_test.head(5).to_csv(SAMPLE_INPUT_PATH, index=False)


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

    best_model = train_pycaret(X_train, y_train)
    metrics = evaluate(best_model, X_test, y_test)
    save_artifacts(metrics, X_test)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
