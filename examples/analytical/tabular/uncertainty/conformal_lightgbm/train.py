import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42
DATA_CACHE = Path(
    os.environ.get("CVL_DATA_CACHE", Path.home() / ".cache" / "cvlization" / "data")
)
DATASET_DIR = DATA_CACHE / "openml_credit_g"
DATASET_CSV = DATASET_DIR / "credit_g.csv"

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
SAMPLE_PREDICTIONS_PATH = ARTIFACTS_DIR / "sample_predictions.csv"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.json"
CONFORMAL_METADATA_PATH = MODEL_DIR / "conformal_metadata.json"
BASE_MODEL_PATH = MODEL_DIR / "lightgbm_classifier.joblib"
SIGMOID_CALIBRATOR_PATH = MODEL_DIR / "sigmoid_calibrator.joblib"
ISOTONIC_CALIBRATOR_PATH = MODEL_DIR / "isotonic_calibrator.joblib"


def fetch_dataset() -> pd.DataFrame:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_CSV.exists():
        return pd.read_csv(DATASET_CSV)

    print("Downloading German Credit dataset from OpenML (id=31)...")
    dataset = fetch_openml(data_id=31, as_frame=True)
    frame = dataset.frame
    frame.to_csv(DATASET_CSV, index=False)
    print(f"Cached dataset at {DATASET_CSV}")
    return frame


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    X = df.drop(columns=["class"]).copy()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    category_map: Dict[str, List[str]] = {}

    for col in categorical_cols:
        col_series = (
            X[col]
            .astype("string")
            .fillna("Unknown")
        )
        categories = sorted(col_series.unique().tolist())
        if "Unknown" not in categories:
            categories.append("Unknown")
        X[col] = col_series.astype(pd.CategoricalDtype(categories=categories))
        category_map[col] = categories

    numeric_cols = X.select_dtypes(include=["int", "float", "bool"]).columns.tolist()
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(0)

    return X, category_map, numeric_cols


def apply_category_map(df: pd.DataFrame, category_map: Dict[str, List[str]]) -> pd.DataFrame:
    result = df.copy()
    for col, categories in category_map.items():
        if col not in result.columns:
            continue
        result[col] = (
            result[col]
            .astype("string")
            .fillna("Unknown")
            .apply(lambda val: val if val in categories else "Unknown")
            .astype(pd.CategoricalDtype(categories=categories))
        )
    return result


def compute_conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(sorted_scores[rank])


def main() -> None:
    raw_df = fetch_dataset()
    X, category_map, numeric_cols = prepare_features(raw_df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_df["class"].astype("string"))
    class_names = label_encoder.classes_.tolist()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    X_train = apply_category_map(X_train.copy(), category_map)
    X_cal = apply_category_map(X_cal.copy(), category_map)
    X_test = apply_category_map(X_test.copy(), category_map)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        subsample_for_bin=200000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print("Training LightGBM classifier...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_cal, y_cal)],
        eval_metric="logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
        categorical_feature=list(category_map.keys()),
    )

    sigmoid_calibrator = CalibratedClassifierCV(
        estimator=model, method="sigmoid", cv="prefit"
    )
    sigmoid_calibrator.fit(X_cal, y_cal)

    isotonic_calibrator = CalibratedClassifierCV(
        estimator=model, method="isotonic", cv="prefit"
    )
    isotonic_calibrator.fit(X_cal, y_cal)

    test_probs_base = model.predict_proba(X_test)
    test_probs_sigmoid = sigmoid_calibrator.predict_proba(X_test)
    test_probs_isotonic = isotonic_calibrator.predict_proba(X_test)

    base_metrics = {
        "accuracy": float(accuracy_score(y_test, np.argmax(test_probs_base, axis=1))),
        "roc_auc": float(roc_auc_score(y_test, test_probs_base[:, 1])),
        "log_loss": float(log_loss(y_test, test_probs_base)),
        "brier": float(brier_score_loss(y_test, test_probs_base[:, 1])),
    }

    sigmoid_metrics = {
        "accuracy": float(accuracy_score(y_test, np.argmax(test_probs_sigmoid, axis=1))),
        "roc_auc": float(roc_auc_score(y_test, test_probs_sigmoid[:, 1])),
        "log_loss": float(log_loss(y_test, test_probs_sigmoid)),
        "brier": float(brier_score_loss(y_test, test_probs_sigmoid[:, 1])),
    }

    isotonic_metrics = {
        "accuracy": float(accuracy_score(y_test, np.argmax(test_probs_isotonic, axis=1))),
        "roc_auc": float(roc_auc_score(y_test, test_probs_isotonic[:, 1])),
        "log_loss": float(log_loss(y_test, test_probs_isotonic)),
        "brier": float(brier_score_loss(y_test, test_probs_isotonic[:, 1])),
    }

    alpha = float(os.environ.get("CONFORMAL_ALPHA", 0.1))
    cal_probs_sigmoid = sigmoid_calibrator.predict_proba(X_cal)
    nonconformity_scores = 1.0 - cal_probs_sigmoid[np.arange(len(y_cal)), y_cal]
    qhat = compute_conformal_threshold(nonconformity_scores, alpha=alpha)
    threshold_probability = 1.0 - qhat

    prediction_sets = test_probs_sigmoid >= threshold_probability
    coverage = float(
        prediction_sets[np.arange(len(y_test)), y_test].mean()
    )
    avg_set_size = float(prediction_sets.sum(axis=1).mean())

    metrics = {
        "base_model": base_metrics,
        "sigmoid_calibrated": sigmoid_metrics,
        "isotonic_calibrated": isotonic_metrics,
        "conformal": {
            "alpha": alpha,
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "threshold_probability": threshold_probability,
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {METRICS_PATH}")

    joblib.dump(model, BASE_MODEL_PATH)
    joblib.dump(sigmoid_calibrator, SIGMOID_CALIBRATOR_PATH)
    joblib.dump(isotonic_calibrator, ISOTONIC_CALIBRATOR_PATH)
    print(f"Models saved under {MODEL_DIR}")

    preprocessor_info = {
        "feature_columns": X.columns.tolist(),
        "categorical_columns": list(category_map.keys()),
        "numeric_columns": numeric_cols,
        "category_map": category_map,
        "classes": class_names,
        "label_mapping": {
            class_name: int(index) for index, class_name in enumerate(class_names)
        },
    }
    with PREPROCESSOR_PATH.open("w", encoding="utf-8") as f:
        json.dump(preprocessor_info, f, indent=2)

    conformal_metadata = {
        "alpha_default": alpha,
        "threshold_probability": threshold_probability,
        "qhat": qhat,
        "calibration_size": len(nonconformity_scores),
        "nonconformity_scores": nonconformity_scores.tolist(),
        "classes": class_names,
    }
    with CONFORMAL_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(conformal_metadata, f, indent=2)

    sample_idx = np.arange(min(5, len(X_test)))
    sample_inputs = X_test.iloc[sample_idx].copy()
    for col in category_map:
        if col in sample_inputs.columns:
            sample_inputs[col] = sample_inputs[col].astype(str)
    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)

    sample_outputs = sample_inputs.copy()
    sample_outputs["true_label"] = [class_names[idx] for idx in y_test[sample_idx]]
    sample_outputs["prob_base_positive"] = test_probs_base[sample_idx, 1]
    sample_outputs["prob_sigmoid_positive"] = test_probs_sigmoid[sample_idx, 1]
    sample_outputs["prob_isotonic_positive"] = test_probs_isotonic[sample_idx, 1]

    sets_as_strings = []
    for row in prediction_sets[sample_idx]:
        included = [class_names[i] for i, flag in enumerate(row) if flag]
        sets_as_strings.append(";".join(included))
    sample_outputs["conformal_set"] = sets_as_strings
    sample_outputs.to_csv(SAMPLE_PREDICTIONS_PATH, index=False)
    print(f"Sample predictions written to {SAMPLE_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
