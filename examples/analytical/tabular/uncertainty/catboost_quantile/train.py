import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"

QUANTILES = [0.1, 0.5, 0.9]
MODEL_FILES = {0.1: "catboost_quantile_10.cbm", 0.5: "catboost_quantile_50.cbm", 0.9: "catboost_quantile_90.cbm"}


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="median_house_value")
    return X, y


def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[float, CatBoostRegressor]:
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    models: Dict[float, CatBoostRegressor] = {}
    for alpha in QUANTILES:
        model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha}",
            iterations=2000,
            depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_seed=42,
            od_type="Iter",
            od_wait=100,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        models[alpha] = model
    return models


def evaluate(models: Dict[float, CatBoostRegressor], X_test: pd.DataFrame, y_test: pd.Series):
    preds = {alpha: model.predict(X_test) for alpha, model in models.items()}
    median_pred = preds[0.5]
    mae = mean_absolute_error(y_test, median_pred)

    lower, upper = preds[0.1], preds[0.9]
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    width = float(np.mean(upper - lower))

    metrics = {
        "mae": float(mae),
        "interval_coverage": coverage,
        "interval_width": width,
    }
    return metrics, preds


def save_artifacts(models: Dict[float, CatBoostRegressor], metrics, X_test: pd.DataFrame, preds):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    for alpha, model in models.items():
        model.save_model(MODEL_DIR / MODEL_FILES[alpha])
    print(f"Models saved under {MODEL_DIR}")

    sample_inputs = X_test.head(5)
    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)
    print(f"Sample inputs saved to {SAMPLE_INPUT_PATH}")

    sample_outputs = sample_inputs.copy()
    sample_outputs["pred_10"] = preds[0.1][:5]
    sample_outputs["pred_50"] = preds[0.5][:5]
    sample_outputs["pred_90"] = preds[0.9][:5]
    sample_outputs.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Sample predictions saved to {PREDICTIONS_PATH}")


def main():
    X, y = load_dataset()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    models = train_models(X_train, y_train, X_valid, y_valid)
    metrics, preds = evaluate(models, X_test, y_test)
    save_artifacts(models, metrics, X_test, preds)


if __name__ == "__main__":
    main()
