import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
BASE_MODEL_PATH = MODEL_DIR / "base_estimator.pkl"
MAPIE_MODEL_PATH = MODEL_DIR / "mapie_regressor.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"

ALPHAS = [0.1]  # 90% prediction interval


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_california_housing(as_frame=True)
    features = dataset.data
    target = dataset.target.rename("median_house_value")
    return features, target


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingRegressor:
    estimator = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.1,
        max_leaf_nodes=64,
    )
    estimator.fit(X_train, y_train)
    return estimator


def calibrate_conformal(
    estimator: HistGradientBoostingRegressor,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
) -> MapieRegressor:
    mapie = MapieRegressor(
        estimator=estimator,
        method="plus",
        cv="prefit",
        agg_function="median",
    )
    mapie.fit(X_calib, y_calib, alpha=ALPHAS)
    return mapie


def evaluate(
    mapie: MapieRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, pd.DataFrame]:
    y_pred, y_interval = mapie.predict(X_test, alpha=ALPHAS)

    if y_interval.ndim != 3:
        raise ValueError(f"Unexpected interval shape: {y_interval.shape}")
    if y_interval.shape[1] == 2:
        lower = y_interval[:, 0, 0]
        upper = y_interval[:, 1, 0]
    elif y_interval.shape[2] == 2:
        lower = y_interval[:, 0, 0]
        upper = y_interval[:, 0, 1]
    else:
        raise ValueError(f"Unable to parse interval shape: {y_interval.shape}")

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    avg_width = float(np.mean(upper - lower))

    metrics = {
        "mae": float(mae),
        "r2": float(r2),
        "interval_coverage": coverage,
        "target_coverage": 1.0 - ALPHAS[0],
        "interval_width": avg_width,
    }

    predictions = pd.DataFrame(
        {
            "prediction": y_pred,
            "lower_pi": lower,
            "upper_pi": upper,
        }
    )
    return metrics, predictions


def save_artifacts(
    estimator: HistGradientBoostingRegressor,
    mapie: MapieRegressor,
    metrics: dict,
    X_test: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    joblib.dump(estimator, BASE_MODEL_PATH)
    joblib.dump(mapie, MAPIE_MODEL_PATH)
    print(f"Models saved to {MODEL_DIR}")

    config = {
        "alphas": ALPHAS,
        "method": "plus",
        "base_estimator": "HistGradientBoostingRegressor",
    }
    with CONFIG_PATH.open("w") as f:
        json.dump(config, f, indent=2)

    sample_inputs = X_test.head(5).copy()
    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)
    print(f"Sample inputs saved to {SAMPLE_INPUT_PATH}")

    sample_preds = sample_inputs.copy()
    sample_preds["prediction"] = predictions["prediction"].head(5).values
    sample_preds["lower_pi"] = predictions["lower_pi"].head(5).values
    sample_preds["upper_pi"] = predictions["upper_pi"].head(5).values
    sample_preds.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Sample predictions saved to {PREDICTIONS_PATH}")


def main() -> None:
    X, y = load_dataset()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    estimator = train_models(X_train, y_train)
    mapie = calibrate_conformal(estimator, X_calib, y_calib)
    metrics, predictions = evaluate(mapie, X_test, y_test)
    save_artifacts(estimator, mapie, metrics, X_test, predictions)


if __name__ == "__main__":
    main()
