import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_DIR = ARTIFACTS_DIR / "models"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"

QUANTILES = [0.1, 0.5, 0.9]
MODEL_FILES = {0.1: "lgb_quantile_10.txt", 0.5: "lgb_quantile_50.txt", 0.9: "lgb_quantile_90.txt"}


def load_dataset():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="median_house_value")
    return X, y


def train_quantile_models(X_train, y_train, X_valid, y_valid):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    models = {}
    for quantile in QUANTILES:
        params = {
            "objective": "quantile",
            "alpha": quantile,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
        }

        train_set = lgb.Dataset(X_train_scaled, label=y_train)
        valid_set = lgb.Dataset(X_valid_scaled, label=y_valid, reference=train_set)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        models[quantile] = (model, scaler)
    return models


def evaluate(models, X_test, y_test):
    scaler = models[0.5][1]
    X_test_scaled = scaler.transform(X_test)

    preds = {}
    for quantile in QUANTILES:
        model = models[quantile][0]
        preds[quantile] = model.predict(X_test_scaled)

    median_pred = preds[0.5]
    mae = mean_absolute_error(y_test, median_pred)

    lower, upper = preds[0.1], preds[0.9]
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    width = np.mean(upper - lower)

    metrics = {
        "mae": float(mae),
        "interval_coverage": float(coverage),
        "interval_width": float(width),
    }

    return metrics, preds


def save_artifacts(models, metrics, X_test, preds):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    scaler = models[0.5][1]
    np.save(MODEL_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(MODEL_DIR / "scaler_scale.npy", scaler.scale_)

    for quantile, (model, _) in models.items():
        model_path = MODEL_DIR / MODEL_FILES[quantile]
        model.save_model(str(model_path))
    print(f"Models saved under {MODEL_DIR}")

    sample_inputs = X_test.head(5)
    sample_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)
    print(f"Sample inputs saved to {SAMPLE_INPUT_PATH}")

    results = sample_inputs.copy()
    results["pred_10"] = preds[0.1][:5]
    results["pred_50"] = preds[0.5][:5]
    results["pred_90"] = preds[0.9][:5]
    results.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Sample predictions saved to {PREDICTIONS_PATH}")


def main():
    X, y = load_dataset()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    models = train_quantile_models(X_train, y_train, X_valid, y_valid)
    metrics, preds = evaluate(models, X_test, y_test)
    save_artifacts(models, metrics, X_test, preds)


if __name__ == "__main__":
    main()
