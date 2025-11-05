import json
import os
from pathlib import Path
from typing import Dict, Tuple

import arviz as az
import joblib
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = int(os.environ.get("PYMC_RANDOM_SEED", "42"))
POSTERIOR_SAMPLES_FILE = "posterior_samples.npz"

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
SAMPLE_PREDICTIONS_PATH = ARTIFACTS_DIR / "sample_predictions.csv"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
POSTERIOR_PATH = MODEL_DIR / POSTERIOR_SAMPLES_FILE
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.joblib"
IDATA_PATH = MODEL_DIR / "posterior_inference.nc"


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    target = df["MedHouseVal"].rename("median_house_value")
    features = df.drop(columns=["MedHouseVal"])
    features.columns = [col.lower() for col in features.columns]
    return features, target


def standardize_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, feature_scaler, target_scaler


def scale_with_existing(
    scaler_X: StandardScaler, scaler_y: StandardScaler, X: pd.DataFrame, y: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.to_numpy().reshape(-1, 1)).ravel()
    return X_scaled, y_scaled


def save_metrics(payload: Dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


def save_metadata(payload: Dict) -> None:
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Metadata saved to {METADATA_PATH}")


def save_sample_artifacts(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mean_preds: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    median_preds: np.ndarray,
) -> None:
    head_inputs = X_test.head(5).copy()
    head_inputs["true_value"] = y_test.head(5).to_numpy()
    head_inputs.to_csv(SAMPLE_INPUT_PATH, index=False)
    print(f"Sample inputs saved to {SAMPLE_INPUT_PATH}")

    sample_outputs = pd.DataFrame(
        {
            "true_value": y_test.head(5).to_numpy(),
            "pred_mean": mean_preds[:5],
            "pred_median": median_preds[:5],
            "pred_lower_p05": lower[:5],
            "pred_upper_p95": upper[:5],
        }
    )
    sample_outputs.to_csv(SAMPLE_PREDICTIONS_PATH, index=False)
    print(f"Sample predictions saved to {SAMPLE_PREDICTIONS_PATH}")


def main() -> None:
    X, y = load_dataset()
    feature_names = X.columns.tolist()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    X_train_scaled, y_train_scaled, feature_scaler, target_scaler = standardize_data(
        X_train_df, y_train
    )
    X_test_scaled, y_test_scaled = scale_with_existing(
        feature_scaler, target_scaler, X_test_df, y_test
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    print(f"Saved scalers to {MODEL_DIR}")

    with pm.Model() as model:
        X_data = pm.Data("X_data", X_train_scaled)
        y_data = pm.Data("y_data", y_train_scaled)

        intercept = pm.Normal("intercept", mu=0.0, sigma=5.0)
        coeffs = pm.Normal("coeffs", mu=0.0, sigma=1.5, shape=X_train_scaled.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = intercept + pm.math.dot(X_data, coeffs)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data)

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )

    # Persist inference results for inspection/debugging.
    idata.to_netcdf(IDATA_PATH)
    print(f"Inference data written to {IDATA_PATH}")

    posterior_samples = az.extract(
        idata, var_names=["coeffs", "intercept", "sigma"], combined=True
    )
    coeffs_samples = np.asarray(posterior_samples["coeffs"])
    if coeffs_samples.ndim == 1:
        coeffs_samples = coeffs_samples[None, :]
    if coeffs_samples.shape[1] != X_train_scaled.shape[1] and coeffs_samples.shape[0] == X_train_scaled.shape[1]:
        coeffs_samples = coeffs_samples.T
    intercept_samples = np.asarray(posterior_samples["intercept"])
    sigma_samples = np.asarray(posterior_samples["sigma"])

    np.savez(
        POSTERIOR_PATH,
        coeffs=coeffs_samples,
        intercept=intercept_samples,
        sigma=sigma_samples,
    )
    print(f"Posterior samples saved to {POSTERIOR_PATH}")

    rng = np.random.default_rng(RANDOM_SEED)
    mu_samples_scaled = coeffs_samples @ X_test_scaled.T + intercept_samples[:, None]
    noise = rng.normal(
        loc=0.0,
        scale=sigma_samples[:, None],
        size=mu_samples_scaled.shape,
    )
    posterior_predictive_scaled = mu_samples_scaled + noise
    predictive_samples = (
        posterior_predictive_scaled * target_scaler.scale_[0]
        + target_scaler.mean_[0]
    )
    y_test_array = y_test.to_numpy()

    predictive_mean = predictive_samples.mean(axis=0)
    predictive_median = np.median(predictive_samples, axis=0)
    lower = np.quantile(predictive_samples, 0.05, axis=0)
    upper = np.quantile(predictive_samples, 0.95, axis=0)

    rmse = float(np.sqrt(mean_squared_error(y_test_array, predictive_mean)))
    mae = float(mean_absolute_error(y_test_array, predictive_mean))
    r2 = float(r2_score(y_test_array, predictive_mean))
    coverage = float(((y_test_array >= lower) & (y_test_array <= upper)).mean())
    avg_interval_width = float(np.mean(upper - lower))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "interval_coverage_p90": coverage,
        "interval_avg_width": avg_interval_width,
    }

    save_metrics(metrics)

    save_metadata(
        {
            "dataset": "california_housing",
            "feature_names": feature_names,
            "num_draws": int(coeffs_samples.shape[0]),
            "train_size": int(len(X_train_df)),
            "test_size": int(len(X_test_df)),
            "random_seed": RANDOM_SEED,
            "posterior_samples_file": POSTERIOR_SAMPLES_FILE,
        }
    )

    save_sample_artifacts(
        X_test_df.reset_index(drop=True),
        y_test.reset_index(drop=True),
        predictive_mean,
        lower,
        upper,
        predictive_median,
    )


if __name__ == "__main__":
    main()
