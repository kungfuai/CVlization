import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from cvlization.paths import resolve_input_path, resolve_output_path

MODEL_DIR = Path("artifacts/models")
POSTERIOR_SAMPLES_FILE = MODEL_DIR / "posterior_samples.npz"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.joblib"
METADATA_PATH = Path("artifacts/metadata.json")


def load_metadata() -> Dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}. Did you run train.py?"
        )
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_columns(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required feature columns: {missing}. "
            f"Expected columns: {expected_cols}"
        )
    return df[expected_cols].copy()


def load_posterior_samples() -> Dict[str, np.ndarray]:
    if not POSTERIOR_SAMPLES_FILE.exists():
        raise FileNotFoundError(
            f"Posterior samples not found at {POSTERIOR_SAMPLES_FILE}. Run train.py first."
        )
    data = np.load(POSTERIOR_SAMPLES_FILE)
    return {key: data[key] for key in data.files}


def compute_predictions(
    X: np.ndarray,
    coeffs_samples: np.ndarray,
    intercept_samples: np.ndarray,
    sigma_samples: np.ndarray,
    target_scaler,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    mu_samples = coeffs_samples @ X.T + intercept_samples[:, None]
    noise = rng.normal(
        loc=0.0,
        scale=sigma_samples[:, None],
        size=mu_samples.shape,
    )
    predictive_samples_scaled = mu_samples + noise
    predictive_samples = predictive_samples_scaled * target_scaler.scale_[0] + target_scaler.mean_[0]

    summary = {
        "mean": predictive_samples.mean(axis=0),
        "median": np.median(predictive_samples, axis=0),
        "std": predictive_samples.std(axis=0),
        "p05": np.quantile(predictive_samples, 0.05, axis=0),
        "p95": np.quantile(predictive_samples, 0.95, axis=0),
    }

    return {"samples": predictive_samples, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Bayesian regression predictions with posterior uncertainty."
    )
    parser.add_argument("--input", required=True, help="Path to CSV with feature columns.")
    parser.add_argument("--output", required=True, help="Where to save the predictions CSV.")
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Include original features alongside prediction summary statistics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for posterior predictive sampling.",
    )
    args = parser.parse_args()

    df = pd.read_csv(resolve_input_path(args.input))
    metadata = load_metadata()
    feature_names: List[str] = metadata["feature_names"]

    features = ensure_columns(df, feature_names)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    X_scaled = feature_scaler.transform(features)

    posterior = load_posterior_samples()
    coeffs_samples = posterior["coeffs"]
    intercept_samples = posterior["intercept"]
    sigma_samples = posterior["sigma"]

    rng = np.random.default_rng(args.seed)
    predictive = compute_predictions(
        X_scaled,
        coeffs_samples,
        intercept_samples,
        sigma_samples,
        target_scaler,
        rng,
    )

    summary = predictive["summary"]
    results = pd.DataFrame(
        {
            "pred_mean": summary["mean"],
            "pred_median": summary["median"],
            "pred_std": summary["std"],
            "pred_lower_p05": summary["p05"],
            "pred_upper_p95": summary["p95"],
        }
    )

    if args.include_features:
        output_df = pd.concat([df.reset_index(drop=True), results], axis=1)
    else:
        output_df = results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
