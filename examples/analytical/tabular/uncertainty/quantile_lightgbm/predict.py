import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from cvlization.paths import resolve_input_path, resolve_output_path

MODEL_DIR = Path("artifacts/models")
SCALER_MEAN_PATH = MODEL_DIR / "scaler_mean.npy"
SCALER_SCALE_PATH = MODEL_DIR / "scaler_scale.npy"
MODEL_FILES = {
    0.1: "lgb_quantile_10.txt",
    0.5: "lgb_quantile_50.txt",
    0.9: "lgb_quantile_90.txt",
}
QUANTILES = sorted(MODEL_FILES.keys())


def load_models():
    scaler_mean = np.load(SCALER_MEAN_PATH)
    scaler_scale = np.load(SCALER_SCALE_PATH)
    models = {}
    for alpha, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        models[alpha] = lgb.Booster(model_file=str(model_path))
    return scaler_mean, scaler_scale, models


def predict(df: pd.DataFrame) -> pd.DataFrame:
    scaler_mean, scaler_scale, models = load_models()
    X_scaled = (df.values - scaler_mean) / scaler_scale

    result = df.copy()
    for alpha in QUANTILES:
        preds = models[alpha].predict(X_scaled)
        result[f"pred_{int(alpha * 100)}"] = preds
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict with LightGBM quantile models")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    # User-provided paths resolve to cwd
    df = pd.read_csv(resolve_input_path(args.input))
    result = predict(df)
    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
