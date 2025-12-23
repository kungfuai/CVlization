import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from cvlization.paths import resolve_input_path, resolve_output_path

MODEL_DIR = Path("artifacts/models")
MODEL_FILES = {
    0.1: "catboost_quantile_10.cbm",
    0.5: "catboost_quantile_50.cbm",
    0.9: "catboost_quantile_90.cbm",
}
QUANTILES = sorted(MODEL_FILES.keys())


def load_models():
    models = {}
    for alpha, filename in MODEL_FILES.items():
        model = CatBoostRegressor()
        model.load_model(MODEL_DIR / filename)
        models[alpha] = model
    return models


def predict(df: pd.DataFrame) -> pd.DataFrame:
    models = load_models()
    result = df.copy()
    for alpha in QUANTILES:
        preds = models[alpha].predict(df)
        result[f"pred_{int(alpha * 100)}"] = preds
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict with CatBoost quantile models")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Where to write predictions CSV")
    args = parser.parse_args()

    df = pd.read_csv(resolve_input_path(args.input))
    result = predict(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
