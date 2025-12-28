import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from cvlization.paths import resolve_input_path, resolve_output_path

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "models/uplift_models.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def load_metadata() -> Dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}. Run train.py first."
        )
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required feature columns: {missing}")
    return df[features].copy()


def predict_uplift(models: Dict[str, object], treatments: List[str], X: pd.DataFrame) -> pd.DataFrame:
    predictions = {}
    available = set(models.keys())
    missing = [t for t in treatments if t not in available]
    if missing:
        raise ValueError(f"Missing uplift models for treatments: {missing}")
    for treat in treatments:
        predictions[f"pred_uplift_{treat}"] = np.asarray(models[treat].predict(X)).reshape(-1)
    stacked = np.column_stack([predictions[f"pred_uplift_{treat}"] for treat in treatments])
    best_idx = stacked.argmax(axis=1)
    recommendations = [treatments[idx] for idx in best_idx]
    predictions["recommended_treatment"] = recommendations
    return pd.DataFrame(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate uplift predictions for campaign optimization."
    )
    parser.add_argument("--input", required=True, help="Path to CSV with feature columns.")
    parser.add_argument("--output", required=True, help="Where to write the predictions CSV.")
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Include original feature columns in the output CSV.",
    )
    args = parser.parse_args()

    metadata = load_metadata()
    features: List[str] = metadata["feature_columns"]
    treatments: List[str] = metadata["treatments"][1:]  # exclude control

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained models not found at {MODEL_PATH}. Run train.py first.")
    models = joblib.load(MODEL_PATH)

    # User-provided paths resolve to cwd
    df = pd.read_csv(resolve_input_path(args.input))
    X = ensure_features(df, features)

    uplift_df = predict_uplift(models, treatments, X)

    if args.include_features:
        output_df = pd.concat([df.reset_index(drop=True), uplift_df], axis=1)
    else:
        output_df = uplift_df

    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
