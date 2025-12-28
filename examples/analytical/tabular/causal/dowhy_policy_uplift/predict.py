import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_MODEL_DIR = "artifacts/model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate policy uplift using fitted DoWhy / DRLearner models."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing saved models (default: bundled sample)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="CSV file with cohort features to score (default: bundled sample)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write predictions CSV (default: artifacts/predictions.csv)",
    )
    return parser.parse_args()


def ensure_encoded(
    df: pd.DataFrame,
    feature_cols: List[str],
    original_categoricals: List[str],
    dummy_cols: List[str],
) -> pd.DataFrame:
    if all(col in df.columns for col in feature_cols):
        return df[feature_cols]

    encoded = df.copy()
    if original_categoricals:
        encoded = pd.get_dummies(encoded, columns=original_categoricals, drop_first=True)

    for col in dummy_cols:
        if col not in encoded.columns:
            encoded[col] = 0.0

    return encoded.reindex(columns=feature_cols, fill_value=0.0)


def main() -> None:
    args = parse_args()
    # Resolve paths: None means use bundled sample, otherwise resolve to user's cwd
    if args.model_dir is None:
        model_dir = Path(DEFAULT_MODEL_DIR)
        print(f"No --model-dir provided, using bundled sample: {model_dir}")
    else:
        model_dir = Path(resolve_input_path(args.model_dir))
    if args.input is None:
        input_path = Path(DEFAULT_INPUT)
        print(f"No --input provided, using bundled sample: {input_path}")
    else:
        input_path = Path(resolve_input_path(args.input))
    # Output always resolves to user's cwd
    output_path = Path(resolve_output_path(args.output))

    config = json.loads((model_dir / "config.json").read_text())
    feature_cols: List[str] = config["feature_columns"]
    dummy_cols: List[str] = config.get("categorical_dummy_columns", [])
    original_cats: List[str] = config.get("original_categorical_columns", [])
    top_k_fraction = config.get("top_k_fraction", 0.2)

    dr_learner = joblib.load(model_dir / "dr_learner.pkl")
    propensity_model = joblib.load(model_dir / "propensity_model.pkl")

    raw_features = pd.read_csv(input_path)
    encoded = ensure_encoded(raw_features, feature_cols, original_cats, dummy_cols)

    uplift_scores = dr_learner.effect(encoded.values)
    treatment_probs = propensity_model.predict_proba(encoded)[:, 1]

    predictions = raw_features.copy()
    predictions["uplift_score"] = uplift_scores
    predictions["treatment_propensity"] = treatment_probs

    top_k = max(1, int(len(predictions) * top_k_fraction))
    predictions["uplift_rank"] = np.argsort(np.argsort(-predictions["uplift_score"])) + 1
    predictions["recommended_treatment"] = predictions["uplift_rank"] <= top_k

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(
        f"Predictions written to {output_path} with uplift ranking; top {top_k_fraction:.0%} flagged for treatment"
    )


if __name__ == "__main__":
    main()
