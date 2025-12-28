import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from cvlization.paths import resolve_input_path, resolve_output_path

MODEL_PATH = Path("artifacts/models/econml_linear_dr.joblib")
METADATA_PATH = Path("artifacts/metadata.json")


def load_metadata() -> Dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file missing at {METADATA_PATH}; run train.py first.")
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required feature columns: {missing}")
    return df[feature_columns].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate heterogeneous treatment effect predictions using the trained EconML learner."
    )
    parser.add_argument("--input", required=True, help="Path to CSV containing feature columns.")
    parser.add_argument("--output", required=True, help="Destination CSV for CATE predictions.")
    parser.add_argument(
        "--policy-threshold",
        type=float,
        default=0.0,
        help="Recommend treatment when predicted CATE exceeds this threshold (default: 0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Significance level for two-sided confidence intervals (default=0.1).",
    )
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Include the input feature columns alongside predictions in the output CSV.",
    )
    args = parser.parse_args()

    metadata = load_metadata()
    feature_columns: List[str] = metadata["feature_columns"]

    # User-provided paths resolve to cwd
    df = pd.read_csv(resolve_input_path(args.input))
    features = ensure_features(df, feature_columns)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    learner = joblib.load(MODEL_PATH)

    cate = learner.effect(features)
    lower, upper = learner.effect_interval(features, alpha=args.alpha)
    recommend = (cate > args.policy_threshold).astype(int)

    summary = pd.DataFrame(
        {
            "predicted_tau": cate,
            "tau_lower": lower,
            "tau_upper": upper,
            "recommend_treatment": recommend,
        }
    )

    if args.include_features:
        output_df = pd.concat([df.reset_index(drop=True), summary], axis=1)
    else:
        output_df = summary

    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
