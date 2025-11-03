import argparse
from pathlib import Path

import pandas as pd

from gbt import load as gbt_load


def parse_args():
    parser = argparse.ArgumentParser(description="Score upsell propensity using trained gbt model.")
    parser.add_argument(
        "--model-dir",
        default="artifacts/model",
        help="Directory containing saved model artifacts (default: artifacts/model)",
    )
    parser.add_argument(
        "--input",
        default="artifacts/sample_input.csv",
        help="CSV file with customer records to score (default: artifacts/sample_input.csv)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/predictions.csv",
        help="Where to write predictions CSV (default: artifacts/predictions.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = gbt_load(str(model_dir))

    features = pd.read_csv(input_path)
    if TARGET_COLUMN in features.columns:
        features = features.drop(columns=[TARGET_COLUMN])

    predictions = model.predict(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = features.copy()
    result["upsell_probability"] = predictions
    result.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


TARGET_COLUMN = "y"

if __name__ == "__main__":
    main()
