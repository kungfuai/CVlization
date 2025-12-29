import argparse
from pathlib import Path

import pandas as pd

from gbt import load as gbt_load
from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_MODEL_DIR = "artifacts/model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Score upsell propensity using trained gbt model.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing saved model artifacts (default: bundled sample)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="CSV file with customer records to score (default: bundled sample)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write predictions CSV (default: artifacts/predictions.csv)",
    )
    return parser.parse_args()


def main():
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
