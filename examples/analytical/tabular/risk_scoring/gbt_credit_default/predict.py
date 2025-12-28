import argparse
from pathlib import Path

import pandas as pd

from gbt import load as gbt_load
from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_MODEL_DIR = "artifacts/model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Score credit default risk using trained gbt model.")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory containing saved model artifacts (default: artifacts/model)",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="CSV file with borrower records to score (default: artifacts/sample_input.csv)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write predictions CSV (default: artifacts/predictions.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Defaults are local to example dir; user-provided paths resolve to cwd
    model_dir = Path(args.model_dir) if args.model_dir == DEFAULT_MODEL_DIR else Path(resolve_input_path(args.model_dir))
    input_path = Path(args.input) if args.input == DEFAULT_INPUT else Path(resolve_input_path(args.input))
    # Output always resolves to user's cwd
    output_path = Path(resolve_output_path(args.output))

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = gbt_load(str(model_dir))

    features = pd.read_csv(input_path)

    predictions = model.predict(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = features.copy()
    result["default_probability"] = predictions
    result.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
