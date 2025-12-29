import argparse
from pathlib import Path

import pandas as pd
from pycaret.classification import load_model, predict_model

from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_MODEL_PATH = "artifacts/model/pycaret_best_model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyCaret predictor on new data.")
    parser.add_argument("--model", default=None, help="Path to saved PyCaret model (default: bundled sample)")
    parser.add_argument("--input", default=None, help="CSV with records to score (default: bundled sample)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to write predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve paths: None means use bundled sample, otherwise resolve to user's cwd
    if args.model is None:
        model_path = DEFAULT_MODEL_PATH
        print(f"No --model provided, using bundled sample: {model_path}")
    else:
        model_path = resolve_input_path(args.model)
    if args.input is None:
        input_path = DEFAULT_INPUT
        print(f"No --input provided, using bundled sample: {input_path}")
    else:
        input_path = resolve_input_path(args.input)
    # Output always resolves to user's cwd
    output_path = Path(resolve_output_path(args.output))

    model = load_model(model_path)
    df = pd.read_csv(input_path)
    preds = predict_model(model, data=df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
