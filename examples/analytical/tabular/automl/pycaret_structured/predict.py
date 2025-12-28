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
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to saved PyCaret model")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV with records to score")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to write predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Defaults are local to example dir; user-provided paths resolve to cwd
    model_path = args.model if args.model == DEFAULT_MODEL_PATH else resolve_input_path(args.model)
    input_path = args.input if args.input == DEFAULT_INPUT else resolve_input_path(args.input)
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
