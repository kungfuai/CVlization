import argparse
from pathlib import Path

import pandas as pd
from pycaret.classification import load_model, predict_model

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

    model = load_model(args.model)
    df = pd.read_csv(args.input)
    preds = predict_model(model, data=df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
