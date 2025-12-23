import argparse
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_MODEL_DIR = "artifacts/model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoGluon predictor on new records.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory containing AutoGluon artifacts")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV file with records to score")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to write predictions CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(resolve_input_path(args.model_dir))
    input_path = Path(resolve_input_path(args.input))
    output_path = Path(args.output)

    predictor = TabularPredictor.load(str(model_dir))

    df = pd.read_csv(input_path)
    preds = predictor.predict(df)
    probs = predictor.predict_proba(df)

    result = df.copy()
    result["predicted_label"] = preds
    if isinstance(probs, pd.DataFrame):
        result = pd.concat([result, probs.add_prefix("prob_")], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
