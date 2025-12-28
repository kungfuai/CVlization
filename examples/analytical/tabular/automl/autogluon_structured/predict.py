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
    parser.add_argument("--model-dir", default=None, help="Directory containing AutoGluon artifacts (default: bundled sample)")
    parser.add_argument("--input", default=None, help="CSV file with records to score (default: bundled sample)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to write predictions CSV")
    return parser.parse_args()


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
