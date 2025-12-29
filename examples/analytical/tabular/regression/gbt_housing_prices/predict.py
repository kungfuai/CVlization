import argparse
from pathlib import Path

import joblib
import pandas as pd

from gbt import load as gbt_load
from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_MODEL_DIR = "artifacts/model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"
DEFAULT_CALIBRATOR = "artifacts/model/isotonic_calibrator.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score housing listings with calibrated gbt regression model."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing saved model artifacts (default: bundled sample)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="CSV file with rows to score (default: bundled sample)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write predictions CSV (default: artifacts/predictions.csv)",
    )
    parser.add_argument(
        "--calibrator",
        default=None,
        help="Path to isotonic calibrator pickle (default: bundled sample)",
    )
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
    if args.calibrator is None:
        calibrator_path = Path(DEFAULT_CALIBRATOR)
    else:
        calibrator_path = Path(resolve_input_path(args.calibrator))
    # Output always resolves to user's cwd
    output_path = Path(resolve_output_path(args.output))

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = gbt_load(str(model_dir))

    calibrator = None
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
        print(f"Loaded calibrator from {calibrator_path}")
    else:
        print("Calibration file not found; using raw predictions")

    features = pd.read_csv(input_path)
    raw_preds = model.predict(features)
    if calibrator is not None:
        calibrated_preds = calibrator.predict(raw_preds)
    else:
        calibrated_preds = raw_preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = features.copy()
    result["prediction_raw"] = raw_preds
    result["prediction_calibrated"] = calibrated_preds
    result.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
