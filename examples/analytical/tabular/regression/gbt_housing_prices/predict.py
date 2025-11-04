import argparse
from pathlib import Path

import joblib
import pandas as pd

from gbt import load as gbt_load

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
        default=DEFAULT_MODEL_DIR,
        help="Directory containing saved model artifacts (default: artifacts/model)",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="CSV file with rows to score (default: artifacts/sample_input.csv)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write predictions CSV (default: artifacts/predictions.csv)",
    )
    parser.add_argument(
        "--calibrator",
        default=DEFAULT_CALIBRATOR,
        help="Path to isotonic calibrator pickle (default: artifacts/model/isotonic_calibrator.pkl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)
    calibrator_path = Path(args.calibrator)

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
