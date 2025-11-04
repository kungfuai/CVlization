import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

DEFAULT_MODEL_DIR = "artifacts/model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyOD fraud detector on tabular transactions."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory containing detector & scaler artifacts (default: artifacts/model)",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="CSV file with records to score (default: artifacts/sample_input.csv)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write predictions CSV (default: artifacts/predictions.csv)",
    )
    return parser.parse_args()


def load_artifacts(model_dir: Path) -> tuple[IForest, StandardScaler]:
    detector_path = model_dir / "iforest_detector.pkl"
    scaler_path = model_dir / "scaler.pkl"

    if not detector_path.exists():
        raise FileNotFoundError(f"Detector not found: {detector_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    detector: IForest = joblib.load(detector_path)
    scaler: StandardScaler = joblib.load(scaler_path)
    print(f"Loaded detector from {detector_path}")
    return detector, scaler


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)

    detector, scaler = load_artifacts(model_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    features = pd.read_csv(input_path)
    feature_cols = features.columns.tolist()

    features_scaled = scaler.transform(features)
    scores = detector.decision_function(features_scaled)
    preds = detector.predict(features_scaled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = features.copy()
    result["anomaly_score"] = scores
    result["is_anomaly_pred"] = preds.astype(int)
    result.to_csv(output_path, index=False)
    print(
        f"Predictions written to {output_path} (columns: {', '.join(feature_cols)} + scores)"
    )


if __name__ == "__main__":
    main()
