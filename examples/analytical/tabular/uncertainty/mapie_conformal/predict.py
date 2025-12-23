import argparse
from pathlib import Path

import joblib
import pandas as pd

from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_ALPHA = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conformal prediction intervals using a trained MAPIE regressor."
    )
    parser.add_argument(
        "--model",
        default="artifacts/model/mapie_regressor.pkl",
        help="Path to the serialized MAPIE regressor (default: artifacts/model/mapie_regressor.pkl)",
    )
    parser.add_argument(
        "--input",
        default="artifacts/sample_input.csv",
        help="CSV file with feature rows to score (default: artifacts/sample_input.csv)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/predictions.csv",
        help="Where to write the predictions CSV (default: artifacts/predictions.csv)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Miscoverage rate for two-sided intervals (default: 0.1 for 90% intervals)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(resolve_input_path(args.model))
    input_path = Path(resolve_input_path(args.input))
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    mapie = joblib.load(model_path)

    features = pd.read_csv(input_path)

    y_pred, intervals = mapie.predict(features, alpha=[args.alpha])
    if intervals.ndim != 3:
        raise ValueError(f"Unexpected interval shape: {intervals.shape}")
    if intervals.shape[1] == 2:
        lower = intervals[:, 0, 0]
        upper = intervals[:, 1, 0]
    elif intervals.shape[2] == 2:
        lower = intervals[:, 0, 0]
        upper = intervals[:, 0, 1]
    else:
        raise ValueError(f"Unable to parse interval shape: {intervals.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = features.copy()
    results["prediction"] = y_pred
    results["lower_pi"] = lower
    results["upper_pi"] = upper
    results.to_csv(output_path, index=False)
    print(
        f"Predictions with {(1 - args.alpha) * 100:.1f}% intervals written to {output_path}"
    )


if __name__ == "__main__":
    main()
