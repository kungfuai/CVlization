import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from cvlization.paths import resolve_input_path, resolve_output_path

MODEL_DIR = Path("artifacts/models")
BASE_MODEL_PATH = MODEL_DIR / "lightgbm_classifier.joblib"
SIGMOID_CALIBRATOR_PATH = MODEL_DIR / "sigmoid_calibrator.joblib"
ISOTONIC_CALIBRATOR_PATH = MODEL_DIR / "isotonic_calibrator.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.json"
CONFORMAL_METADATA_PATH = MODEL_DIR / "conformal_metadata.json"


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_category_map(df: pd.DataFrame, category_map: Dict[str, List[str]]) -> pd.DataFrame:
    result = df.copy()
    for col, categories in category_map.items():
        if col not in result.columns:
            raise ValueError(f"Input data is missing categorical column '{col}'")
        result[col] = (
            result[col]
            .astype("string")
            .fillna("Unknown")
            .apply(lambda val: val if val in categories else "Unknown")
            .astype(pd.CategoricalDtype(categories=categories))
        )
    return result


def ensure_feature_columns(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")
    return df[expected_cols].copy()


def compute_conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(sorted_scores[rank])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict calibrated probabilities and conformal sets using LightGBM.")
    parser.add_argument("--input", required=True, help="Path to CSV containing feature columns.")
    parser.add_argument("--output", required=True, help="Destination CSV for predictions.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Miscoverage level for conformal sets (default: uses training alpha).",
    )
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Include original features alongside predictions in the output CSV.",
    )
    args = parser.parse_args()

    raw_input = pd.read_csv(resolve_input_path(args.input))

    preprocessor = load_json(PREPROCESSOR_PATH)
    feature_cols: List[str] = preprocessor["feature_columns"]
    categorical_cols: List[str] = preprocessor["categorical_columns"]
    numeric_cols: List[str] = preprocessor.get("numeric_columns", [])
    category_map: Dict[str, List[str]] = preprocessor["category_map"]
    classes: List[str] = preprocessor["classes"]

    features = ensure_feature_columns(raw_input, feature_cols)
    for col in numeric_cols:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)
    features = apply_category_map(features, category_map)

    base_model = joblib.load(BASE_MODEL_PATH)
    sigmoid_calibrator = joblib.load(SIGMOID_CALIBRATOR_PATH)
    isotonic_calibrator = joblib.load(ISOTONIC_CALIBRATOR_PATH)

    base_probs = base_model.predict_proba(features)
    sigmoid_probs = sigmoid_calibrator.predict_proba(features)
    isotonic_probs = isotonic_calibrator.predict_proba(features)

    conformal_metadata = load_json(CONFORMAL_METADATA_PATH)
    scores = np.asarray(conformal_metadata["nonconformity_scores"], dtype=float)
    alpha = args.alpha if args.alpha is not None else float(conformal_metadata["alpha_default"])
    alpha = min(max(alpha, 1e-6), 0.5)
    qhat = compute_conformal_threshold(scores, alpha)
    threshold_probability = 1.0 - qhat

    prediction_sets = sigmoid_probs >= threshold_probability
    conformal_sets = []
    for flags in prediction_sets:
        selected = [classes[i] for i, flag in enumerate(flags) if flag]
        conformal_sets.append(";".join(selected))

    predictions = pd.DataFrame({"conformal_threshold": [threshold_probability] * len(features)})
    for idx, class_name in enumerate(classes):
        predictions[f"prob_base_{class_name}"] = base_probs[:, idx]
        predictions[f"prob_sigmoid_{class_name}"] = sigmoid_probs[:, idx]
        predictions[f"prob_isotonic_{class_name}"] = isotonic_probs[:, idx]

    predictions["predicted_class_base"] = [classes[idx] for idx in np.argmax(base_probs, axis=1)]
    predictions["predicted_class_sigmoid"] = [classes[idx] for idx in np.argmax(sigmoid_probs, axis=1)]
    predictions["predicted_class_isotonic"] = [classes[idx] for idx in np.argmax(isotonic_probs, axis=1)]
    predictions["conformal_set"] = conformal_sets

    if args.include_features:
        readable_input = raw_input.copy()
        for col in categorical_cols:
            if col in readable_input.columns:
                readable_input[col] = readable_input[col].astype(str)
        output_df = pd.concat([readable_input.reset_index(drop=True), predictions], axis=1)
    else:
        output_df = predictions

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
