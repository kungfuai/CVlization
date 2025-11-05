import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from typing import List

from featuretools import dfs
from featuretools.entityset import EntitySet
from featuretools.primitives import Count, Max, Mean, Min, Sum

DEFAULT_MODEL_DIR = "artifacts/model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature pipeline + classifier on new data.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory with saved model artifacts")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV file with raw records to score")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to write predictions CSV")
    return parser.parse_args()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    es = EntitySet(id="customers")
    es = es.add_dataframe(
        dataframe_name="records",
        dataframe=df.reset_index(drop=True),
        make_index=True,
        index="record_id",
    )
    feature_matrix, _ = dfs(
        entityset=es,
        target_dataframe_name="records",
        agg_primitives=[Sum, Mean, Max, Min, Count],
        trans_primitives=["day", "month", "weekday"],
        max_depth=1,
        verbose=False,
    )
    engineered = pd.concat([df.reset_index(drop=True), feature_matrix.reset_index(drop=True)], axis=1)
    engineered = engineered.loc[:, ~engineered.columns.duplicated()]
    return engineered


def ensure_columns(engineered: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    engineered = engineered.copy()
    for col in expected_cols:
        if col not in engineered.columns:
            engineered[col] = 0
    return engineered.reindex(columns=expected_cols, fill_value=0)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)

    pipeline = joblib.load(model_dir / "gbm_classifier.pkl")
    config = json.loads((model_dir / "config.json").read_text())
    engineered_cols = config["feature_columns"]["engineered"]
    numeric_cols = config.get("numeric_columns", [])
    categorical_cols = config.get("categorical_columns", [])

    raw_df = pd.read_csv(input_path)
    processed = preprocess(raw_df)
    # Determine if input already contains engineered columns
    if set(engineered_cols).issubset(processed.columns):
        engineered = processed[engineered_cols]
    else:
        engineered = build_feature_matrix(processed)
        engineered = ensure_columns(engineered, engineered_cols)

    if numeric_cols:
        engineered[numeric_cols] = engineered[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    if categorical_cols:
        for col in categorical_cols:
            if col in engineered.columns:
                engineered[col] = engineered[col].astype(str).fillna("missing")

    preds = pipeline.predict(engineered)
    proba = pipeline.predict_proba(engineered)[:, 1]

    result = raw_df.copy()
    result["predicted_label"] = preds
    result["predicted_prob"] = proba

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
