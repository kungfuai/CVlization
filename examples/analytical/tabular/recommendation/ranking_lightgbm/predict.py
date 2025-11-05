import argparse
import json
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_PATH = Path("artifacts/ranking_lightgbm.txt")
FEATURE_NAMES_PATH = Path("artifacts/feature_names.json")


def load_feature_names() -> List[str]:
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature list at {FEATURE_NAMES_PATH}. Run train.py first."
        )
    with FEATURE_NAMES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model() -> lgb.Booster:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file {MODEL_PATH}. Run train.py first.")
    return lgb.Booster(model_file=str(MODEL_PATH))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates using the trained LightGBM model.")
    parser.add_argument("--input", required=True, help="Path to CSV containing candidate rows.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path where ranked predictions will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k cutoff per query (defaults to returning all rows).",
    )
    args = parser.parse_args()

    feature_names = load_feature_names()
    df = pd.read_csv(args.input)

    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"Input is missing expected feature columns: {missing[:10]}")

    model = load_model()
    scores = model.predict(df[feature_names])

    result = df.copy()
    result["score"] = scores

    if "query_id" in result.columns:
        result = result.sort_values(["query_id", "score"], ascending=[True, False])
        result["rank"] = (
            result.groupby("query_id")["score"].rank(method="first", ascending=False).astype(int)
        )
        if args.top_k:
            result = result.groupby("query_id", group_keys=False).head(args.top_k)
    else:
        result = result.sort_values("score", ascending=False)
        result["rank"] = np.arange(1, len(result) + 1, dtype=int)
        if args.top_k:
            result = result.head(args.top_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Ranked predictions saved to {output_path}")


if __name__ == "__main__":
    main()
