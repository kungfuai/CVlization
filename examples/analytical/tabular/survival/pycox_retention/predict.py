import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH

from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_MODEL_DIR = "artifacts/model"
DEFAULT_INPUT = "artifacts/sample_input.csv"
DEFAULT_OUTPUT = "artifacts/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate survival forecasts with a trained PyCox retention model."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory containing saved model/scaler artifacts (default: artifacts/model)",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="CSV file with cohort features to score (default: artifacts/sample_input.csv)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write predictions CSV (default: artifacts/predictions.csv)",
    )
    return parser.parse_args()


def encode_inputs(df: pd.DataFrame, categorical_cols: List[str], feature_cols: List[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return encoded.reindex(columns=feature_cols, fill_value=0.0)


def build_model(input_dim: int, config: dict) -> CoxPH:
    layers = config.get("network", {}).get("layers", [128, 64])
    dropout = config.get("network", {}).get("dropout", 0.2)
    net = tt.practical.MLPVanilla(
        in_features=input_dim,
        num_nodes=layers,
        out_features=1,
        batch_norm=True,
        dropout=dropout,
    )
    optimizer = tt.optim.Adam()
    return CoxPH(net, optimizer)


def main() -> None:
    args = parse_args()
    model_dir = Path(resolve_input_path(args.model_dir))
    input_path = Path(resolve_input_path(args.input))
    output_path = Path(args.output)

    config = json.loads((model_dir / "config.json").read_text())
    feature_cols: List[str] = config["feature_columns"]
    categorical_cols: List[str] = config.get("categorical_columns", [])
    horizons: List[int] = config.get("horizons_days", [180, 365, 730])

    scaler = joblib.load(model_dir / "scaler.pkl")
    baseline_hazards = pd.read_csv(model_dir / "baseline_hazards.csv", index_col=0)
    baseline_hazards.index = baseline_hazards.index.astype(float)
    baseline_hazards = baseline_hazards.sort_index()

    baseline_cum_hazards = pd.read_csv(model_dir / "baseline_cum_hazards.csv", index_col=0)
    baseline_cum_hazards.index = baseline_cum_hazards.index.astype(float)
    baseline_cum_hazards = baseline_cum_hazards.sort_index()

    raw_features = pd.read_csv(input_path)
    encoded = encode_inputs(raw_features, categorical_cols, feature_cols)
    scaled = scaler.transform(encoded)
    tensor_inputs = torch.tensor(scaled, dtype=torch.float32)

    model = build_model(tensor_inputs.shape[1], config)
    model.load_model_weights(str(model_dir / "coxph_weights.pt"))
    model.baseline_hazards_ = baseline_hazards
    model.baseline_cumulative_hazards_ = baseline_cum_hazards

    surv_df = model.predict_surv_df(tensor_inputs)
    risk_scores = model.predict(tensor_inputs).flatten()

    predictions = raw_features.copy()
    predictions["predicted_risk"] = risk_scores

    for horizon in horizons:
        horizon_idx = np.searchsorted(surv_df.index.values, horizon, side="left")
        if horizon_idx >= len(surv_df.index):
            surv_probs = surv_df.iloc[-1].values
        else:
            surv_probs = surv_df.iloc[horizon_idx].values
        predictions[f"survival_prob_{horizon}d"] = surv_probs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(
        "Predictions written to",
        output_path,
        "with survival probabilities for horizons:",
        ", ".join(str(h) for h in horizons),
    )


if __name__ == "__main__":
    main()
