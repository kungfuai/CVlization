import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from dowhy import CausalModel

GRAPH = r"""
digraph {
    gender_binary -> department;
    gender_binary -> admitted;
    department -> admitted;
}
"""


def expand_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {"department", "gender", "admitted", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")

    expanded = df.loc[df.index.repeat(df["count"])].drop(columns=["count"]).reset_index(drop=True)
    expanded["gender"] = expanded["gender"].str.lower()
    expanded["admitted"] = expanded["admitted"].astype(int)
    expanded["gender_binary"] = (expanded["gender"] == "female").astype(int)
    return expanded


def compute_naive_gap(df: pd.DataFrame) -> Dict[str, float]:
    rates = df.groupby("gender")["admitted"].mean()
    return {
        "female_rate": float(rates.get("female", np.nan)),
        "male_rate": float(rates.get("male", np.nan)),
        "female_minus_male": float(rates.get("female", 0.0) - rates.get("male", 0.0)),
    }


def causal_effect(df: pd.DataFrame) -> float:
    model = CausalModel(
        data=df,
        treatment="gender_binary",
        outcome="admitted",
        graph=GRAPH,
        common_causes=["department"],
        instruments=None,
    )

    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    return float(estimate.value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gender admission gap and causal adjustment for Berkeley-style aggregated data."
    )
    parser.add_argument("--input", required=True, help="Path to CSV with columns department,gender,admitted,count.")
    parser.add_argument("--output", required=True, help="Destination CSV for per-department summary with predictions.")
    parser.add_argument(
        "--summary",
        default="artifacts/predict_summary.json",
        help="Path to write JSON summary of overall metrics.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    expanded = expand_input(raw)

    naive_gap = compute_naive_gap(expanded)
    adjusted = causal_effect(expanded)

    department_summary = (
        expanded.groupby(["department", "gender"])["admitted"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "admission_rate", "count": "observations"})
        .reset_index()
    )
    pivot = department_summary.pivot(index="department", columns="gender", values="admission_rate")
    pivot = pivot.rename(columns={"female": "rate_female", "male": "rate_male"})
    pivot["rate_diff_female_minus_male"] = pivot["rate_female"] - pivot["rate_male"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.reset_index().to_csv(output_path, index=False)
    print(f"Per-department summary written to {output_path}")

    summary = {
        "naive_rates": naive_gap,
        "adjusted_effect": adjusted,
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
