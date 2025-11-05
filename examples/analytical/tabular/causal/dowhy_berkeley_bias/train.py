import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dowhy import CausalModel

ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
SAMPLE_OUTPUT_PATH = ARTIFACTS_DIR / "sample_predictions.csv"

BERKELEY_COUNTS: List[Tuple[str, str, int, int]] = [
    ("A", "male", 1, 512),
    ("A", "male", 0, 313),
    ("A", "female", 1, 89),
    ("A", "female", 0, 19),
    ("B", "male", 1, 353),
    ("B", "male", 0, 207),
    ("B", "female", 1, 17),
    ("B", "female", 0, 8),
    ("C", "male", 1, 120),
    ("C", "male", 0, 205),
    ("C", "female", 1, 202),
    ("C", "female", 0, 391),
    ("D", "male", 1, 138),
    ("D", "male", 0, 279),
    ("D", "female", 1, 131),
    ("D", "female", 0, 244),
    ("E", "male", 1, 53),
    ("E", "male", 0, 138),
    ("E", "female", 1, 94),
    ("E", "female", 0, 299),
    ("F", "male", 1, 22),
    ("F", "male", 0, 351),
    ("F", "female", 1, 24),
    ("F", "female", 0, 317),
]

GRAPH = r"""
digraph {
    gender_binary -> department;
    gender_binary -> admitted;
    department -> admitted;
}
"""


def expand_counts(counts: List[Tuple[str, str, int, int]]) -> pd.DataFrame:
    df = pd.DataFrame(counts, columns=["department", "gender", "admitted", "count"])
    expanded = df.loc[df.index.repeat(df["count"])].drop(columns=["count"]).reset_index(drop=True)
    expanded["gender_binary"] = (expanded["gender"].str.lower() == "female").astype(int)
    return expanded


def compute_naive_gap(df: pd.DataFrame) -> Dict[str, float]:
    rates = df.groupby("gender")["admitted"].mean()
    return {
        "female_rate": float(rates.get("female", np.nan)),
        "male_rate": float(rates.get("male", np.nan)),
        "female_minus_male": float(rates.get("female", 0.0) - rates.get("male", 0.0)),
    }


def run_causal_analysis(
    df: pd.DataFrame, run_estimation: bool, run_refutation: bool
) -> Dict[str, Optional[float]]:
    model = CausalModel(
        data=df,
        treatment="gender_binary",
        outcome="admitted",
        graph=GRAPH,
        common_causes=["department"],
        instruments=None,
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate_value: Optional[float] = None
    refute_new_effect: Optional[float] = None
    refute_effect_strength_on_t: Optional[float] = None

    if run_estimation:
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
        )
        estimate_value = float(estimate.value)

        if run_refutation:
            refutation = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause",
            )
            refute_new_effect = float(getattr(refutation, "new_effect", float("nan")))
            strength = getattr(refutation, "effect_strength_on_t", float("nan"))
            refute_effect_strength_on_t = float(strength)

    return {
        "identified_estimand": str(identified_estimand),
        "adjusted_effect": estimate_value,
        "refute_new_effect": refute_new_effect,
        "refute_effect_strength_on_t": refute_effect_strength_on_t,
    }


def per_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["department", "gender"])["admitted"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "admission_rate", "count": "observations"})
        .reset_index()
    )
    pivot = summary.pivot(index="department", columns="gender", values="admission_rate")
    pivot = pivot.rename(columns={"female": "rate_female", "male": "rate_male"})
    pivot["rate_diff_female_minus_male"] = pivot["rate_female"] - pivot["rate_male"]
    return pivot.reset_index()


def save_artifacts(
    df: pd.DataFrame,
    naive_gap: Dict[str, float],
    causal_results: Dict[str, Optional[float]],
    department_summary: pd.DataFrame,
    stage: str,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "naive_rates": naive_gap,
        "causal": causal_results,
        "simpsons_paradox_gap": float(naive_gap["female_minus_male"]),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "dataset": "uc_berkeley_grad_admissions_1973",
        "source_notes": "Aggregated counts from UC Berkeley admission data (Bickel, Hammel, O'Connell, 1975).",
        "records": int(len(df)),
        "departments": sorted(df["department"].unique().tolist()),
        "graph": GRAPH.strip(),
        "stage": stage,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    aggregated = pd.DataFrame(BERKELEY_COUNTS, columns=["department", "gender", "admitted", "count"])
    aggregated.to_csv(SAMPLE_INPUT_PATH, index=False)

    department_summary.to_csv(SAMPLE_OUTPUT_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="DoWhy Berkeley gender bias analysis")
    parser.add_argument(
        "--stage",
        choices=["discovery", "identify", "estimate", "refute", "run"],
        default="refute",
        help="Pipeline stage to execute (run=full pipeline).",
    )
    args = parser.parse_args()
    stage = "refute" if args.stage == "run" else args.stage

    df = expand_counts(BERKELEY_COUNTS)

    naive_gap = compute_naive_gap(df)
    dept_summary = per_department_summary(df)

    run_identify = stage in {"identify", "estimate", "refute"}
    run_estimation = stage in {"estimate", "refute"}
    run_refutation = stage == "refute"

    if run_identify:
        causal_results = run_causal_analysis(df, run_estimation, run_refutation)
    else:
        causal_results = {
            "identified_estimand": None,
            "adjusted_effect": None,
            "refute_new_effect": None,
            "refute_effect_strength_on_t": None,
        }

    save_artifacts(df, naive_gap, causal_results, dept_summary, stage)

    print("Stage:", stage)
    print("Naive admission gap (female - male):", naive_gap["female_minus_male"])
    if causal_results["adjusted_effect"] is not None:
        print("Adjusted causal effect (DoWhy):", causal_results["adjusted_effect"])
    if causal_results["refute_new_effect"] is not None:
        print(
            "Refutation (random common cause) new_effect:",
            causal_results["refute_new_effect"],
        )


if __name__ == "__main__":
    main()
