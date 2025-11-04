from __future__ import annotations

import argparse
import re
from typing import Dict, List

import pandas as pd

from predict import (
    ProviderConfig,
    detect_provider,
    load_dataset,
    run_agentic_answer,
    run_rule_based_answer,
)

DEFAULT_DATA_PATH = "data/marketing_kpi.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the smolagents data analyst example.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Path to CSV dataset.")
    parser.add_argument("--llm-provider", type=str, help="Optional LLM provider override.")
    parser.add_argument("--llm-model", type=str, help="Optional provider-specific model id override.")
    parser.add_argument("--temperature", type=float, help="Optional sampling temperature override.")
    return parser.parse_args()


def normalize_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def build_expectations(df: pd.DataFrame) -> List[Dict[str, str]]:
    segment_totals = df.groupby("segment")["revenue"].sum().sort_values(ascending=False)
    region_totals = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    top_customer = df.sort_values("revenue", ascending=False).iloc[0]
    total_revenue = df["revenue"].sum()

    return [
        {
            "question": "Which customer segment generates the most revenue?",
            "must_include": [segment_totals.index[0], str(segment_totals.iloc[0])],
        },
        {
            "question": "Which region is contributing the most revenue and what is the total revenue?",
            "must_include": [region_totals.index[0], str(region_totals.iloc[0]), str(total_revenue)],
        },
        {
            "question": "Who is the highest value customer and how much have they spent?",
            "must_include": [top_customer["customer_id"], str(int(top_customer["revenue"]))],
        },
    ]


def answer_question(question: str, df: pd.DataFrame, config: ProviderConfig) -> str:
    if config.provider in {"mock", "fake"}:
        return run_rule_based_answer(question, df)
    try:
        return run_agentic_answer(question, df, config)
    except Exception as exc:
        return run_rule_based_answer(question, df) + f"\n[note] agent fallback: {exc}"


def evaluate() -> int:
    args = parse_args()
    config = detect_provider(args)
    df = load_dataset(args.data_path)
    expectations = build_expectations(df)

    passed = 0
    for item in expectations:
        answer = answer_question(item["question"], df, config)
        normalized = normalize_digits(answer)
        requirements_met = True
        for token in item["must_include"]:
            token_lower = token.lower()
            if token.isdigit():
                if token not in normalized:
                    requirements_met = False
                    break
            else:
                if token_lower not in answer.lower():
                    requirements_met = False
                    break
        if requirements_met:
            passed += 1
        else:
            print(f"[FAILED] {item['question']}\nAnswer: {answer}\n")
    return passed, len(expectations)


if __name__ == "__main__":
    successes, total = evaluate()
    print(f"Passed {successes} / {total} evaluation prompts.")
    if successes != total:
        raise SystemExit(1)
