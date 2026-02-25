#!/usr/bin/env python3
"""
OCR Reading Benchmark Evaluator

Aggregates per-sample metrics from a predictions CSV produced by
vllm_ocr_eval.run_evaluation into a summary JSON and prints a table.

Usage:
    python evaluate.py --pred data/predictions/predictions.csv --output results/metrics.json
    python evaluate.py --pred predictions.csv --task lines_reading
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas and numpy required. Install with: pip install pandas numpy", file=sys.stderr)
    sys.exit(1)


# Columns produced by vllm_ocr_eval scoring scripts
METRIC_COLUMNS = {
    "character_error_rate": "CER",
    "word_error_rate": "WER",
    "anls": "ANLS",
    "bbox_precision": "BBox Precision",
    "bbox_recall": "BBox Recall",
    "bbox_f1": "BBox F1",
    "matched_box_ratio": "Matched Box Ratio",
}


def aggregate_metrics(df: pd.DataFrame) -> dict:
    """Compute mean of each available metric column."""
    results = {}
    for col, label in METRIC_COLUMNS.items():
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            valid = numeric.dropna()
            if len(valid) > 0:
                results[col] = {
                    "label": label,
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                    "count": int(len(valid)),
                }
    return results


def print_table(metrics: dict, num_samples: int) -> None:
    """Print a formatted table of aggregated metrics."""
    print("\n" + "=" * 60)
    print("OCR READING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'N':>6}")
    print("-" * 60)
    for col, info in metrics.items():
        print(
            f"{info['label']:<25} {info['mean']:>10.4f} {info['std']:>10.4f} {info['count']:>6}"
        )
    print("=" * 60)
    print(f"Total samples: {num_samples}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate OCR benchmark metrics from predictions CSV"
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Path to predictions CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save metrics summary JSON",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter to a specific task_type (e.g. 'lines_reading')",
    )

    args = parser.parse_args()

    if not args.pred.exists():
        print(f"Error: Predictions file not found: {args.pred}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.pred)
    print(f"Loaded {len(df)} rows from {args.pred}")

    if args.task and "task_type" in df.columns:
        df = df[df["task_type"] == args.task]
        print(f"Filtered to task_type='{args.task}': {len(df)} rows")

    if len(df) == 0:
        print("Warning: No rows remaining after filtering.", file=sys.stderr)
        sys.exit(1)

    metrics = aggregate_metrics(df)

    if not metrics:
        print(
            "Warning: No recognized metric columns found in CSV.\n"
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print_table(metrics, num_samples=len(df))

    summary = {
        "num_samples": len(df),
        "task_filter": args.task,
        "metrics": {col: info["mean"] for col, info in metrics.items()},
        "metrics_detail": metrics,
        "source": str(args.pred),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics saved to {args.output}")

    return summary


if __name__ == "__main__":
    main()
