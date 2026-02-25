#!/usr/bin/env python3
"""
Run the AdderBoard benchmark across one or more submissions and emit a leaderboard.
"""

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
from typing import Any, Dict, List

from verify import load_submission, run_test


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_results(submission_paths: List[str], num_tests: int, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in submission_paths:
        print(f"\n=== Running: {path} ===")
        row: Dict[str, Any] = {"submission_path": path}
        try:
            mod = load_submission(path)
            result = run_test(mod, num_tests=num_tests, seed=seed)
            metadata = result.get("metadata", {})
            row.update(
                {
                    "model": metadata.get("name", pathlib.Path(path).stem),
                    "author": metadata.get("author", "unknown"),
                    "params": metadata.get("params", ""),
                    "architecture": metadata.get("architecture", ""),
                    "tricks": "; ".join(metadata.get("tricks", [])),
                    "accuracy": round(result["accuracy"], 4),
                    "qualified": result["qualified"],
                    "passed": result["passed"],
                    "total": result["total"],
                    "time_seconds": round(result["time"], 4),
                    "throughput": round(result["throughput"], 4),
                    "status": "ok",
                    "error": "",
                }
            )
            print(
                f"Result: {row['accuracy']:.2f}% | "
                f"{'QUALIFIED' if row['qualified'] else 'NOT QUALIFIED'} | "
                f"{row['time_seconds']:.2f}s"
            )
        except Exception as exc:  # pylint: disable=broad-except
            row.update(
                {
                    "model": pathlib.Path(path).stem,
                    "author": "",
                    "params": "",
                    "architecture": "",
                    "tricks": "",
                    "accuracy": 0.0,
                    "qualified": False,
                    "passed": 0,
                    "total": 0,
                    "time_seconds": 0.0,
                    "throughput": 0.0,
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"Result: ERROR ({exc})")
        rows.append(row)
    return rows


def _rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda x: (
            not bool(x.get("qualified")),
            -_safe_float(x.get("accuracy"), default=-1.0),
            _safe_float(x.get("params"), default=float("inf")),
            _safe_float(x.get("time_seconds"), default=float("inf")),
        ),
    )


def _write_csv(rows: List[Dict[str, Any]], output_path: pathlib.Path):
    fields = [
        "rank",
        "model",
        "author",
        "params",
        "accuracy",
        "qualified",
        "passed",
        "total",
        "time_seconds",
        "throughput",
        "architecture",
        "tricks",
        "status",
        "error",
        "submission_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = idx
            writer.writerow(out)


def _write_markdown(rows: List[Dict[str, Any]], output_path: pathlib.Path):
    lines = [
        "# AdderBoard Benchmark Leaderboard",
        "",
        "| Rank | Model | Author | Params | Accuracy | Qualified | Time (s) | Status |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{idx} | "
            f"{row.get('model', '')} | "
            f"{row.get('author', '')} | "
            f"{row.get('params', '')} | "
            f"{_safe_float(row.get('accuracy'), 0.0):.2f}% | "
            f"{'yes' if row.get('qualified') else 'no'} | "
            f"{_safe_float(row.get('time_seconds'), 0.0):.2f} | "
            f"{row.get('status', '')} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run AdderBoard benchmark on submissions")
    parser.add_argument("submissions", nargs="+", help="Submission files to benchmark")
    parser.add_argument("--num-tests", type=int, default=10000, help="Number of random tests")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where timestamped benchmark output is written",
    )
    args = parser.parse_args()

    rows = _compute_results(args.submissions, num_tests=args.num_tests, seed=args.seed)
    ranked = _rank_rows(rows)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = pathlib.Path(args.results_dir)
    run_dir = base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(ranked, run_dir / "scores.csv")
    _write_markdown(ranked, run_dir / "leaderboard.md")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "num_submissions": len(ranked),
                "num_tests": args.num_tests,
                "seed": args.seed,
                "results": ranked,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    latest = base / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    print(f"\nWrote results to: {run_dir}")
    print(f"Updated latest link: {latest} -> {run_dir.name}")


if __name__ == "__main__":
    main()
