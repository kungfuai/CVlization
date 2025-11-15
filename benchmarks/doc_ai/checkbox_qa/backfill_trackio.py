#!/usr/bin/env python3
"""
Backfill existing experiment results to trackio.

Reads eval_results.json files from results/ directories and logs them to trackio.
"""

import json
from pathlib import Path
import re

try:
    import trackio
except ImportError:
    print("Error: trackio not installed. Run: pip install trackio")
    exit(1)


def parse_config_from_dirname(dirname: str) -> dict:
    """Parse experiment configuration from directory name."""
    config = {}

    # Extract model
    if dirname.startswith("qwen_"):
        config["model"] = "qwen3-vl-2b"
    elif dirname.startswith("phi4_"):
        config["model"] = "phi-4-14b"
    else:
        config["model"] = "unknown"

    # Extract pages
    pages_match = re.search(r'_p(\d+)_', dirname)
    if pages_match:
        config["max_pages"] = int(pages_match.group(1))

    # Extract image size
    size_match = re.search(r'_s(\d+)_', dirname) or re.search(r'_(\d+)px', dirname)
    if size_match:
        config["max_image_size"] = int(size_match.group(1))

    # Determine sampling config
    if "greedy" in dirname:
        config["sampling"] = "greedy"
    elif re.search(r'_T([\d.]+)', dirname):
        temp_match = re.search(r'_T([\d.]+)', dirname)
        config["sampling"] = "enabled"
        config["temperature"] = float(temp_match.group(1))

        # Check for top-k
        k_match = re.search(r'_k(\d+)', dirname)
        if k_match:
            config["top_k"] = int(k_match.group(1))

    return config


def backfill_results(results_dir: Path, project: str = "checkbox-qa", dry_run: bool = False):
    """Backfill all eval_results.json files to trackio."""

    logged = 0
    skipped = 0

    for result_path in sorted(results_dir.glob("*/eval_results.json")):
        dirname = result_path.parent.name

        # Skip if not qwen or phi4
        if not (dirname.startswith("qwen_") or dirname.startswith("phi4_")):
            print(f"Skipping {dirname} (not a benchmark result)")
            skipped += 1
            continue

        # Parse config
        config = parse_config_from_dirname(dirname)

        # Load results
        with open(result_path) as f:
            eval_results = json.load(f)

        # Generate run name
        run_name = dirname

        if dry_run:
            print(f"[DRY RUN] Would log: {run_name}")
            print(f"  Config: {config}")
            print(f"  ANLS: {eval_results['anls_score']:.4f}")
            print()
            logged += 1
            continue

        # Log to trackio
        try:
            run = trackio.init(
                project=project,
                name=run_name,
                config=config
            )
            run.log({
                "anls_score": eval_results["anls_score"],
                "num_correct": eval_results["num_correct"],
                "total_questions": eval_results["total_questions"],
                "accuracy": eval_results["num_correct"] / eval_results["total_questions"]
            })
            run.finish()
            print(f"✓ Logged: {run_name} (ANLS {eval_results['anls_score']:.4f})")
            logged += 1
        except Exception as e:
            print(f"✗ Failed to log {run_name}: {e}")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Backfill Summary")
    print(f"{'='*60}")
    print(f"Logged: {logged}")
    print(f"Skipped: {skipped}")
    print(f"Total: {logged + skipped}")

    if not dry_run:
        print(f"\nView dashboard: trackio ui")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill experiment results to trackio")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="checkbox-qa",
        help="Trackio project name"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be logged without actually logging"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        exit(1)

    backfill_results(args.results_dir, args.project, args.dry_run)
