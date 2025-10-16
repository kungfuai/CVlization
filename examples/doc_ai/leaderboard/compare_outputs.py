#!/usr/bin/env python3
"""
Simple script to compare outputs from different models side-by-side.
Usage: python compare_outputs.py results/20250116_123456/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Compare model outputs side-by-side")
    parser.add_argument("results_dir", type=str, help="Path to benchmark results directory")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text",
                       help="Output format")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        return 1

    # Read benchmark CSV
    csv_path = results_path / "benchmark.csv"
    if not csv_path.exists():
        print(f"Error: benchmark.csv not found in {results_path}")
        return 1

    # Group outputs by image
    outputs_by_image = defaultdict(dict)

    # Find all model output directories
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Find output files in this model's directory
        for output_file in model_dir.glob("*_output.txt"):
            image_name = output_file.stem.replace("_output", "")

            try:
                output_text = output_file.read_text()
                outputs_by_image[image_name][model_name] = output_text
            except Exception as e:
                outputs_by_image[image_name][model_name] = f"[Error reading output: {e}]"

    # Display comparison
    if args.format == "json":
        print(json.dumps(dict(outputs_by_image), indent=2))

    elif args.format == "markdown":
        print("# Doc AI Model Comparison\n")
        for image_name, model_outputs in sorted(outputs_by_image.items()):
            print(f"## {image_name}\n")
            for model_name, output in sorted(model_outputs.items()):
                print(f"### {model_name}\n")
                print("```")
                print(output.strip())
                print("```\n")

    else:  # text format
        for image_name, model_outputs in sorted(outputs_by_image.items()):
            print("=" * 80)
            print(f"Image: {image_name}")
            print("=" * 80)
            print()

            for model_name, output in sorted(model_outputs.items()):
                print(f"--- {model_name} " + "-" * (70 - len(model_name)))
                print(output.strip())
                print()

            print()

    return 0


if __name__ == "__main__":
    exit(main())
