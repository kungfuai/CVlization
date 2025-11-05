from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    results_path = Path("var/results.json")
    subprocess.run(
        [sys.executable, "optimize.py", "--output", str(results_path)],
        check=True,
    )

    results = json.loads(results_path.read_text())
    baseline = results.get("baseline_test_accuracy")
    optimized = results.get("optimized_test_accuracy")

    if baseline is None or optimized is None:
        raise SystemExit("Missing accuracy metrics in results.json")

    if optimized < baseline:
        raise SystemExit(
            f"Optimized prompt underperformed baseline (baseline={baseline:.3f}, optimized={optimized:.3f})"
        )

    print(
        "Evaluation succeeded. "
        f"Baseline test accuracy: {baseline:.3f}, Optimized test accuracy: {optimized:.3f}"
    )


if __name__ == "__main__":
    main()
