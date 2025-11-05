from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_optimize(json_mode: bool = True) -> dict:
    cmd = [sys.executable, "optimize.py"]
    if json_mode:
        cmd += ["--output", "var/results.json"]
    subprocess.run(cmd, check=True)
    results = json.loads(Path("var/results.json").read_text())
    return results


def main() -> None:
    results = run_optimize()
    baseline = results.get("baseline_score") or results.get("baseline_accuracy")
    optimized = results.get("optimized_score") or results.get("optimized_accuracy")
    if optimized is None or baseline is None:
        raise SystemExit("Optimization summary missing metrics")
    if optimized < baseline:
        raise SystemExit("Optimized prompt underperformed baseline")
    print("Evaluation succeeded. Baseline:", baseline, "Optimized:", optimized)


if __name__ == "__main__":
    main()
