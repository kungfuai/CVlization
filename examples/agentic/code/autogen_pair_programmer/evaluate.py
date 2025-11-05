from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_pair_programmer(args: list[str]) -> dict:
    cmd = [sys.executable, "pair_programmer.py", "--task", "simple_math", "--json"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main() -> None:
    summary = run_pair_programmer([])
    if not summary.get("tests_passed"):
        raise SystemExit("Pair programmer failed to satisfy the tests")
    print("Evaluation succeeded: mock provider produced passing solution.")


if __name__ == "__main__":
    main()
