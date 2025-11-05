from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_predict(question: str) -> str:
    cmd = [sys.executable, "query.py", question]
    subprocess.run(cmd, check=True)
    output_file = Path("outputs") / "response.txt"
    return output_file.read_text()


def main() -> None:
    single = run_predict("What was Lyft's revenue growth in 2021?")
    if "Lyft" not in single:
        raise SystemExit("Lyft response missing expected context.")

    compare = run_predict(
        "Compare the revenue growth between Lyft and Uber in 2021 and summarize the key differences."
    )
    if "Lyft" not in compare or "Uber" not in compare:
        raise SystemExit("Comparison response missing Lyft or Uber context.")

    print("Evaluation succeeded; mock outputs contain expected company mentions.")


if __name__ == "__main__":
    main()
