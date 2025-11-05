from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv


def _load_env() -> None:
    candidates = []
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.append(parent / ".env")
    candidates.append(Path("/cvlization_repo/.env"))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            load_dotenv(candidate, override=False)


_load_env()


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
