from __future__ import annotations

import json
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

QUESTIONS = [
    "What is the maximum age among the individuals?",
    "How many individuals have an occupation related to science or engineering?",
    "How many individuals have a phone number starting with '+1 234'?",
    "What is the percentage of individuals residing in California (CA)?",
    "How many individuals have a major in Psychology?",
]


def run_query(question: str) -> dict:
    cmd = [sys.executable, "query.py", question, "--provider", "mock"]
    subprocess.run(cmd, check=True)
    result_path = Path("outputs") / "result.json"
    return json.loads(result_path.read_text())


def main() -> None:
    for question in QUESTIONS:
        result = run_query(question)
        answer = result.get("answer", "").lower()
        if "no heuristic" in answer:
            raise SystemExit(f"Mock evaluation failed for question: {question}")
    print("All mock evaluations succeeded.")


if __name__ == "__main__":
    main()
