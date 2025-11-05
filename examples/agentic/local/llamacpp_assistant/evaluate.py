from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

QUESTIONS = [
    ("What was Lyft's 2021 revenue growth?", {"lyft"}),
    ("Summarize Uber's growth catalysts in 2021.", {"uber"}),
    ("Compare Lyft and Uber revenue performance in 2021.", {"lyft", "uber"}),
]


def run_query(question: str) -> dict:
    cmd = [sys.executable, "query.py", question, "--json"]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def main() -> None:
    results = []
    for question, required_tokens in QUESTIONS:
        result = run_query(question)
        answer = result.get("answer", "").lower()
        missing = [token for token in required_tokens if token not in answer]
        if missing:
            raise SystemExit(
                f"Evaluation failed for question '{question}': answer missing {', '.join(missing)}."
            )
        results.append(result)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("outputs/eval_results.json").write_text(json.dumps(results, indent=2))
    print("Evaluation succeeded; results stored in outputs/eval_results.json")


if __name__ == "__main__":
    main()
