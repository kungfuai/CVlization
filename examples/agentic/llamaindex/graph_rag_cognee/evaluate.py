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

QUESTION = "Tell me who are the people mentioned?"
EXPECTED_NAMES = {"jessica miller", "david thompson"}


def run_query() -> dict:
    cmd = [sys.executable, "query.py", QUESTION, "--provider", "mock"]
    subprocess.run(cmd, check=True)
    result_path = Path("outputs") / "result.json"
    return json.loads(result_path.read_text())


def main() -> None:
    result = run_query()
    answer_text = "\n".join(result.get("graph_answers", []))
    missing = [name for name in EXPECTED_NAMES if name not in answer_text.lower()]
    if missing:
        raise SystemExit(f"Mock graph answer missing: {', '.join(missing)}")
    print("Mock GraphRAG smoke test passed.")


if __name__ == "__main__":
    main()
