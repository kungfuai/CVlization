from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import build_agent
from dotenv import load_dotenv

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "response.txt"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the finance ReAct agent on a question.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What was Lyft's revenue growth in 2021?",
        help="Question to ask the agent.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider override (mock, openai, hf).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = build_agent(args.provider)
    answer = agent.query(args.question)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_FILE.write_text(answer)

    print("Question:", args.question)
    print("---")
    print(answer)
    print(f"\nSaved response to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
