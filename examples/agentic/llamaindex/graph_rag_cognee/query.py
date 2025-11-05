from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import run_query
from dotenv import load_dotenv

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    parser = argparse.ArgumentParser(description="Run Cognee GraphRAG example")
    parser.add_argument(
        "question",
        nargs="?",
        default="Tell me who are the people mentioned?",
        help="Question to ask the GraphRAG pipeline.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider (mock, openai, hf). Defaults to environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_query(args.question, provider=args.provider)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2))

    print(f"Question: {args.question}")
    print(f"Mode: {result['mode']}")
    print("Graph answers:")
    for answer in result.get("graph_answers", []):
        print(f"- {answer}")
    print("RAG answers:")
    for answer in result.get("rag_answers", []):
        print(f"- {answer}")
    print("Related nodes:")
    for node in result.get("related_nodes", []):
        print(f"- {node}")
    print(f"Saved structured output to {OUTPUT_DIR / 'result.json'}")


if __name__ == "__main__":
    main()
