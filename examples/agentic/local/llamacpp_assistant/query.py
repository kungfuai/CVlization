from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import OUTPUT_DIR, run_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline llama.cpp finance QA query.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Compare Lyft and Uber revenue growth in 2021.",
        help="Question to pose to the offline assistant.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override number of retrieved chunks (defaults to LLAMACPP_TOP_K or 4).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_query(args.question, top_k=args.top_k)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "response.json"
    out_path.write_text(json.dumps(result, indent=2))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Question: {result['question']}")
        print("---")
        print(result["answer"])
        print("\nTop context snippets:")
        for chunk in result["retrieved"]:
            print(f"- {chunk['doc_id']} (score={chunk['score']:.3f}): {chunk['text'][:120]}...")
        print(f"\nSaved response to {out_path}")


if __name__ == "__main__":
    main()
