from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import build_graph, build_retriever, get_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph helpdesk agent.")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question to route through the helpdesk agent.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("var") / "chroma",
        help="Vector store directory (output of ingest.py).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of documents to retrieve as context.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved context and metadata after answering.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit answer payload as JSON (answer, citations).",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        help="Override LLM provider (mock, ollama, openai, groq).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        help="Override model name for the selected provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override sampling temperature for the selected provider.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = build_retriever(args.persist_dir, top_k=args.top_k)
    llm, provider = get_llm(
        provider_override=args.llm_provider,
        model_override=args.llm_model,
        temperature_override=args.temperature,
    )
    graph = build_graph(retriever=retriever, llm=llm, provider=provider)

    result = graph.invoke({"question": args.question})
    answer = result.get("answer", "").strip()

    if args.json:
        payload = {
            "question": args.question,
            "answer": answer,
            "citations": result.get("citations", []),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Q: {args.question}")
        print(f"A: {answer}")
        if args.show_context:
            print("\nRetrieved context:")
            for citation in result.get("citations", []):
                print(
                    f"- [doc:{citation['doc']}] {citation.get('source') or 'unknown source'}"
                )


if __name__ == "__main__":
    main()
