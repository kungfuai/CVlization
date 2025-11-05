from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from pipeline import build_graph, build_retriever, get_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LangGraph helpdesk agent with heuristic metrics."
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("var") / "chroma",
        help="Vector store directory.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=Path("data") / "evals" / "helpdesk_eval.jsonl",
        help="JSONL file with question, ground_truth fields.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Retriever top_k value.",
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

    dataset = load_dataset(
        "json", data_files=str(args.eval_data), split="train"
    )

    total = len(dataset)
    exact_hits = 0
    context_hits = 0
    answers: List[Dict[str, Any]] = []

    for row in dataset:
        question = row["question"]
        ground_truth = row["answer"]
        result = graph.invoke({"question": question})
        docs = result.get("citations", [])
        contexts = [doc.get("content", "") for doc in docs]
        answer = result.get("answer", "")

        if ground_truth and ground_truth.lower() in answer.lower():
            exact_hits += 1

        context_text = " ".join(contexts).lower()
        if ground_truth and ground_truth.lower() in context_text:
            context_hits += 1

        answers.append(
            {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": contexts,
            }
        )

    print("Evaluation summary:")
    print(f"- Samples evaluated: {total}")
    print(f"- Exact answer hit rate: {exact_hits / total:.2%}" if total else "- No samples")
    print(f"- Context coverage rate: {context_hits / total:.2%}" if total else "- No samples")

    print("\nSample outputs:")
    for entry in answers[:3]:
        print(f"Q: {entry['question']}")
        print(f"A: {entry['answer']}")
        print(f"GT: {entry['ground_truth']}")
        if entry["contexts"]:
            print(f"Context snippet: {entry['contexts'][0][:120]}...")
        print("---")


if __name__ == "__main__":
    main()
