#!/usr/bin/env python3
"""
CheckboxQA Benchmark Runner

Runs models on CheckboxQA dataset and evaluates results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from dataset_builder import CheckboxQADataset


def run_model_on_document(
    adapter_path: Path,
    pdf_path: Path,
    question: str,
    output_file: Path
) -> str:
    """
    Run a model adapter on a single question.

    Returns:
        The model's answer (string)
    """
    cmd = [
        str(adapter_path),
        str(pdf_path),
        question,
        "--output", str(output_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per question
        )

        if result.returncode != 0:
            print(f"Warning: Adapter failed for question: {question[:50]}...", file=sys.stderr)
            print(f"Error: {result.stderr}", file=sys.stderr)
            return ""

        # Read answer from output file
        if output_file.exists():
            with open(output_file, 'r') as f:
                answer = f.read().strip()
            return answer
        else:
            print(f"Warning: Output file not created: {output_file}", file=sys.stderr)
            return ""

    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout for question: {question[:50]}...", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Warning: Exception for question: {question[:50]}...", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        return ""


def run_benchmark(
    model_name: str,
    adapter_path: Path,
    dataset: CheckboxQADataset,
    output_dir: Path,
    max_docs: int = None
) -> Path:
    """
    Run benchmark for a single model.

    Returns:
        Path to predictions JSONL file
    """
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = model_dir / "predictions.jsonl"

    print(f"\nRunning {model_name}...")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {predictions_file}")

    # Limit documents if specified
    documents = dataset.documents[:max_docs] if max_docs else dataset.documents
    total_questions = sum(len(doc.questions) for doc in documents)

    with open(predictions_file, 'w') as pred_file:
        # Progress bar over all questions
        with tqdm(total=total_questions, desc=f"{model_name}", unit="q") as pbar:
            for doc in documents:
                if doc.pdf_path is None or not doc.pdf_path.exists():
                    print(f"Warning: PDF not found for {doc.document_id}, skipping", file=sys.stderr)
                    pbar.update(len(doc.questions))
                    continue

                # Collect answers for this document
                annotations = []
                for q in doc.questions:
                    output_file = model_dir / f"{doc.document_id}_q{q.id}.txt"

                    answer = run_model_on_document(
                        adapter_path,
                        doc.pdf_path,
                        q.question,
                        output_file
                    )

                    annotations.append({
                        "id": q.id,
                        "key": q.question,
                        "values": [{"value": answer}]
                    })

                    pbar.update(1)

                # Write document predictions
                pred_entry = {
                    "name": doc.document_id,
                    "extension": "pdf",
                    "annotations": annotations
                }
                pred_file.write(json.dumps(pred_entry) + "\n")

    print(f"✓ Predictions saved to {predictions_file}")
    return predictions_file


def main():
    parser = argparse.ArgumentParser(description='Run CheckboxQA benchmark')
    parser.add_argument('models', nargs='+', help='Model names to evaluate')
    parser.add_argument('--adapters-dir', type=Path, default=Path('adapters'),
                        help='Directory containing model adapters')
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                        help='Output directory for results')
    parser.add_argument('--subset', type=Path,
                        help='Path to subset JSONL file (default: use full gold.jsonl)')
    parser.add_argument('--max-docs', type=int,
                        help='[Deprecated] Use --subset instead. Maximum number of documents to process')
    parser.add_argument('--gold', type=Path, default=Path('data/gold.jsonl'),
                        help='Path to gold standard file (used for loading dataset and evaluation)')

    args = parser.parse_args()

    # If subset is specified, use it as gold for both loading and evaluation
    if args.subset:
        if not args.subset.exists():
            print(f"Error: Subset file not found: {args.subset}", file=sys.stderr)
            return 1
        gold_file = args.subset
        print(f"Using subset: {args.subset}")
    else:
        gold_file = args.gold
        if args.max_docs:
            print(f"Warning: --max-docs is deprecated. Use --subset instead.", file=sys.stderr)
            print(f"Creating temporary subset with {args.max_docs} documents...", file=sys.stderr)

    # Load dataset
    print("=" * 80)
    print("CheckboxQA Benchmark")
    print("=" * 80)
    print("\nLoading dataset...")

    # Load from subset if specified, otherwise full gold
    dataset = CheckboxQADataset(use_hf=False, data_dir=gold_file.parent if args.subset else Path('data'))

    # If using subset, load from that file directly
    if args.subset:
        # Reload with correct path
        import json
        dataset.documents = []
        with open(gold_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                questions = []
                for annotation in item["annotations"]:
                    answers = []
                    for value_dict in annotation["values"]:
                        answers.append(value_dict["value"])
                        if "value_variants" in value_dict:
                            answers.extend(value_dict["value_variants"])
                    from dataset_builder import Question
                    questions.append(Question(
                        id=annotation["id"],
                        question=annotation["key"],
                        answers=answers,
                        document_id=item["name"]
                    ))
                pdf_path = Path('data/documents') / f"{item['name']}.{item['extension']}"
                if not pdf_path.exists():
                    pdf_path = None
                from dataset_builder import Document
                dataset.documents.append(Document(
                    document_id=item["name"],
                    pdf_path=pdf_path,
                    questions=questions
                ))

    print(f"Loaded {len(dataset)} documents with {dataset.total_questions()} questions")

    if args.max_docs and not args.subset:
        print(f"Note: Processing only first {args.max_docs} documents (use --subset for better control)")

    # Create timestamped results directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.output_dir / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults directory: {results_dir}")

    # Run each model
    results = {}
    for model_name in args.models:
        adapter_path = args.adapters_dir / f"{model_name}.sh"

        if not adapter_path.exists():
            print(f"Error: Adapter not found: {adapter_path}", file=sys.stderr)
            continue

        try:
            pred_file = run_benchmark(
                model_name,
                adapter_path,
                dataset,
                results_dir,
                max_docs=args.max_docs
            )

            # Evaluate
            print(f"\nEvaluating {model_name}...")
            eval_output = results_dir / model_name / "eval_results.json"

            eval_cmd = [
                "python3", "evaluate.py",
                "--pred", str(pred_file),
                "--gold", str(gold_file),  # Use same gold file as dataset loading
                "--output", str(eval_output)
            ]

            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)

            if eval_result.returncode == 0:
                print(eval_result.stdout)

                # Parse results
                with open(eval_output, 'r') as f:
                    eval_data = json.load(f)
                results[model_name] = eval_data
            else:
                print(f"Error evaluating {model_name}:", file=sys.stderr)
                print(eval_result.stderr, file=sys.stderr)

        except Exception as e:
            print(f"Error running {model_name}: {e}", file=sys.stderr)
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    if results:
        # Sort by ANLS score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['anls_score'], reverse=True)

        print(f"\n{'Model':<20} {'ANLS*':<10} {'Accuracy':<12} {'Total Q'}")
        print("-" * 60)

        for model_name, res in sorted_results:
            anls = res['anls_score']
            acc = res['num_correct'] / res['total_questions'] * 100
            total = res['total_questions']
            print(f"{model_name:<20} {anls:<10.4f} {acc:<12.1f}% {total}")

        # Save leaderboard
        leaderboard_file = results_dir / "leaderboard.json"
        with open(leaderboard_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nLeaderboard saved to {leaderboard_file}")
    else:
        print("No results to display")

    print("=" * 80)


if __name__ == "__main__":
    main()
