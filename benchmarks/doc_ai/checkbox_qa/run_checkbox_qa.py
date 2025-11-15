#!/usr/bin/env python3
"""
CheckboxQA Benchmark Runner

Runs models on CheckboxQA dataset and evaluates results.
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[2]
CHECKBOX_QA_IMAGE = os.environ.get("CHECKBOX_QA_IMAGE", "checkbox_qa")


def ensure_page_cache(pdf_path: Path, doc_id: str, cache_root: Path) -> Path:
    """
    Ensure PNG page cache exists for the given document.

    Returns directory containing cached PNGs.
    """
    cache_dir = cache_root / doc_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    if any(cache_dir.glob("page-*.png")):
        return cache_dir

    if pdf_path is None or not pdf_path.exists():
        return cache_dir

    render_pages_in_docker(pdf_path, doc_id, cache_root)
    return cache_dir


def normalize_answer(answer: str) -> List[str]:
    """
    Convert adapter output into a list of plain strings.
    Handles Python list literals such as ["Yes", "No"].
    """
    stripped = answer.strip()

    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            values: List[str] = []

            def flatten(obj):
                if obj is None:
                    return
                if isinstance(obj, (list, tuple)):
                    for item in obj:
                        flatten(item)
                else:
                    text = str(obj).strip()
                    if text:
                        values.append(text)

            if isinstance(parsed, (list, tuple)):
                flatten(parsed)
                if values:
                    return values
        except (ValueError, SyntaxError):
            pass

    cleaned = stripped.strip().strip('"').strip("'")
    return [cleaned] if cleaned else [answer]


def render_pages_in_docker(pdf_path: Path, doc_id: str, cache_root: Path) -> None:
    """
    Render all PDF pages to PNGs inside the checkbox_qa Docker image.
    """
    pdf_dir = pdf_path.parent.resolve()
    cache_root = cache_root.resolve()

    cache_root.mkdir(parents=True, exist_ok=True)

    python_cmd = (
        "from page_cache import render_pdf_to_images; "
        "from pathlib import Path; "
        f"render_pdf_to_images(Path('/pdfs/{pdf_path.name}'), "
        f"Path('/page_cache/{doc_id}'), overwrite=False)"
    )

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{BENCHMARK_DIR}:/workspace",
        "-v", f"{pdf_dir}:/pdfs:ro",
        "-v", f"{cache_root}:/page_cache",
        "-w", "/workspace",
        CHECKBOX_QA_IMAGE,
        "python", "-c", python_cmd
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error rendering pages for {pdf_path}: {exc}", file=sys.stderr)
        raise


def run_benchmark(
    model_name: str,
    adapter_path: Path,
    dataset: CheckboxQADataset,
    output_dir: Path,
    max_docs: int = None,
    page_cache_dir: Optional[Path] = None,
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

                if page_cache_dir:
                    ensure_page_cache(doc.pdf_path, doc.document_id, page_cache_dir)

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

                    normalized = normalize_answer(answer)
                    annotations.append({
                        "id": q.id,
                        "key": q.question,
                        "values": [{"value": val} for val in normalized]
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
    parser.add_argument('--page-cache-dir', type=Path,
                        default=Path(os.environ.get('CHECKBOX_QA_PAGE_CACHE', 'data/page_images')),
                        help='Directory for cached PNG page images (default: data/page_images)')

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

    if args.subset:
        dataset = CheckboxQADataset.from_jsonl(gold_file, data_dir=Path('data'))
    else:
        dataset = CheckboxQADataset(use_hf=False, data_dir=Path('data'))

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
                max_docs=args.max_docs,
                page_cache_dir=args.page_cache_dir
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
