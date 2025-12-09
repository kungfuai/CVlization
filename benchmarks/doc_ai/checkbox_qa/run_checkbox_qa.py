#!/usr/bin/env python3
"""
CheckboxQA Benchmark Runner

Runs models on CheckboxQA dataset and evaluates results.

The dataset is automatically downloaded on first use to:
    ~/.cache/cvlization/data/checkbox_qa/

Override with CHECKBOX_QA_CACHE_DIR environment variable.
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

# Use new checkbox_qa package for auto-download support
try:
    from checkbox_qa import load_checkbox_qa, get_cache_dir, CheckboxQADataset
except ImportError:
    # Fallback to legacy dataset_builder
    from dataset_builder import CheckboxQADataset
    get_cache_dir = None
    load_checkbox_qa = None


def get_data_dir() -> Path:
    """Get the data directory, preferring cache over local ./data/."""
    if get_cache_dir:
        cache_dir = get_cache_dir()
        if (cache_dir / "gold.jsonl").exists():
            return cache_dir
    # Fallback to local data directory
    local_data = Path(__file__).parent / "data"
    if (local_data / "gold.jsonl").exists():
        return local_data
    # Return cache dir (will trigger download)
    if get_cache_dir:
        return get_cache_dir()
    return local_data


def check_docker_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# Mapping from adapter names to Docker image names
# (when they differ from the adapter name)
ADAPTER_TO_DOCKER_IMAGE = {
    "qwen3_vl_2b": "qwen3-vl",
    "qwen3_vl_2b_multipage": "qwen3-vl",
    "qwen3_vl_4b_multipage": "qwen3-vl",
    "florence_2": "florence-2",
    "phi_4_multimodal": "phi-4-multimodal",
    "phi_4_multimodal_multipage": "phi-4-multimodal",
}


def check_model_docker_images(model_names: List[str], adapters_dir: Path) -> List[str]:
    """
    Check if Docker images exist for the specified models.

    Returns list of missing image names.
    """
    missing = []
    for model_name in model_names:
        # Get actual Docker image name (may differ from adapter name)
        docker_image = ADAPTER_TO_DOCKER_IMAGE.get(model_name, model_name)
        if not check_docker_image_exists(docker_image):
            missing.append(docker_image)
    return missing


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
    # Determine default data directory
    default_data_dir = get_data_dir()

    parser = argparse.ArgumentParser(
        description='Run CheckboxQA benchmark',
        epilog=f'Data directory: {default_data_dir}'
    )
    parser.add_argument('models', nargs='+', help='Model names to evaluate')
    parser.add_argument('--adapters-dir', type=Path, default=Path('adapters'),
                        help='Directory containing model adapters')
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                        help='Output directory for results')
    parser.add_argument('--subset', type=Path,
                        help='Path to subset JSONL file (default: use full gold.jsonl). '
                             'Can use "dev" or "test" as shortcuts.')
    parser.add_argument('--max-docs', type=int,
                        help='[Deprecated] Use --subset instead. Maximum number of documents to process')
    parser.add_argument('--gold', type=Path,
                        help='Path to gold standard file (default: auto-detect from cache or data/)')
    parser.add_argument('--data-dir', type=Path,
                        help='Data directory containing gold.jsonl and documents/ '
                             f'(default: {default_data_dir})')
    parser.add_argument('--page-cache-dir', type=Path,
                        help='Directory for cached PNG page images (default: disabled - adapters handle conversion)')
    parser.add_argument('--enable-page-cache', action='store_true',
                        help='Enable page caching (requires checkbox_qa Docker image)')

    args = parser.parse_args()

    # Resolve data directory
    data_dir = args.data_dir if args.data_dir else default_data_dir

    # Set page cache - disabled by default since adapters handle PDF->image conversion
    if args.enable_page_cache:
        if args.page_cache_dir is None:
            args.page_cache_dir = Path(os.environ.get('CHECKBOX_QA_PAGE_CACHE', str(data_dir / 'page_images')))
    else:
        args.page_cache_dir = None

    # Handle subset shortcuts
    if args.subset:
        subset_str = str(args.subset)
        if subset_str in ('dev', 'test'):
            args.subset = data_dir / f'subset_{subset_str}.jsonl'
        elif not args.subset.exists():
            # Try in data_dir
            alt_path = data_dir / args.subset.name
            if alt_path.exists():
                args.subset = alt_path

    # Resolve gold file
    if args.gold:
        gold_file = args.gold
    elif args.subset:
        gold_file = args.subset
    else:
        gold_file = data_dir / 'gold.jsonl'

    # Load dataset
    print("=" * 80)
    print("CheckboxQA Benchmark")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")

    # Auto-download if needed
    if not gold_file.exists():
        if load_checkbox_qa:
            print("\nDataset not found. Downloading...")
            try:
                # This triggers download
                _ = load_checkbox_qa(cache_dir=data_dir)
                print("Download complete!")
            except Exception as e:
                print(f"Error downloading dataset: {e}", file=sys.stderr)
                print("Please download manually with: python -m checkbox_qa.dataset --download-only", file=sys.stderr)
                return 1
        else:
            print(f"Error: Gold file not found: {gold_file}", file=sys.stderr)
            print("Please download the dataset first.", file=sys.stderr)
            return 1

    # If subset is specified, use it as gold for both loading and evaluation
    if args.subset:
        if not args.subset.exists():
            print(f"Error: Subset file not found: {args.subset}", file=sys.stderr)
            return 1
        gold_file = args.subset
        print(f"Using subset: {args.subset}")
    else:
        if args.max_docs:
            print(f"Warning: --max-docs is deprecated. Use --subset instead.", file=sys.stderr)
            print(f"Creating temporary subset with {args.max_docs} documents...", file=sys.stderr)

    print("\nLoading dataset...")

    if args.subset:
        dataset = CheckboxQADataset.from_jsonl(gold_file, data_dir=data_dir)
    else:
        dataset = CheckboxQADataset(use_hf=False, data_dir=data_dir)

    print(f"Loaded {len(dataset)} documents with {dataset.total_questions()} questions")

    if args.max_docs and not args.subset:
        print(f"Note: Processing only first {args.max_docs} documents (use --subset for better control)")

    # Check if Docker images exist for all models
    print("\nChecking Docker images...")
    missing_images = check_model_docker_images(args.models, args.adapters_dir)
    if missing_images:
        print("\n" + "=" * 70)
        print("ERROR: Missing Docker images for the following models:")
        print("=" * 70)
        for model in missing_images:
            print(f"  • {model}")
        print("\nBuild the missing images first:")
        for model in missing_images:
            print(f"  cvl run {model} build")
        print("\nOr using shell scripts:")
        for model in missing_images:
            print(f"  cd examples/perception/vision_language/{model} && ./build.sh")
        print("=" * 70)
        return 1
    print("All Docker images found.")

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
