#!/usr/bin/env python3
"""
Run Nanonets-OCR2-3B batch_predict on CheckboxQA dev subset.

Nanonets-OCR2-3B is a 3.75B parameter VLM based on Qwen2.5-VL-3B-Instruct.
Specializes in document understanding, OCR, and visual question answering.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# Optional trackio support
try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
NANONETS_DIR = REPO_ROOT / "examples/perception/doc_ai/nanonets_ocr"
PAGE_CACHE_ROOT = SCRIPT_DIR / "data/page_images"


def load_checkbox_qa_subset(subset_file: Path) -> List[Dict]:
    """Load CheckboxQA subset JSONL file."""
    documents = []
    with open(subset_file, 'r') as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def get_page_images(doc_id: str, max_pages: int = 20) -> List[Path]:
    """Get list of page image paths for a document."""
    doc_cache = PAGE_CACHE_ROOT / doc_id
    if not doc_cache.exists():
        print(f"Warning: No page cache for {doc_id}", file=sys.stderr)
        return []

    page_files = sorted(doc_cache.glob("page-*.png"))
    if max_pages and len(page_files) > max_pages:
        page_files = page_files[:max_pages]

    return page_files


def create_batch_input(documents: List[Dict], output_file: Path, prompt_template: str, max_pages: int = 20) -> None:
    """Convert CheckboxQA format to batch_predict JSONL format with container paths."""
    with open(output_file, 'w') as f:
        for doc in documents:
            doc_id = doc["name"]
            page_images = get_page_images(doc_id, max_pages)

            if not page_images:
                print(f"Skipping {doc_id}: no page images", file=sys.stderr)
                continue

            # Use container paths: /page_cache/{doc_id}/page-XXX.png
            # Nanonets processes single-page at a time
            image_paths = [f"/page_cache/{doc_id}/{page_images[0].name}"]  # First page only

            for ann in doc["annotations"]:
                question_id = ann["id"]
                question = ann["key"]

                # Create enhanced prompt
                enhanced_prompt = prompt_template.replace("{question}", question)

                request = {
                    "images": image_paths,
                    "prompt": enhanced_prompt,
                    "id": f"{doc_id}_q{question_id}",
                    "output": f"{doc_id}_q{question_id}.txt",
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "question": question
                }

                f.write(json.dumps(request) + "\n")


def run_batch_predict(
    batch_input: Path,
    output_dir: Path,
    model_id: str,
    max_tokens: int = 512,
    max_image_size: int = None
) -> int:
    """Run batch_predict.py inside Docker container."""
    batch_input = batch_input.absolute()
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Docker command
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{NANONETS_DIR}:/workspace",
        "-v", f"{batch_input.parent}:/batch_input:ro",
        "-v", f"{output_dir}:/outputs",
        "-v", f"{PAGE_CACHE_ROOT}:/page_cache:ro",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-w", "/workspace",
        "nanonets_ocr",
        "python3", "batch_predict.py",
        "--batch-input", f"/batch_input/{batch_input.name}",
        "--output-dir", "/outputs",
        "--max-tokens", str(max_tokens)
    ]

    if max_image_size:
        cmd.extend(["--max-image-size", str(max_image_size)])

    print(f"Running Nanonets-OCR2-3B batch_predict in Docker...")
    print(f"  Model: {model_id}")
    print(f"  Batch input: {batch_input}")
    print(f"  Output dir: {output_dir}")
    print(f"  Max tokens: {max_tokens}")
    if max_image_size:
        print(f"  Max image size: {max_image_size}px")

    result = subprocess.run(cmd)
    return result.returncode


def convert_to_predictions(output_dir: Path, predictions_file: Path, batch_input_file: Path) -> None:
    """Convert batch_predict outputs to CheckboxQA predictions format."""
    # Load request metadata from batch input
    requests = {}
    with open(batch_input_file, 'r') as f:
        for line in f:
            req = json.loads(line)
            requests[req["id"]] = req

    # Group by document
    docs = {}
    for request_id, req in requests.items():
        doc_id = req["doc_id"]
        question_id = req["question_id"]
        question = req["question"]

        # Read model output
        output_file = output_dir / f"{request_id}.txt"
        if output_file.exists():
            answer = output_file.read_text().strip()
        else:
            answer = ""
            print(f"Warning: Missing output for {request_id}", file=sys.stderr)

        if doc_id not in docs:
            docs[doc_id] = {
                "name": doc_id,
                "extension": "pdf",
                "annotations": []
            }

        values = [{"value": answer}] if answer else [{"value": ""}]

        docs[doc_id]["annotations"].append({
            "id": question_id,
            "key": question,
            "values": values
        })

    # Write predictions JSONL (same format as gold)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_file, 'w') as f:
        for doc_id in sorted(docs.keys()):
            docs[doc_id]["annotations"].sort(key=lambda x: x["id"])
            f.write(json.dumps(docs[doc_id]) + "\n")

    print(f"✓ Converted {len(docs)} documents with {sum(len(d['annotations']) for d in docs.values())} predictions to {predictions_file}")


def evaluate_predictions(predictions_file: Path, gold_file: Path, results_file: Path) -> float:
    """Run evaluation using evaluate.py script."""
    cmd = [
        "python3", str(SCRIPT_DIR / "evaluate.py"),
        "--gold", str(gold_file),
        "--pred", str(predictions_file),
        "--output", str(results_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)

    print(result.stdout)

    # Load and return results
    with open(results_file) as f:
        eval_results = json.load(f)

    return eval_results['anls_score']


def main():
    parser = argparse.ArgumentParser(
        description="Run Nanonets-OCR2-3B on CheckboxQA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Nanonets-OCR2-3B with 1 page
  ./run_nanonets_batch.py --max-pages 1 --track

  # With custom prompt and token limit
  ./run_nanonets_batch.py --max-pages 1 --max-tokens 128 --prompt-template "{question}"

  # Full dev subset
  ./run_nanonets_batch.py --max-pages 1 --max-tokens 512
"""
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="nanonets/Nanonets-OCR2-3B",
        help="HuggingFace model ID (default: nanonets/Nanonets-OCR2-3B)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum pages to include per document (default: 1, nanonets is single-image)"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Answer this question using information from the form: {question}\n\nProvide only the answer value. Do not describe the form or repeat the question.",
        help="Prompt template with {question} placeholder"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50, suitable for short answers)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Maximum image dimension (width or height) in pixels. Resize larger images to save memory. Default: no resizing"
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log run to trackio database"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / f"results/nanonets_ocr_{args.max_pages}p"

    # Setup paths
    subset_file = SCRIPT_DIR / "data/subset_dev.jsonl"
    batch_input_file = SCRIPT_DIR / "data/batch_input_nanonets.jsonl"
    predictions_file = args.output_dir / "predictions.jsonl"
    results_file = args.output_dir / "eval_results.json"

    print(f"\n{'='*60}")
    print(f"Nanonets-OCR2-3B CheckboxQA Benchmark")
    print(f"{'='*60}")
    print(f"Model: {args.model_id}")
    print(f"Max pages: {args.max_pages}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Prompt template: {args.prompt_template}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}\n")

    # Step 1: Load CheckboxQA subset
    print("Step 1: Loading CheckboxQA subset...")
    documents = load_checkbox_qa_subset(subset_file)
    print(f"  Loaded {len(documents)} documents")

    # Step 2: Create batch input
    print("\nStep 2: Creating batch input...")
    create_batch_input(documents, batch_input_file, args.prompt_template, args.max_pages)
    print(f"  Created {batch_input_file}")

    # Step 3: Run batch prediction
    print("\nStep 3: Running batch prediction...")
    returncode = run_batch_predict(
        batch_input_file,
        args.output_dir,
        args.model_id,
        args.max_tokens,
        args.max_image_size
    )

    if returncode != 0:
        print(f"Error: batch_predict failed with code {returncode}", file=sys.stderr)
        sys.exit(returncode)

    # Step 4: Convert outputs to predictions
    print("\nStep 4: Converting outputs to predictions...")
    convert_to_predictions(args.output_dir, predictions_file, batch_input_file)

    # Step 5: Evaluate
    print("\nStep 5: Evaluating predictions...")
    anls_score = evaluate_predictions(predictions_file, subset_file, results_file)

    # Log to trackio
    if args.track:
        if not TRACKIO_AVAILABLE:
            print("Warning: trackio not available, skipping tracking")
        else:
            run_name = f"nanonets_ocr_{args.max_pages}p"

            config = {
                "model": args.model_id,
                "max_pages": args.max_pages,
                "prompt_template": args.prompt_template,
                "max_tokens": args.max_tokens,
                "max_image_size": args.max_image_size,
            }

            try:
                run = trackio.init(
                    project="checkbox-qa",
                    name=run_name,
                    config=config
                )
                run.log({"anls_score": anls_score})
                run.finish()
                print(f"✓ Logged to trackio project 'checkbox-qa' as '{run_name}'\n")
            except Exception as e:
                print(f"Warning: Failed to log to trackio: {e}", file=sys.stderr)

    print(f"\n✓ All done! Results saved to {args.output_dir}")
    print(f"  ANLS Score: {anls_score:.4f}")


if __name__ == "__main__":
    main()
