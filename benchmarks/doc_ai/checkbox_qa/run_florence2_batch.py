#!/usr/bin/env python3
"""
Run Florence-2 batch_predict on CheckboxQA dev subset.

Supports both base Florence-2 models and fine-tuned DocVQA variants.
Florence-2-Large has 770M parameters and excels at OCR and document understanding.
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
FLORENCE_DIR = REPO_ROOT / "examples/perception/vision_language/florence_2"

# Import lazy page caching utilities
from page_cache import get_page_images, get_page_cache_root

PAGE_CACHE_ROOT = get_page_cache_root()


def load_checkbox_qa_subset(subset_file: Path) -> List[Dict]:
    """Load CheckboxQA subset JSONL file."""
    documents = []
    with open(subset_file, 'r') as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


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
            image_paths = [f"/page_cache/{doc_id}/{img.name}" for img in page_images]

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
    task_prompt: str = None,
    max_image_size: int = None
) -> int:
    """Run batch_predict.py inside Docker container."""
    batch_input = batch_input.absolute()
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Docker command
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{FLORENCE_DIR}:/workspace",
        "-v", f"{batch_input.parent}:/batch_input:ro",
        "-v", f"{output_dir}:/outputs",
        "-v", f"{PAGE_CACHE_ROOT}:/page_cache:ro",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-w", "/workspace",
        "florence-2",
        "python3", "batch_predict.py",
        "--model-id", model_id,
        "--batch-input", f"/batch_input/{batch_input.name}",
        "--output-dir", "/outputs"
    ]

    if task_prompt:
        cmd.extend(["--task-prompt", task_prompt])

    if max_image_size:
        cmd.extend(["--max-image-size", str(max_image_size)])

    print(f"Running Florence-2 batch_predict in Docker...")
    print(f"  Model: {model_id}")
    print(f"  Batch input: {batch_input}")
    print(f"  Output dir: {output_dir}")
    if task_prompt:
        print(f"  Task prompt: {task_prompt}")
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
            answer = output_file.read_text(encoding='utf-8').strip()

            # For multi-image outputs, Florence-2 might have "Image X:\n..." format
            # Extract the actual answer
            if "\n\nImage " in answer:
                parts = answer.split("\n\nImage ")
                last_response = parts[-1]
                if ":\n" in last_response:
                    answer = last_response.split(":\n", 1)[1].strip()
            elif answer.startswith("Image 1:\n"):
                answer = answer.split(":\n", 1)[1].strip() if ":\n" in answer else answer
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
    # Run evaluate.py as subprocess
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

    # Print evaluation output
    print(result.stdout)

    # Load and return results
    with open(results_file) as f:
        eval_results = json.load(f)

    return eval_results['anls_score']


def main():
    parser = argparse.ArgumentParser(
        description="Run Florence-2 on CheckboxQA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Base Florence-2-large with detailed caption task (1 page)
  ./run_florence2_batch.py --model-id microsoft/Florence-2-large --task-prompt "<DETAILED_CAPTION>" --max-pages 1

  # Fine-tuned DocVQA model (recommended for QA)
  ./run_florence2_batch.py --model-id sahilnishad/Florence-2-FT-DocVQA --task-prompt "<DocVQA>" --max-pages 2

  # With trackio logging
  ./run_florence2_batch.py --model-id microsoft/Florence-2-large --task-prompt "<CAPTION>" --track

  # Custom image size
  ./run_florence2_batch.py --model-id sahilnishad/Florence-2-FT-DocVQA --task-prompt "<DocVQA>" --max-image-size 1024
"""
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/Florence-2-large",
        help="HuggingFace model ID (default: microsoft/Florence-2-large). Try: sahilnishad/Florence-2-FT-DocVQA for DocVQA"
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        default=None,
        help="Task prompt (e.g., '<CAPTION>', '<OCR>', '<DocVQA>'). Required for base models, optional for fine-tuned."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=2,
        help="Maximum pages to include per document (default: 2)"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{question}",
        help="Prompt template with {question} placeholder (default: '{question}')"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Maximum image dimension (width or height). Images larger than this will be resized. Default: no resizing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: results/florence2_{model_variant}_{pages}p)"
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log run to trackio database"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        # Extract model variant from model_id (e.g., "large", "FT-DocVQA")
        model_parts = args.model_id.split("/")[-1].lower().replace("florence-2-", "").replace("florence-", "")
        task_suffix = f"_{args.task_prompt.strip('<>').lower()}" if args.task_prompt else ""
        size_suffix = f"_{args.max_image_size}px" if args.max_image_size else ""
        args.output_dir = SCRIPT_DIR / f"results/florence2_{model_parts}_{args.max_pages}p{task_suffix}{size_suffix}"

    # Setup paths
    subset_file = SCRIPT_DIR / "data/subset_dev.jsonl"
    batch_input_file = SCRIPT_DIR / "data/batch_input_florence2.jsonl"
    predictions_file = args.output_dir / "predictions.jsonl"
    results_file = args.output_dir / "eval_results.json"

    print(f"\n{'='*60}")
    print(f"Florence-2 CheckboxQA Benchmark")
    print(f"{'='*60}")
    print(f"Model: {args.model_id}")
    print(f"Task prompt: {args.task_prompt or 'None (using prompt as-is)'}")
    print(f"Max pages: {args.max_pages}")
    print(f"Prompt template: {args.prompt_template}")
    if args.max_image_size:
        print(f"Max image size: {args.max_image_size}px")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}\n")

    # Prepare trackio if requested
    trackio_run = None
    if args.track:
        if not TRACKIO_AVAILABLE:
            print("Warning: trackio not available, skipping tracking")
            args.track = False

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
        args.task_prompt,
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
        # Extract model name for run naming
        model_variant = args.model_id.split("/")[-1].lower().replace("florence-2-", "").replace("florence-", "")
        task_name = args.task_prompt.strip("<>").lower() if args.task_prompt else "custom"
        size_suffix = f"_{args.max_image_size}px" if args.max_image_size else ""
        run_name = f"florence_{model_variant}_{task_name}_{args.max_pages}p{size_suffix}"

        config = {
            "model": args.model_id,
            "task_prompt": args.task_prompt or "None",
            "max_pages": args.max_pages,
            "prompt_template": args.prompt_template,
            "max_image_size": args.max_image_size or "None",
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
