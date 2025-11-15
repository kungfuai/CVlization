#!/usr/bin/env python3
"""
Run Qwen3-VL 2B batch_predict on CheckboxQA dev subset

Similar to run_phi4_batch.py but for Qwen3-VL.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
QWEN3_DIR = REPO_ROOT / "examples/perception/vision_language/qwen3_vl"
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
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "question": question
                }

                f.write(json.dumps(request) + "\n")


def run_batch_predict(batch_input: Path, output_dir: Path, variant: str = "2b", max_image_size: int = None, sample: bool = False, temperature: float = 0.2, top_p: float = None, top_k: int = None) -> int:
    """Run batch_predict.py inside Docker container."""
    batch_input = batch_input.absolute()
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Docker command
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{QWEN3_DIR}:/workspace",
        "-v", f"{batch_input.parent}:/batch_input:ro",
        "-v", f"{output_dir}:/outputs",
        "-v", f"{PAGE_CACHE_ROOT}:/page_cache:ro",
        "-v", f"{REPO_ROOT}:/cvlization_repo:ro",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-e", "PYTHONPATH=/cvlization_repo",
        "-w", "/workspace",
        "qwen3-vl",
        "python3", "batch_predict.py",
        "--batch-input", f"/batch_input/{batch_input.name}",
        "--output-dir", "/outputs",
        "--variant", variant
    ]

    if max_image_size:
        cmd.extend(["--max-image-size", str(max_image_size)])

    if sample:
        cmd.append("--sample")
        cmd.extend(["--temperature", str(temperature)])
        if top_p is not None:
            cmd.extend(["--top-p", str(top_p)])
        if top_k is not None:
            cmd.extend(["--top-k", str(top_k)])

    print(f"Running Qwen3-VL {variant} batch_predict in Docker...")
    print(f"  Batch input: {batch_input}")
    print(f"  Output dir: {output_dir}")
    if max_image_size:
        print(f"  Max image size: {max_image_size}px")
    if sample:
        params = [f"temperature={temperature}"]
        if top_p is not None:
            params.append(f"top_p={top_p}")
        if top_k is not None:
            params.append(f"top_k={top_k}")
        print(f"  Sampling: enabled ({', '.join(params)})")
    else:
        print(f"  Decoding: model defaults")
    result = subprocess.run(cmd)
    return result.returncode


def convert_to_predictions(batch_input: Path, output_dir: Path, predictions_file: Path) -> None:
    """Convert batch_predict outputs back to CheckboxQA predictions format."""
    requests = {}
    with open(batch_input, 'r') as f:
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

    # Write predictions JSONL
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_file, 'w') as f:
        for doc_id in sorted(docs.keys()):
            docs[doc_id]["annotations"].sort(key=lambda x: x["id"])
            f.write(json.dumps(docs[doc_id]) + "\n")

    print(f"✓ Predictions saved to {predictions_file}")


def run_evaluation(predictions_file: Path, gold_file: Path, eval_output: Path) -> None:
    """Run CheckboxQA evaluation."""
    cmd = [
        "python3", str(SCRIPT_DIR / "evaluate.py"),
        "--pred", str(predictions_file),
        "--gold", str(gold_file),
        "--output", str(eval_output)
    ]

    print(f"\nRunning evaluation...")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL batch_predict on CheckboxQA subset"
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=SCRIPT_DIR / "data/subset_dev.jsonl",
        help="CheckboxQA subset JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results/qwen3_vl_2b_batch",
        help="Output directory for results"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Look carefully at this form/document image and answer concisely (Yes/No or brief text). {question}",
        help="Prompt template (use {question} placeholder)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum pages per document"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["2b", "4b", "8b"],
        default="2b",
        help="Qwen3-VL model variant"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary batch input file"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Maximum image dimension (width or height) in pixels. Resize larger images to save memory. Default: no resizing"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling (default: use model defaults)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (only used if --sample is set, default: 0.2)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (only used if --sample is set)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (only used if --sample is set)"
    )

    args = parser.parse_args()

    if not args.subset.exists():
        print(f"Error: Subset file not found: {args.subset}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(f"Qwen3-VL {args.variant.upper()} Batch Evaluation on CheckboxQA")
    print("=" * 80)
    print(f"Subset: {args.subset}")
    print(f"Output: {args.output_dir}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load subset
    print("Loading CheckboxQA subset...")
    documents = load_checkbox_qa_subset(args.subset)
    print(f"Loaded {len(documents)} documents")

    # Create batch input
    batch_input = args.output_dir / "batch_input.jsonl"
    print(f"\nCreating batch input: {batch_input}")
    create_batch_input(documents, batch_input, args.prompt_template, args.max_pages)

    # Count requests
    with open(batch_input) as f:
        num_requests = sum(1 for _ in f)
    print(f"Created {num_requests} inference requests")

    # Run batch prediction
    print("\nRunning batch prediction...")
    batch_output_dir = args.output_dir / "batch_outputs"
    returncode = run_batch_predict(batch_input, batch_output_dir, args.variant, args.max_image_size, args.sample, args.temperature, args.top_p, args.top_k)

    if returncode != 0:
        print(f"Warning: batch_predict exited with code {returncode}", file=sys.stderr)

    # Convert to predictions format
    print("\nConverting to CheckboxQA predictions format...")
    predictions_file = args.output_dir / "predictions.jsonl"
    convert_to_predictions(batch_input, batch_output_dir, predictions_file)

    # Run evaluation
    eval_output = args.output_dir / "eval_results.json"
    run_evaluation(predictions_file, args.subset, eval_output)

    # Clean up temp file
    if not args.keep_temp:
        batch_input.unlink()
        print(f"\nCleaned up temporary file: {batch_input}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Predictions: {predictions_file}")
    print(f"Evaluation: {eval_output}")


if __name__ == "__main__":
    main()
