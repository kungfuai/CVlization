#!/usr/bin/env python3
"""
Qwen3-VL - Batch Inference

Process multiple images with different prompts from a JSONL input file.

Input JSONL schema (per line):
{
    "images": ["path/to/image1.jpg", "path/to/image2.jpg"],  # Required: list of image paths
    "prompt": "Your question here",                           # Required: text prompt
    "output": "output.txt",                                   # Optional: output file path
    "id": "unique_id"                                         # Optional: request identifier
}

Usage:
    python batch_predict.py --batch-input requests.jsonl --output-dir outputs/ --variant 2b
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

# Import reusable functions from predict.py
from predict import load_model, load_images, run_inference, save_output, MODEL_VARIANTS


def process_batch(
    model,
    processor,
    batch_input_file: Path,
    output_dir: Path,
    max_new_tokens: int = 128,
    default_format: str = "txt",
    model_id: str = None,
    max_image_size: int = None,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = None,
    top_k: int = None
) -> Dict[str, Any]:
    """
    Process a batch of inference requests from JSONL file.

    Args:
        model: Loaded Qwen3-VL model
        processor: Model processor
        batch_input_file: Path to JSONL file with requests
        output_dir: Directory to save outputs
        max_new_tokens: Maximum tokens to generate
        default_format: Default output format if not specified
        model_id: Model identifier for metadata
        max_image_size: Maximum dimension for images (resizes if larger)
        do_sample: Enable sampling (default: False for greedy decoding)
        temperature: Sampling temperature (only used if do_sample=True)

    Returns:
        Dict with processing statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_requests = 0
    successful = 0
    failed = 0
    errors = []

    # Count total requests for progress bar
    with open(batch_input_file, 'r') as f:
        total_requests = sum(1 for _ in f)

    print(f"Processing {total_requests} requests from {batch_input_file}")

    with open(batch_input_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, total=total_requests, desc="Batch inference"), start=1):
            try:
                request = json.loads(line)

                # Validate required fields
                if "images" not in request:
                    raise ValueError("Missing required field: 'images'")
                if "prompt" not in request:
                    raise ValueError("Missing required field: 'prompt'")

                image_paths = request["images"]
                prompt = request["prompt"]
                request_id = request.get("id", f"request_{line_num}")

                # Ensure images is a list
                if not isinstance(image_paths, list):
                    image_paths = [image_paths]

                # Load images (with optional resizing)
                images = load_images(image_paths, max_image_size)

                # Run inference
                result = run_inference(model, processor, images, prompt, max_new_tokens, do_sample, temperature, top_p, top_k)

                # Determine output path
                if "output" in request:
                    output_path = output_dir / request["output"]
                else:
                    output_path = output_dir / f"{request_id}.txt"

                # Determine format
                output_format = request.get("format", default_format)

                # Save result
                save_output(result, output_path, output_format, model_id)

                successful += 1

            except Exception as e:
                failed += 1
                error_msg = f"Line {line_num}: {str(e)}"
                errors.append(error_msg)
                print(f"Error: {error_msg}", file=sys.stderr)
                continue

    # Summary statistics
    stats = {
        "total_requests": total_requests,
        "successful": successful,
        "failed": failed,
        "errors": errors
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Batch Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
JSONL Input Format:
Each line should be a JSON object with:
  - "images": ["path/to/image.jpg"] (required, list of image paths)
  - "prompt": "Your question" (required, text prompt)
  - "output": "result.txt" (optional, output filename)
  - "id": "unique_id" (optional, request identifier)
  - "format": "txt" or "json" (optional, output format)

Example JSONL:
{"images": ["doc1.jpg"], "prompt": "Extract text", "output": "doc1.txt"}
{"images": ["page1.jpg", "page2.jpg"], "prompt": "Summarize", "id": "multi_001"}

Example Usage:
  python batch_predict.py --batch-input requests.jsonl --output-dir results/ --variant 2b
  python batch_predict.py --batch-input requests.jsonl --variant 4b --max-new-tokens 512
        """
    )

    parser.add_argument(
        "--batch-input",
        type=Path,
        required=True,
        help="Path to JSONL file with batch requests"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch"),
        help="Directory to save all outputs"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=list(MODEL_VARIANTS.keys()),
        default="2b",
        help="Model variant (2b, 4b, or 8b)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="Override model ID (instead of using variant)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Maximum tokens to generate (default: varies by variant)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "json"],
        default="txt",
        help="Default output format"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        help="Save batch processing statistics to JSON file"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Maximum image dimension (width or height). Images larger than this will be resized maintaining aspect ratio. Default: no resizing"
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

    # Set seed for reproducibility (commented out for testing)
    # set_seed(42)

    # Validate input file
    if not args.batch_input.exists():
        print(f"Error: Batch input file not found: {args.batch_input}", file=sys.stderr)
        sys.exit(1)

    # Determine model ID and max tokens
    if args.model_id:
        model_id = args.model_id
        max_new_tokens = args.max_new_tokens or 128
    else:
        variant_config = MODEL_VARIANTS[args.variant]
        model_id = variant_config["model_id"]
        max_new_tokens = args.max_new_tokens or variant_config["max_new_tokens"]

    print("=" * 60)
    print(f"Qwen3-VL Batch Processing")
    print(f"Model: {model_id}")
    print(f"Max tokens: {max_new_tokens}")
    print("=" * 60)

    # Load model once for all requests
    print("\nLoading model...")
    model, processor = load_model(model_id, args.device)

    # Process batch
    print()
    if args.max_image_size:
        print(f"Images will be resized to max dimension: {args.max_image_size}px")
    if args.sample:
        params = [f"temperature={args.temperature}"]
        if args.top_p:
            params.append(f"top_p={args.top_p}")
        if args.top_k:
            params.append(f"top_k={args.top_k}")
        print(f"Sampling enabled: {', '.join(params)}")
    else:
        print("Using model defaults (no explicit sampling params)")
    stats = process_batch(
        model,
        processor,
        args.batch_input,
        args.output_dir,
        max_new_tokens,
        args.format,
        model_id,
        args.max_image_size,
        args.sample,
        args.temperature,
        args.top_p,
        args.top_k
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    if stats['errors']:
        print(f"\nErrors encountered:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    print("=" * 60)

    # Save statistics if requested
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {args.stats_output}")

    print(f"\nOutputs saved to: {args.output_dir}")
    print("\nDone!")

    # Exit with error code if any requests failed
    sys.exit(1 if stats['failed'] > 0 else 0)


if __name__ == "__main__":
    main()
