#!/usr/bin/env python3
"""
Phi-4-multimodal-instruct - Batch Inference

Process multiple images with different prompts from a JSONL input file.
Supports native multi-image input for image comparison and reasoning tasks.

Input JSONL schema (per line):
{
    "images": ["path/to/image1.jpg", "path/to/image2.jpg"],  # Required: list of image paths
    "prompt": "Your question here",                           # Required: text prompt
    "output": "output.txt",                                   # Optional: output file path
    "id": "unique_id"                                         # Optional: request identifier
}

Usage:
    python batch_predict.py --batch-input requests.jsonl --output-dir outputs/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

# Import reusable functions from predict.py
from predict import load_model, load_image, run_inference, save_output


def process_batch(
    model,
    processor,
    generation_config,
    batch_input_file: Path,
    output_dir: Path,
    default_format: str = "txt",
    max_image_size: int = None
) -> Dict[str, Any]:
    """
    Process a batch of inference requests from JSONL file.

    Args:
        model: Loaded Phi-4 model
        processor: Model processor
        generation_config: Generation configuration
        batch_input_file: Path to JSONL file with requests
        output_dir: Directory to save outputs
        default_format: Default output format if not specified
        max_image_size: Maximum dimension for images (resizes if larger)

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

                images = request["images"]
                prompt = request["prompt"]
                request_id = request.get("id", f"request_{line_num}")

                # Load images (support both single and multiple images)
                if not isinstance(images, list):
                    images = [images]

                # Load all images (with optional resizing)
                loaded_images = [load_image(img_path, max_image_size) for img_path in images]

                # Pass images to run_inference (handles both single and multi-image natively)
                # For single image, pass as single PIL.Image; for multiple, pass as list
                if len(loaded_images) == 1:
                    result = run_inference(model, processor, generation_config, loaded_images[0], prompt)
                else:
                    result = run_inference(model, processor, generation_config, loaded_images, prompt)

                # Determine output path
                if "output" in request:
                    output_path = output_dir / request["output"]
                else:
                    output_path = output_dir / f"{request_id}.txt"

                # Determine format
                output_format = request.get("format", default_format)

                # Save result
                save_output(result, str(output_path), output_format)

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
        description="Phi-4-multimodal-instruct Batch Inference",
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
  python batch_predict.py --batch-input requests.jsonl --output-dir results/
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
        "--model-id",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="HuggingFace model ID"
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

    args = parser.parse_args()

    # Validate input file
    if not args.batch_input.exists():
        print(f"Error: Batch input file not found: {args.batch_input}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"Phi-4-multimodal-instruct - Batch Processing")
    print(f"Model: {args.model_id}")
    print("=" * 60)

    # Load model once for all requests
    print("\nLoading model...")
    model, processor, generation_config, device = load_model(args.model_id, args.device)

    # Process batch
    print()
    if args.max_image_size:
        print(f"Images will be resized to max dimension: {args.max_image_size}px")
    stats = process_batch(
        model,
        processor,
        generation_config,
        args.batch_input,
        args.output_dir,
        args.format,
        args.max_image_size
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
