#!/usr/bin/env python3
"""
Granite-Docling - Batch Inference with Multi-Image Support

Process multiple document images with different prompts from a JSONL input file.
Supports multi-page documents by passing multiple images in one request.

Input JSONL schema (per line):
{
    "images": ["page1.png", "page2.png"],  # Required: list of image paths
    "prompt": "Your question here",         # Required: text prompt
    "output": "output.txt",                 # Optional: output file path
    "id": "unique_id"                       # Optional: request identifier
}

Usage:
    python batch_predict.py --batch-input requests.jsonl --output-dir outputs/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


def load_model(model_id: str, device: str = "cuda"):
    """Load Granite-Docling model and processor."""
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        _attn_implementation="sdpa",
    ).to(device)
    return model, processor


def load_images(image_paths: List[str], max_size: int = None) -> List:
    """Load multiple images from paths with optional resizing."""
    from PIL import Image as PILImage
    images = []
    for img_path in image_paths:
        # Load with transformers for consistency
        image = load_image(img_path)

        # Resize if max_size specified
        if max_size is not None and hasattr(image, 'size'):
            width, height = image.size
            if width > max_size or height > max_size:
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                image = image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

        images.append(image)
    return images


def run_inference(
    model,
    processor,
    images: List,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 4096
) -> str:
    """Run inference on multiple images with a prompt."""
    # Build message with multiple images
    content = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # Prepare inputs
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def process_batch(
    model,
    processor,
    batch_input_file: Path,
    output_dir: Path,
    device: str = "cuda",
    max_new_tokens: int = 4096,
    max_image_size: int = None
) -> Dict[str, Any]:
    """Process a batch of inference requests from JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_requests = 0
    successful = 0
    failed = 0
    errors = []

    # Count total requests
    with open(batch_input_file, 'r') as f:
        total_requests = sum(1 for _ in f)

    print(f"Processing {total_requests} requests from {batch_input_file}")

    with open(batch_input_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, total=total_requests, desc="Batch inference"), start=1):
            try:
                request = json.loads(line)

                # Validate required fields
                if "images" not in request:
                    raise ValueError("Missing required field: images")
                if "prompt" not in request:
                    raise ValueError("Missing required field: prompt")

                image_paths = request["images"]
                prompt = request["prompt"]
                request_id = request.get("id", f"request_{line_num}")
                output_file = request.get("output", f"{request_id}.txt")

                # Load images
                print(f"Processing {len(image_paths)} image(s)", flush=True)
                images = load_images(image_paths, max_image_size)

                # Run inference
                output_text = run_inference(
                    model, processor, images, prompt, device, max_new_tokens
                )

                # Save output
                output_path = output_dir / output_file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as out_f:
                    out_f.write(output_text)

                print(f"Output saved to: {output_path}")
                successful += 1

            except Exception as e:
                error_msg = f"Line {line_num}: {str(e)}"
                print(f"Error: {error_msg}", file=sys.stderr)
                errors.append(error_msg)
                failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total requests: {total_requests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")

    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}\n")

    return {
        "total": total_requests,
        "successful": successful,
        "failed": failed,
        "errors": errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="Granite-Docling batch inference with multi-image support"
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
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-granite/granite-docling-258M",
        help="Model to use (default: ibm-granite/granite-docling-258M)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run inference on (default: cuda)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Maximum image dimension (width or height). Images larger than this will be resized maintaining aspect ratio. Default: no resizing"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.batch_input.exists():
        print(f"Error: Batch input file not found: {args.batch_input}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Granite-Docling - Batch Processing")
    print(f"Model: {args.model}")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    model, processor = load_model(args.model, args.device)
    print(f"Model loaded successfully on {args.device}!")
    print()

    # Process batch
    stats = process_batch(
        model,
        processor,
        args.batch_input,
        args.output_dir,
        args.device,
        args.max_new_tokens,
        args.max_image_size
    )

    print("Done!")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
