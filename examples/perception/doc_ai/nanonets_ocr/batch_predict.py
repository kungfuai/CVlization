#!/usr/bin/env python3
"""
Batch inference script for Nanonets-OCR2-3B with VQA support.

Reads a JSONL file with:
  {
    "images": ["/path/to/page1.png"],  # For single-page docs
    "prompt": "Answer this question...",
    "id": "unique_request_id",
    "output": "output_filename.txt"
  }

Processes all requests and saves each output to the specified file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL = "nanonets/Nanonets-OCR2-3B"


def process_document(image_path: str, model, processor, device, prompt: str, max_tokens: int = 4096, max_image_size: int = None):
    """Process a document image with the model."""
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Resize if requested (maintains aspect ratio)
    if max_image_size:
        w, h = image.size
        if max(w, h) > max_image_size:
            if w > h:
                new_w = max_image_size
                new_h = int(h * max_image_size / w)
            else:
                new_h = max_image_size
                new_w = int(w * max_image_size / h)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create messages following nanonets format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    # Trim input tokens
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

    # Decode
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nanonets-OCR2-3B batch inference with VQA support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="HF model ID to load")
    parser.add_argument(
        "--batch-input",
        type=Path,
        required=True,
        help="Input JSONL file with requests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Force a specific device (otherwise auto-detect)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Maximum image dimension (width or height) in pixels. Resize larger images to save memory. Default: no resizing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load batch input
    with open(args.batch_input) as f:
        requests = [json.loads(line) for line in f]

    print(f"\n{'='*60}")
    print(f"Nanonets-OCR2-3B Batch Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model_id}")
    print(f"Batch input: {args.batch_input}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}")

    # Load model once
    print(f"Loading Nanonets-OCR2-3B model: {args.model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    if device == "cpu":
        model = model.to(device)

    print("Model loaded successfully!\n")

    print(f"Processing {len(requests)} requests...")
    print(f"Max tokens: {args.max_tokens}\n")

    # Process each request
    for i, req in enumerate(requests, 1):
        request_id = req["id"]
        image_paths = req["images"]
        prompt = req["prompt"]
        output_filename = req["output"]

        print(f"[{i}/{len(requests)}] Processing {request_id}...", end=" ", flush=True)

        # For now, only process first image (nanonets is single-image)
        if not image_paths:
            print(f"✗ No images for {request_id}", file=sys.stderr)
            continue

        image_path = image_paths[0]  # Take first page only

        # Run inference
        try:
            result = process_document(image_path, model, processor, device, prompt, max_tokens=args.max_tokens, max_image_size=args.max_image_size)

            # Save output
            output_file = args.output_dir / output_filename
            output_file.write_text(result, encoding="utf-8")
            print(f"✓ Saved to {output_filename}")

        except Exception as e:
            print(f"✗ Error: {e}", file=sys.stderr)
            continue

    print(f"\n✓ Completed {len(requests)} requests")
    print(f"  Output directory: {args.output_dir}\n")
    print("Done!")


if __name__ == "__main__":
    main()
