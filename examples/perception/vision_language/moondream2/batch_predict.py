#!/usr/bin/env python3
"""
Batch inference for Moondream2 Vision Language Model.

Processes batches of images with custom prompts from JSONL input.
Moondream2 is a compact 1.93B parameter VLM optimized for OCR and document understanding.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"  # Stable revision


def detect_device():
    """Detect the best available device and dtype for inference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        major_cc = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        print(f"Using CUDA: {torch.cuda.get_device_name()} (dtype: {dtype})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("Using Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("Using CPU")
    return device, dtype


def load_model(model_id: str = MODEL_ID, revision: str = REVISION, device: str = None):
    """Load Moondream2 model and tokenizer."""
    print(f"Loading Moondream2 (revision: {revision})...")

    if device is None:
        detected_device, dtype = detect_device()
        device = detected_device
    else:
        device = torch.device(device)
        if device.type == "cuda":
            major_cc = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        else:
            dtype = torch.float32 if device.type == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=dtype
    )
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer


def load_image(image_path: str, max_size: int = None) -> Image.Image:
    """Load an image with optional resizing."""
    image = Image.open(image_path).convert("RGB")

    if max_size is not None:
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def process_batch(
    model,
    tokenizer,
    batch_input_file: Path,
    output_dir: Path,
    max_image_size: int = None
) -> Dict[str, Any]:
    """Process a batch of images from JSONL input."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read batch input
    with open(batch_input_file) as f:
        requests = [json.loads(line) for line in f]

    print(f"\nProcessing {len(requests)} requests...")
    if max_image_size:
        print(f"Max image size: {max_image_size}px")
    print()

    total = len(requests)
    for idx, request in enumerate(requests, 1):
        request_id = request.get("id", f"request_{idx}")
        image_paths = request["images"]
        prompt = request["prompt"]
        output_file = request.get("output", f"{request_id}.txt")

        print(f"[{idx}/{total}] Processing {request_id}...", end=" ", flush=True)

        # Process each image separately (moondream2 doesn't support multi-image)
        responses = []
        for img_path in image_paths:
            # Load image
            image = load_image(img_path, max_image_size)

            # Encode image
            image_embeds = model.encode_image(image)

            # Get answer
            with torch.no_grad():
                answer = model.answer_question(image_embeds, prompt, tokenizer)

            responses.append(answer)

        # Combine responses if multiple images
        if len(responses) > 1:
            final_response = "\n\n".join(f"Page {i+1}:\n{resp}" for i, resp in enumerate(responses))
        else:
            final_response = responses[0]

        # Save output
        output_path = output_dir / output_file
        output_path.write_text(final_response, encoding='utf-8')

        print(f"✓ Saved to {output_file}")

    print(f"\n✓ Completed {total} requests")
    print(f"  Output directory: {output_dir}")

    return {
        "total_requests": total,
        "output_dir": str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Moondream2 batch inference from JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
JSONL Input Format:
Each line should be a JSON object with:
  {
    "images": ["path/to/image1.png", "path/to/image2.png"],  // Multiple images OK
    "prompt": "Your question here",
    "id": "unique_request_id",
    "output": "output_filename.txt"
  }

Example:
  python batch_predict.py \\
    --batch-input batch_requests.jsonl \\
    --output-dir outputs/
"""
    )

    parser.add_argument(
        "--batch-input",
        type=Path,
        required=True,
        help="Path to JSONL file with batch inputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Maximum image dimension. Images larger than this will be resized maintaining aspect ratio. Default: no resizing"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=REVISION,
        help=f"Model revision (default: {REVISION})"
    )

    args = parser.parse_args()

    print("="*60)
    print("Moondream2 Batch Inference")
    print("="*60)
    print(f"Model: {args.model_id} (revision: {args.revision})")
    print(f"Batch input: {args.batch_input}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)

    # Load model
    model, tokenizer = load_model(args.model_id, args.revision, args.device)

    # Process batch
    process_batch(
        model=model,
        tokenizer=tokenizer,
        batch_input_file=args.batch_input,
        output_dir=args.output_dir,
        max_image_size=args.max_image_size
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
