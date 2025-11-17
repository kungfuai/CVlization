#!/usr/bin/env python3
"""
Batch inference for MolmoE-1B Vision Language Model.

Processes batches of images with custom prompts from JSONL input.
MolmoE-1B is a 1.2B active parameter MoE model from Allen Institute.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL_ID = "allenai/MolmoE-1B-0924"


def detect_device():
    """Detect the best available device for inference."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS)")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def load_model(model_id: str = MODEL_ID, device: str = None):
    """Load MolmoE-1B model and processor."""
    print(f"Loading {model_id}...")

    if device is None:
        device = detect_device()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True
    )

    print(f"Model loaded successfully on {device}!")
    return model, processor


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
    processor,
    batch_input_file: Path,
    output_dir: Path,
    max_new_tokens: int = 200,
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
    print(f"Max new tokens: {max_new_tokens}")
    print()

    total = len(requests)
    for idx, request in enumerate(requests, 1):
        request_id = request.get("id", f"request_{idx}")
        image_paths = request["images"]
        prompt = request["prompt"]
        output_file = request.get("output", f"{request_id}.txt")

        print(f"[{idx}/{total}] Processing {request_id}...", end=" ", flush=True)

        # Process each image separately (MolmoE processes one at a time)
        responses = []
        for img_path in image_paths:
            # Load image
            image = load_image(img_path, max_image_size)

            # Process inputs
            inputs = processor.process(
                images=[image],
                text=prompt
            )

            # Move to device and make batch of size 1
            inputs = {
                k: v.to(model.device).unsqueeze(0).to(model.dtype)
                if v.dtype.is_floating_point
                else v.to(model.device).unsqueeze(0)
                for k, v in inputs.items()
            }

            # Generate with memory optimization
            with torch.no_grad():
                torch.cuda.empty_cache()  # Clear cache before generation
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )

            # Decode
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            responses.append(generated_text)

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
        description="MolmoE-1B batch inference from JSONL",
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
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)"
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

    args = parser.parse_args()

    print("="*60)
    print("MolmoE-1B Batch Inference")
    print("="*60)
    print(f"Model: {args.model_id}")
    print(f"Batch input: {args.batch_input}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)

    # Load model
    model, processor = load_model(args.model_id, args.device)

    # Process batch
    process_batch(
        model=model,
        processor=processor,
        batch_input_file=args.batch_input,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        max_image_size=args.max_image_size
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
