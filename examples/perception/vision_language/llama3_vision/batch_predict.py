#!/usr/bin/env python3
"""
Batch inference script for Llama 3.2 Vision (multi-image, single-turn per request).

Reads a JSONL file with:
  {
    "images": ["/path/to/page1.png", "/path/to/page2.png", ...],
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
from typing import List, Optional

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor, BitsAndBytesConfig

DEFAULT_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"


def detect_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS)")
    else:
        device = "cpu"
        print("Using CPU (no GPU detected)")
    return device


def load_model(model_id: str, device: Optional[str]):
    """Load Llama 3.2 Vision model with BitsAndBytes quantization if needed."""
    if device is None:
        device = detect_device()

    # Use 8-bit quantization for the full precision unsloth model
    use_8bit = model_id == "unsloth/Llama-3.2-11B-Vision-Instruct"

    if use_8bit:
        print(f"Loading {model_id} on {device} with 8-bit quantization...")
    else:
        print(f"Loading {model_id} on {device}...")

    processor = MllamaProcessor.from_pretrained(model_id)

    if device == "cuda":
        if use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
            )
        else:
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            )
    else:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to(device)

    return model, processor, device


def maybe_resize(image: Image.Image, max_dim: int) -> Image.Image:
    """Resize image in-place if larger than max_dim while preserving aspect ratio."""
    if not max_dim:
        return image

    if max(image.size) <= max_dim:
        return image

    # Use thumbnail to preserve aspect ratio and avoid enlarging.
    image = image.copy()
    image.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return image


def build_inputs(processor, images: List[Image.Image], prompt: str, device: str):
    """Build inputs for multi-image inference."""
    # Create content with multiple images
    content = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt.strip()})

    messages = [{"role": "user", "content": content}]

    # Build template text then tokenize with images
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def generate(model, processor, inputs, max_new_tokens: int):
    """Generate text from model."""
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for consistency
        )
    # Remove prompt tokens from generated sequence
    prompt_len = inputs["input_ids"].shape[1]
    generated_text = processor.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return generated_text.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Llama 3.2 Vision batch inference",
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
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Force a specific device (otherwise auto-detect)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Optional max dimension (px). Images larger than this are downscaled before inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = None if args.device == "auto" else args.device

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load batch input
    with open(args.batch_input) as f:
        requests = [json.loads(line) for line in f]

    print(f"\n{'='*60}")
    print(f"Llama 3.2 Vision Batch Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model_id}")
    print(f"Batch input: {args.batch_input}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}")

    # Load model once
    model, processor, device = load_model(args.model_id, device)
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

        # Load images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = maybe_resize(img, args.max_image_size)
                images.append(img)
            except Exception as e:
                print(f"\nWarning: Failed to load {img_path}: {e}", file=sys.stderr)
                continue

        if not images:
            print(f"✗ No valid images for {request_id}", file=sys.stderr)
            continue

        # Run inference
        try:
            inputs = build_inputs(processor, images, prompt, device)
            result = generate(model, processor, inputs, max_new_tokens=args.max_tokens)

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
