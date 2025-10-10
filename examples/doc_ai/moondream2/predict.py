#!/usr/bin/env python3
"""
Moondream2 Vision Language Model - OCR & Document Understanding

This script demonstrates OCR and document understanding using Moondream2,
a compact 1.93B parameter vision language model.
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM


# OCR prompts
OCR_PROMPTS = {
    "default": "Transcribe the text",
    "ordered": "Transcribe the text in natural reading order",
    "detailed": "Extract all text from this document with their layout structure"
}


def load_model(model_id: str, revision: str = "2025-06-21", device: str = "cuda"):
    """Load the Moondream2 model."""
    print(f"Loading Moondream2 model (revision: {revision})...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )

    print("Model loaded successfully!")
    return model


def load_image(image_path: str):
    """Load an image from file path or URL."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        from io import BytesIO
        import requests
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    return image.convert("RGB")


def run_ocr(model, image, prompt: str, stream: bool = False):
    """Run OCR on an image using Moondream2."""
    print("Running OCR...")

    result = model.query(image, prompt)

    return result["answer"]


def run_caption(model, image, length: str = "normal", stream: bool = False):
    """Generate caption for an image."""
    print("Generating caption...")

    result = model.caption(image, length=length, stream=stream)

    if stream:
        full_caption = ""
        for token in result["caption"]:
            print(token, end="", flush=True)
            full_caption += token
        print()  # New line after streaming
        return full_caption
    else:
        return result["caption"]


def save_output(output: str, output_path: str, format: str = "txt"):
    """Save the output to a file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as JSON with metadata
        data = {
            "text": output,
            "model": "moondream2",
            "version": "2025-06-21"
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_file, "w") as f:
            f.write(output)

    print(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Moondream2 OCR and document understanding"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/sample.jpg",
        help="Path to input image or URL"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="vikhyatk/moondream2",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="2025-06-21",
        help="Model revision/version"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["ocr", "caption", "query"],
        default="ocr",
        help="Task to perform"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (for ocr or query tasks)"
    )
    parser.add_argument(
        "--ocr-mode",
        type=str,
        choices=["default", "ordered", "detailed"],
        default="ordered",
        help="OCR mode preset"
    )
    parser.add_argument(
        "--caption-length",
        type=str,
        choices=["short", "normal", "long"],
        default="normal",
        help="Caption length"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "json"],
        default="txt",
        help="Output format"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output for caption task"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_id, args.revision, args.device)

    # Load image
    print(f"Loading image from {args.image}...")
    image = load_image(args.image)
    print(f"Image loaded: {image.size}")

    # Run task
    if args.task == "ocr":
        prompt = args.prompt if args.prompt else OCR_PROMPTS[args.ocr_mode]
        output = run_ocr(model, image, prompt)
    elif args.task == "caption":
        output = run_caption(model, image, args.caption_length, args.stream)
    elif args.task == "query":
        if not args.prompt:
            print("Error: --prompt is required for query task")
            return
        output = run_ocr(model, image, args.prompt)

    # Print output
    if args.task != "caption" or not args.stream:
        print("\n" + "="*80)
        print(f"{args.task.upper()} OUTPUT:")
        print("="*80)
        print(output)
        print("="*80 + "\n")

    # Save output
    save_output(output, args.output, args.format)
    print("Done!")


if __name__ == "__main__":
    main()
