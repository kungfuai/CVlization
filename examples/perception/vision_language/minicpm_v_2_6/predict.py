#!/usr/bin/env python3
"""
MiniCPM-V-2.6 - OpenBMB's Efficient Multimodal Model

This script demonstrates multimodal understanding using MiniCPM-V-2.6,
an 8B parameter model with state-of-the-art OCR and visual understanding.

Model: openbmb/MiniCPM-V-2_6
Features: Best-in-class OCR, VQA, image captioning, multi-image support
License: Apache 2.0
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

# CVL dual-mode execution support
from cvlization.paths import (
    resolve_input_path,
    resolve_output_path,
)


# Model configuration
MODEL_ID = "openbmb/MiniCPM-V-2_6"
DEFAULT_SAMPLE = "examples/sample.jpg"

# Task prompts
TASK_PROMPTS = {
    "caption": "Describe this image in detail.",
    "ocr": "Extract all text from this image in reading order.",
    "vqa": None,  # User provides custom question
}


def detect_device():
    """
    Detect the best available device for inference.

    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name()
        print(f"Using CUDA device: {gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS)")
    else:
        device = "cpu"
        print("Using CPU (no GPU detected)")

    return device


def load_model(model_id: str = MODEL_ID, device: str = None):
    """
    Load the MiniCPM-V-2.6 model and tokenizer.

    Args:
        model_id: HuggingFace model ID
        device: Device to use (auto-detect if None)

    Returns:
        tuple: (model, tokenizer, device)
    """
    print(f"Loading {model_id}...")

    # Auto-detect device if not specified
    if device is None:
        device = detect_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # Load model
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation='sdpa',  # Use SDPA attention
        torch_dtype=torch.bfloat16
    )

    # Move to device
    if device == "cuda":
        model = model.eval().cuda()
    else:
        model = model.eval()

    print(f"Model loaded successfully on {device}!")
    return model, tokenizer, device


def load_image(image_path: str):
    """
    Load an image from file path or URL.

    Args:
        image_path: Path to image file or URL

    Returns:
        PIL.Image: Loaded image in RGB format
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        from io import BytesIO
        import requests
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    return image.convert("RGB")


def run_inference(model, tokenizer, image, prompt: str):
    """
    Run inference on an image using MiniCPM-V-2.6.

    Args:
        model: Loaded MiniCPM-V model
        tokenizer: Model tokenizer
        image: PIL Image
        prompt: Text prompt/question

    Returns:
        str: Model response text
    """
    print(f"Running inference with prompt: '{prompt[:50]}...'")

    # Construct messages in MiniCPM-V format
    msgs = [
        {
            'role': 'user',
            'content': [image, prompt]
        }
    ]

    # Run inference using model's chat method
    response = model.chat(
        image=None,  # Image is passed in msgs
        msgs=msgs,
        tokenizer=tokenizer
    )

    return response


def save_output(output: str, output_path: str, format: str = "txt", model_id: str = MODEL_ID):
    """
    Save the output to a file.

    Args:
        output: Text output to save
        output_path: Path to output file
        format: Output format (txt or json)
        model_id: Model identifier
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as JSON with metadata
        data = {
            "text": output,
            "model": model_id,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Output saved to: {output_file} (JSON)")
    else:
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Output saved to: {output_file}")


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description="MiniCPM-V-2.6 Vision Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image captioning
  python predict.py --image photo.jpg --task caption

  # OCR (text extraction)
  python predict.py --image document.jpg --task ocr

  # Visual question answering
  python predict.py --image chart.png --task vqa --prompt "What is shown in this chart?"

  # Save as JSON
  python predict.py --image doc.jpg --task ocr --output result.json --format json
        """
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image or URL (default: bundled sample)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["caption", "ocr", "vqa"],
        default="caption",
        help="Task to perform"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (required for VQA)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result.txt",
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
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (auto-detect if not specified)"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.task == "vqa" and args.prompt is None:
        parser.error("--prompt is required for VQA task")

    # Handle bundled sample vs user-provided input
    if args.image is None:
        image_path = DEFAULT_SAMPLE
        print(f"No --image provided, using bundled sample: {image_path}")
    elif args.image.startswith("http"):
        image_path = args.image
    else:
        image_path = resolve_input_path(args.image)

    # Resolve output path
    output_path = resolve_output_path(args.output)

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = TASK_PROMPTS.get(args.task, TASK_PROMPTS["caption"])

    print("=" * 60)
    print(f"MiniCPM-V-2.6 - OpenBMB")
    print(f"Model: {args.model_id}")
    print(f"Task: {args.task}")
    print("=" * 60)

    # Load model
    model, tokenizer, device = load_model(args.model_id, args.device)

    # Load image
    print(f"\nLoading image: {image_path}")
    image = load_image(image_path)
    print(f"Image size: {image.size}")

    # Run inference
    print()
    result = run_inference(model, tokenizer, image, prompt)

    # Display result
    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Save output
    print()
    save_output(result, output_path, args.format, args.model_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
