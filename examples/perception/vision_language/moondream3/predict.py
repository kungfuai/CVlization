#!/usr/bin/env python3
"""
Moondream3 Vision Language Model - Advanced OCR & Visual Reasoning

This script demonstrates OCR, captioning, object detection, and visual question
answering using Moondream3, a 9B parameter MoE vision language model.

Model: https://huggingface.co/moondream/moondream3-preview
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from cvlization.paths import resolve_input_path, resolve_output_path


def ensure_sample_image():
    """Download sample image from HuggingFace if not present."""
    cache_dir = Path.home() / ".cache" / "cvlization" / "moondream3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_path = cache_dir / "sample.jpg"

    if not sample_path.exists():
        print("Downloading sample image from HuggingFace...")
        hf_hub_download(
            repo_id="zzsi/cvl",
            filename="moondream3/sample.jpg",
            repo_type="dataset",
            local_dir=cache_dir,
        )
        # Move from subdirectory to cache root
        src = cache_dir / "moondream3" / "sample.jpg"
        if src.exists():
            src.rename(sample_path)
            # Clean up subdirectory
            (cache_dir / "moondream3").rmdir()

    return str(sample_path)


# Default output path
DEFAULT_OUTPUT = "outputs/result.txt"

# OCR prompts
OCR_PROMPTS = {
    "default": "Transcribe the text",
    "ordered": "Transcribe the text in natural reading order",
    "detailed": "Extract all text from this document with their layout structure",
    "markdown": "Convert this document to markdown format"
}


def load_model(model_id: str, device: str = "cuda", compile_model: bool = True):
    """Load the Moondream3 model."""
    print(f"Loading Moondream3 model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
    except Exception as e:
        error_msg = str(e)
        if "gated repo" in error_msg.lower() or "access is restricted" in error_msg.lower():
            print("\n" + "="*80)
            print("ERROR: Moondream3 is a gated model that requires authentication")
            print("="*80)
            print("\nTo access this model:")
            print("1. Request access: https://huggingface.co/moondream/moondream3-preview")
            print("2. Get your token: https://huggingface.co/settings/tokens")
            print("3. Set environment variable: export HF_TOKEN=your_token")
            print("\n" + "="*80 + "\n")
        raise

    # Compile model for faster inference (optional)
    if compile_model and device == "cuda":
        print("Compiling model for faster inference...")
        model.compile()

    print("Model loaded successfully!")
    print(f"Model size: 9B total params, 2B active (MoE)")
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


def run_ocr(model, image, prompt: str):
    """Run OCR on an image using Moondream3."""
    print("Running OCR...")

    result = model.query(image=image, question=prompt)

    return result["answer"]


def run_caption(model, image, length: str = "normal"):
    """Generate caption for an image."""
    print("Generating caption...")

    # Moondream3 uses query for captioning
    prompts = {
        "short": "Describe this image briefly in one sentence.",
        "normal": "Describe this image.",
        "long": "Provide a detailed description of this image."
    }

    result = model.query(image=image, question=prompts[length])

    return result["answer"]


def run_detect(model, image, object_name: str = None):
    """Detect objects in an image."""
    print("Running object detection...")

    if object_name:
        prompt = f"Detect all instances of {object_name} in this image."
    else:
        prompt = "Detect and list all objects in this image."

    result = model.query(image=image, question=prompt)

    return result["answer"]


def run_point(model, image, object_name: str):
    """Point to an object in an image (return coordinates)."""
    print(f"Pointing to '{object_name}'...")

    prompt = f"Point to the {object_name} in this image."

    result = model.query(image=image, question=prompt)

    return result["answer"]


def save_output(output: str, output_path: str, format: str = "txt", metadata: dict = None):
    """Save the output to a file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as JSON with metadata
        data = {
            "text": output,
            "model": "moondream3-preview",
            "model_size": "9B total, 2B active",
            **(metadata or {})
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_file, "w") as f:
            f.write(output)

    print(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Moondream3 OCR, captioning, detection, and visual reasoning"
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
        default="moondream/moondream3-preview",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["ocr", "caption", "query", "detect", "point"],
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
        choices=["default", "ordered", "detailed", "markdown"],
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
        "--object",
        type=str,
        default=None,
        help="Object name for detect or point tasks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
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
        "--no-compile",
        action="store_true",
        help="Disable model compilation (slower but more compatible)"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_id, args.device, compile_model=not args.no_compile)

    # Load image - use lazy download if default sample or specified image not found
    if args.image is None:
        # Use lazy download from HuggingFace for default sample
        image_path = ensure_sample_image()
        print(f"No --image provided, using bundled sample: {image_path}")
    elif args.image.startswith("http"):
        image_path = args.image
    else:
        image_path = resolve_input_path(args.image)
    print(f"Loading image from {image_path}...")
    image = load_image(image_path)
    print(f"Image loaded: {image.size}")

    # Run task
    metadata = {"task": args.task}

    if args.task == "ocr":
        prompt = args.prompt if args.prompt else OCR_PROMPTS[args.ocr_mode]
        metadata["ocr_mode"] = args.ocr_mode
        output = run_ocr(model, image, prompt)

    elif args.task == "caption":
        metadata["caption_length"] = args.caption_length
        output = run_caption(model, image, args.caption_length)

    elif args.task == "query":
        if not args.prompt:
            print("Error: --prompt is required for query task")
            return
        metadata["prompt"] = args.prompt
        output = run_ocr(model, image, args.prompt)

    elif args.task == "detect":
        if args.object:
            metadata["object"] = args.object
        output = run_detect(model, image, args.object)

    elif args.task == "point":
        if not args.object:
            print("Error: --object is required for point task")
            return
        metadata["object"] = args.object
        output = run_point(model, image, args.object)

    # Print output
    print("\n" + "="*80)
    print(f"{args.task.upper()} OUTPUT:")
    print("="*80)
    print(output)
    print("="*80 + "\n")

    # Save output
    output_path = resolve_output_path(args.output)
    save_output(output, output_path, args.format, metadata)
    print("Done!")


if __name__ == "__main__":
    main()
