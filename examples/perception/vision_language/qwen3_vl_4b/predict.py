#!/usr/bin/env python3
"""
Qwen3-VL-4B-Instruct - Alibaba Cloud's Advanced Vision Language Model

This script demonstrates multimodal understanding using Qwen3-VL-4B-Instruct from Alibaba,
a 4B parameter model with enhanced visual perception and reasoning capabilities.

Model: Qwen/Qwen3-VL-4B-Instruct
Features: Advanced OCR, VQA, image captioning, visual reasoning, video understanding
License: Apache 2.0
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


# Model configuration
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Task prompts
TASK_PROMPTS = {
    "caption": "Describe this image in detail.",
    "ocr": "Extract all text from this image.",
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
    Load the Qwen3-VL-4B-Instruct model and processor.

    Args:
        model_id: HuggingFace model ID
        device: Device to use (auto-detect if None)

    Returns:
        tuple: (model, processor, device)
    """
    print(f"Loading {model_id}...")

    # Auto-detect device if not specified
    if device is None:
        device = detect_device()

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    print(f"Model loaded successfully on {device}!")
    return model, processor, device


def load_image(image_path: str):
    """
    Load an image from file path or URL.

    Args:
        image_path: Path to image file or URL

    Returns:
        str or PIL.Image: Image URL or loaded image
    """
    # Qwen3-VL can handle both URLs and PIL images
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    else:
        return Image.open(image_path).convert("RGB")


def run_inference(model, processor, images, prompt: str):
    """
    Run inference on one or more images using Qwen3-VL-4B-Instruct.

    Args:
        model: Loaded Qwen3-VL model
        processor: Model processor
        images: Single PIL Image/URL string, or list of PIL Images/URL strings
        prompt: Text prompt/question

    Returns:
        str: Model response text
    """
    print(f"Running inference with prompt: '{prompt[:50]}...'")

    # Ensure images is a list
    if not isinstance(images, list):
        images = [images]

    print(f"Processing {len(images)} image(s)")

    # Construct messages with Qwen3-VL format
    # Build content array with all images followed by the prompt
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Process inputs using chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Move to device
    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512
        )

    # Decode (trim input prompt)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

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
        description="Qwen3-VL-4B-Instruct Vision Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image captioning
  python predict.py --image photo.jpg --task caption

  # OCR (text extraction)
  python predict.py --image document.jpg --task ocr

  # Visual question answering
  python predict.py --image chart.png --task vqa --prompt "What trends are visible?"

  # Save as JSON
  python predict.py --image doc.jpg --task ocr --output result.json --format json
        """
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single input image or URL"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs='+',
        default=None,
        help="Paths to multiple input images (for multi-image tasks)"
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
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (auto-detect if not specified)"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.task == "vqa" and args.prompt is None:
        parser.error("--prompt is required for VQA task")

    # Validate image arguments
    if args.image and args.images:
        parser.error("Cannot specify both --image and --images")
    if not args.image and not args.images:
        # Default to single test image for backward compatibility
        args.image = "test_images/sample.jpg"

    # Resolve paths for CVL compatibility
    try:
        if args.image:
            image_paths = [resolve_input_path(args.image)]
        else:
            image_paths = [resolve_input_path(img) for img in args.images]
        output_path = resolve_output_path(args.output)
    except:
        # Fallback to direct paths if CVL not available
        if args.image:
            image_paths = [args.image]
        else:
            image_paths = args.images
        output_path = args.output

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = TASK_PROMPTS.get(args.task, TASK_PROMPTS["caption"])

    print("=" * 60)
    print(f"Qwen3-VL-4B-Instruct - Alibaba Cloud")
    print(f"Model: {args.model_id}")
    print(f"Task: {args.task}")
    print("=" * 60)

    # Load model
    model, processor, device = load_model(args.model_id, args.device)

    # Load image(s)
    print(f"\nLoading {len(image_paths)} image(s)...")
    images = []
    for img_path in image_paths:
        print(f"  - {img_path}")
        img = load_image(img_path)
        images.append(img)
        if not isinstance(img, str):
            print(f"    Size: {img.size}")

    # Run inference
    print()
    result = run_inference(model, processor, images, prompt)

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
