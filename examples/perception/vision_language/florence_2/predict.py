#!/usr/bin/env python3
"""
Florence-2 Vision Language Model - Microsoft's Unified Vision Foundation Model

This script demonstrates multimodal understanding using Florence-2 (Base & Large) from Microsoft,
covering captioning, OCR, object detection, and more via task prompts.

Features: Captioning, OCR, object detection, segmentation, grounding
License: MIT
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# CVL dual-mode execution support
from cvlization.paths import resolve_input_path, resolve_output_path


# Model configuration
VARIANTS = {
    "base": {
        "model_id": "microsoft/Florence-2-base",
        "description": "0.23B params (~1GB VRAM)",
    },
    "large": {
        "model_id": "microsoft/Florence-2-large",
        "description": "0.77B params (~2GB VRAM)",
    },
}
DEFAULT_VARIANT = "base"
DEFAULT_MODEL_ID = VARIANTS[DEFAULT_VARIANT]["model_id"]
DEFAULT_IMAGE = "examples/sample.jpg"
DEFAULT_OUTPUT = "result.txt"

# Task prompts
TASK_PROMPTS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
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


def load_model(model_id: str = DEFAULT_MODEL_ID, device: str = None):
    """
    Load the Florence-2 model and processor.

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

    # Determine dtype based on device
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    print(f"Model loaded successfully on {device}!")
    return model, processor, device


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


def run_inference(model, processor, image, task_prompt: str, text_input: str = None):
    """
    Run inference on an image using Florence-2.

    Args:
        model: Loaded Florence-2 model
        processor: Model processor
        image: PIL Image
        task_prompt: Task prompt (e.g., "<CAPTION>", "<OCR>")
        text_input: Optional text input for certain tasks

    Returns:
        dict: Model response with parsed results
    """
    print(f"Running inference with task: {task_prompt}")

    # Construct prompt
    if text_input:
        prompt = task_prompt + text_input
    else:
        prompt = task_prompt

    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Move to device with proper dtype
    # Convert floating point tensors to model's dtype
    inputs = {
        k: v.to(model.device).to(model.dtype)
        if v.dtype.is_floating_point
        else v.to(model.device)
        for k, v in inputs.items()
    }

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )

    # Decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-process
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


def format_output(result: dict, task: str) -> str:
    """
    Format the model output for display.

    Args:
        result: Parsed result from model
        task: Task name

    Returns:
        str: Formatted output string
    """
    if not result:
        return "No results"

    # Extract the actual result (Florence-2 returns nested dict)
    task_key = list(result.keys())[0] if result else None
    if not task_key:
        return str(result)

    output = result[task_key]

    # Format based on task type
    if task in ["caption", "detailed_caption", "more_detailed_caption"]:
        return output if isinstance(output, str) else str(output)
    elif task == "ocr":
        if isinstance(output, dict) and 'text' in output:
            return output['text']
        return str(output)
    elif task in ["object_detection", "ocr_with_region", "dense_region_caption", "region_proposal"]:
        # These return structured data with bboxes
        return json.dumps(output, indent=2, ensure_ascii=False)
    else:
        return str(output)


def save_output(output: str, output_path: str, format: str = "txt", model_id: str = DEFAULT_MODEL_ID):
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
        description="Florence-2 Vision Language Model (Base & Large)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image captioning
  python predict.py --image photo.jpg --task caption

  # Detailed captioning
  python predict.py --image photo.jpg --task detailed_caption

  # OCR (text extraction)
  python predict.py --image document.jpg --task ocr

  # Object detection
  python predict.py --image scene.jpg --task object_detection

  # Large variant
  python predict.py --variant large --image photo.jpg --task caption

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
        "--variant",
        type=str,
        choices=sorted(VARIANTS.keys()),
        default=DEFAULT_VARIANT,
        help="Florence-2 checkpoint to load (default: base)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override HuggingFace model ID (defaults to --variant mapping)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(TASK_PROMPTS.keys()),
        default="caption",
        help="Task to perform"
    )
    parser.add_argument(
        "--text-input",
        type=str,
        default=None,
        help="Optional text input for certain tasks"
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
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (auto-detect if not specified)"
    )

    args = parser.parse_args()

    # Resolve paths for CVL compatibility
    # Defaults are local to example dir; user-provided paths resolve to cwd
    if args.image is None:
        image_path = DEFAULT_IMAGE
        print(f"No --image provided, using bundled sample: {image_path}")
    elif args.image.startswith("http"):
        image_path = args.image
    else:
        image_path = resolve_input_path(args.image)
    # Output always resolves to user's cwd
    output_path = resolve_output_path(args.output)

    # Get task prompt
    task_prompt = TASK_PROMPTS[args.task]

    variant_cfg = VARIANTS[args.variant]
    model_id = args.model_id or variant_cfg["model_id"]

    print("=" * 60)
    print("Florence-2 - Microsoft Research")
    print(f"Variant: {args.variant} ({variant_cfg['description']})")
    print(f"Model: {model_id}")
    print(f"Task: {args.task} ({task_prompt})")
    print("=" * 60)

    # Load model
    model, processor, device = load_model(model_id, args.device)

    # Load image
    print(f"\nLoading image: {image_path}")
    image = load_image(image_path)
    print(f"Image size: {image.size}")

    # Run inference
    print()
    result = run_inference(model, processor, image, task_prompt, args.text_input)

    # Format output
    formatted_result = format_output(result, args.task)

    # Display result
    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(formatted_result)
    print("=" * 60)

    # Save output
    print()
    save_output(formatted_result, output_path, args.format, model_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
