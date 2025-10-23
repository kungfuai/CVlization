#!/usr/bin/env python3
"""
Moondream2 Vision Language Model - OCR & Document Understanding

This script demonstrates OCR and document understanding using Moondream2,
a compact 1.93B parameter vision language model.

Simple implementation using transformers AutoModelForCausalLM with trust_remote_code=True.
"""

import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model configuration
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"  # Stable revision with best OCR quality

# OCR prompts
OCR_PROMPTS = {
    "default": "What text is in this image?",
    "ordered": "Extract all text from this document in reading order",
    "detailed": "Extract all text from this document with their exact layout and structure"
}


def detect_device():
    """
    Detect the best available device and dtype for inference.

    Returns:
        tuple: (device, dtype)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("Using Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("Using CPU (no GPU detected)")

    return device, dtype


def load_model(model_id: str = MODEL_ID, revision: str = REVISION, device: str = None):
    """
    Load the Moondream2 model.

    Args:
        model_id: HuggingFace model ID
        revision: Model revision/version
        device: Device to use (auto-detect if None)

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading Moondream2 (revision: {revision})...")

    # Auto-detect device if not specified
    if device is None:
        detected_device, dtype = detect_device()
        device = detected_device
    else:
        device = torch.device(device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        print(f"Using device: {device} with dtype: {dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    # Load model with trust_remote_code to get implementation from HuggingFace
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


def run_ocr(model, tokenizer, image, prompt: str):
    """
    Run OCR on an image using Moondream2.

    Args:
        model: Loaded Moondream2 model
        tokenizer: Model tokenizer
        image: PIL Image
        prompt: OCR prompt/question

    Returns:
        str: OCR result text
    """
    print("Running OCR...")

    # Encode image
    image_embeds = model.encode_image(image)

    # Get answer using old API (most reliable for OCR)
    answer = model.answer_question(image_embeds, prompt, tokenizer)

    return answer


def run_caption(model, tokenizer, image, length: str = "normal"):
    """
    Generate caption for an image.

    Args:
        model: Loaded Moondream2 model
        tokenizer: Model tokenizer
        image: PIL Image
        length: Caption length (short, normal, long)

    Returns:
        str: Generated caption
    """
    print("Generating caption...")

    # Use old API with images list
    result = model.caption(images=[image], tokenizer=tokenizer)

    return result[0] if isinstance(result, list) else result


def save_output(output: str, output_path: str, format: str = "txt"):
    """
    Save the output to a file.

    Args:
        output: Text output to save
        output_path: Path to output file
        format: Output format (txt or json)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as JSON with metadata
        data = {
            "text": output,
            "model": "moondream2",
            "revision": REVISION
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_file, "w") as f:
            f.write(output)

    print(f"Output saved to {output_file}")


def main():
    import os

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
        default=MODEL_ID,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=REVISION,
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
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to run inference on (auto-detect if not specified)"
    )

    args = parser.parse_args()

    # Resolve paths for CVL dual-mode support
    # If CVL_INPUTS/CVL_OUTPUTS are set (running via cvl run), use them as base
    cvl_inputs = os.getenv('CVL_INPUTS')
    cvl_outputs = os.getenv('CVL_OUTPUTS')

    # Resolve image path
    if cvl_inputs and not args.image.startswith(('http://', 'https://', '/')):
        # Relative path + CVL_INPUTS set → resolve against CVL_INPUTS
        image_path = str(Path(cvl_inputs) / args.image)
    else:
        # Absolute path, URL, or no CVL_INPUTS → use as-is
        image_path = args.image

    # Resolve output path
    if cvl_outputs and not args.output.startswith('/'):
        # Relative path + CVL_OUTPUTS set → resolve against CVL_OUTPUTS
        output_path = str(Path(cvl_outputs) / args.output)
    else:
        # Absolute path or no CVL_OUTPUTS → use as-is
        output_path = args.output

    # Load model
    model, tokenizer = load_model(args.model_id, args.revision, args.device)

    # Load image
    print(f"Loading image from {image_path}...")
    image = load_image(image_path)
    print(f"Image loaded: {image.size}")

    # Run task
    if args.task == "ocr":
        prompt = args.prompt if args.prompt else OCR_PROMPTS[args.ocr_mode]
        output = run_ocr(model, tokenizer, image, prompt)
    elif args.task == "caption":
        output = run_caption(model, tokenizer, image, args.caption_length)
    elif args.task == "query":
        if not args.prompt:
            print("Error: --prompt is required for query task")
            return
        output = run_ocr(model, tokenizer, image, args.prompt)

    # Print output
    print("\n" + "="*80)
    print(f"{args.task.upper()} OUTPUT:")
    print("="*80)
    print(output)
    print("="*80 + "\n")

    # Save output
    save_output(output, output_path, args.format)
    print("Done!")


if __name__ == "__main__":
    main()
