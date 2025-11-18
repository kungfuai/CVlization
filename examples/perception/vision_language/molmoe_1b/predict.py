#!/usr/bin/env python3
"""
MolmoE-1B Vision Language Model - Efficient Multimodal Understanding

This script demonstrates multimodal understanding using MolmoE-1B from Allen Institute (AI2),
a compact 1.2B active parameter MoE model that nearly matches GPT-4V performance.

Model: allenai/MolmoE-1B-0924
Features: OCR, VQA, image captioning, visual reasoning
License: Apache 2.0
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.dynamic_module_utils import get_imports

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


# Model configuration
MODEL_ID = "allenai/MolmoE-1B-0924"


def fixed_get_imports(filename):
    """Workaround: Remove tensorflow from imports (not needed for PyTorch)."""
    imports = get_imports(filename)
    if "tensorflow" in imports:
        imports.remove("tensorflow")
    return imports

# Task prompts
PROMPTS = {
    "ocr": "Extract all text from this image in reading order.",
    "caption": "Describe this image in detail.",
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
    Load the MolmoE-1B model and processor.

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

    # Load with tensorflow workaround
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
            device_map='cuda:0',  # Force GPU (bfloat16: ~13.5GB fits in 23GB L4)
            low_cpu_mem_usage=True
        )

    print(f"Model loaded successfully!")
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


def run_inference(model, processor, image, prompt: str):
    """
    Run inference on an image using MolmoE-1B.

    Args:
        model: Loaded MolmoE-1B model
        processor: Model processor
        image: PIL Image
        prompt: Text prompt/question

    Returns:
        str: Model response text
    """
    print(f"Running inference with prompt: '{prompt[:50]}...'")

    # Process inputs
    inputs = processor.process(
        images=[image],
        text=prompt
    )

    # Move to device and make batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate with memory optimization
    with torch.no_grad():
        torch.cuda.empty_cache()  # Clear cache before generation
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=150, stop_strings="<|endoftext|>"),  # Reduced tokens
            tokenizer=processor.tokenizer
        )

    # Decode
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


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
        description="MolmoE-1B Vision Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OCR on an image
  python predict.py --image document.jpg --task ocr

  # Image captioning
  python predict.py --image photo.jpg --task caption

  # Visual question answering
  python predict.py --image chart.png --task vqa --prompt "What is the trend?"

  # Save as JSON
  python predict.py --image doc.jpg --task ocr --output result.json --format json
        """
    )

    parser.add_argument(
        "--image",
        type=str,
        default="test_images/sample.jpg",
        help="Path to input image or URL"
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
        choices=["ocr", "caption", "vqa"],
        default="ocr",
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

    # Resolve paths for CVL compatibility
    try:
        image_path = resolve_input_path(args.image)
        output_path = resolve_output_path(args.output)
    except:
        # Fallback to direct paths if CVL not available
        image_path = args.image
        output_path = args.output

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = PROMPTS.get(args.task, PROMPTS["caption"])

    print("=" * 60)
    print(f"MolmoE-1B - Allen Institute AI2")
    print(f"Model: {args.model_id}")
    print(f"Task: {args.task}")
    print("=" * 60)

    # Load model
    model, processor, device = load_model(args.model_id, args.device)

    # Load image
    print(f"\nLoading image: {image_path}")
    image = load_image(image_path)
    print(f"Image size: {image.size}")

    # Run inference
    print()
    result = run_inference(model, processor, image, prompt)

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
