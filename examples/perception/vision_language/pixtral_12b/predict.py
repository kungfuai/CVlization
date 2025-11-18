#!/usr/bin/env python3
"""
Pixtral 12B Vision Language Model - Efficient Multimodal Understanding

This script demonstrates multimodal understanding using Pixtral 12B from Mistral AI,
a 12B parameter model with 4-bit quantization via Unsloth for efficient inference.

Model: unsloth/Pixtral-12B-2409
Features: OCR, VQA, image captioning, visual reasoning
License: Apache 2.0
"""

import argparse
import json
from pathlib import Path
from typing import List

from PIL import Image
from unsloth import FastVisionModel

from cvlization.paths import (
    resolve_input_path,
    resolve_output_path,
)


# Model configuration
MODEL_ID = "unsloth/Pixtral-12B-2409"

# Task prompts
TASK_PROMPTS = {
    "ocr": "Extract all text from this image in reading order.",
    "caption": "Describe this image in detail.",
    "vqa": None,  # User provides custom question
}


def load_model(model_id: str = MODEL_ID):
    """
    Load the Pixtral 12B model using Unsloth with 4-bit quantization.

    Args:
        model_id: HuggingFace model ID

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading {model_id} with 4-bit quantization...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=True,
    )

    # Prepare for inference
    FastVisionModel.for_inference(model)

    print(f"Model loaded successfully!")
    return model, tokenizer


def run_inference(model, tokenizer, images: List[Image.Image], prompt: str, max_tokens: int = 512) -> str:
    """
    Run inference on images using Pixtral 12B.

    Args:
        model: Loaded Pixtral model
        tokenizer: Model tokenizer
        images: List of PIL Images (supports multiple images)
        prompt: Text prompt/question
        max_tokens: Maximum tokens to generate

    Returns:
        str: Model response text
    """
    print(f"Running inference with {len(images)} image(s) and prompt: '{prompt[:50]}...'")

    # Prepare messages with support for multiple images
    # Pixtral supports multiple images natively
    content = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    # Tokenize inputs with all images
    inputs = tokenizer(
        images,  # Pass all images
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        repetition_penalty=1.2,
    )

    # Decode only the new tokens (skip the input)
    input_length = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0][input_length:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return result


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
        description="Pixtral 12B Vision Language Model (Unsloth 4-bit)",
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
        default=None,
        help="Path to input image"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=None,
        help="Paths to multiple input images (Pixtral supports multi-image inference)"
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
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.task == "vqa" and args.prompt is None:
        parser.error("--prompt is required for VQA task")

    if args.image is None and args.images is None:
        parser.error("Either --image or --images must be provided")

    # Collect image paths
    if args.images is not None:
        image_paths = args.images
    else:
        image_paths = [args.image]

    # Resolve paths for CVL compatibility
    try:
        image_paths = [resolve_input_path(p) for p in image_paths]
        output_path = resolve_output_path(args.output)
    except:
        # Fallback to direct paths if CVL not available
        output_path = args.output

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = TASK_PROMPTS.get(args.task, TASK_PROMPTS["caption"])

    print("=" * 60)
    print(f"Pixtral 12B - Mistral AI (Unsloth 4-bit)")
    print(f"Model: {args.model_id}")
    print(f"Task: {args.task}")
    print(f"Images: {len(image_paths)}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args.model_id)

    # Load images
    print(f"\nLoading {len(image_paths)} image(s)...")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    for i, img in enumerate(images):
        print(f"  Image {i+1}: {img.size}")

    # Run inference
    print()
    result = run_inference(model, tokenizer, images, prompt, args.max_tokens)

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
