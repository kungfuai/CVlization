#!/usr/bin/env python3
"""
Gemma-3 Vision Language Model - Inference

Run vision-language inference using Google's Gemma-3 multimodal model.
Supports image description, OCR, captioning, and custom queries.

Uses Unsloth for efficient inference with 4-bit quantization.
Reference: https://docs.unsloth.ai/models/gemma-3-how-to-run-and-fine-tune
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


# Model configuration
DEFAULT_MODEL_ID = "unsloth/gemma-3-4b-it"

# Task prompts
TASK_PROMPTS = {
    "describe": "Describe this image in detail.",
    "ocr": "Extract all text from this image.",
    "caption": "Write a short caption for this image.",
    "analyze": "Analyze what is happening in this image.",
}


def load_model(model_id: str = DEFAULT_MODEL_ID, load_in_4bit: bool = True):
    """
    Load the Gemma-3 vision model using Unsloth.

    Args:
        model_id: HuggingFace model ID
        load_in_4bit: Use 4-bit quantization to reduce memory

    Returns:
        tuple: (model, processor)
    """
    from unsloth import FastVisionModel

    print(f"Loading Gemma-3 vision model: {model_id}...")

    model, processor = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    # Set model to inference mode
    FastVisionModel.for_inference(model)

    print("Model loaded successfully!")
    return model, processor


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


def run_inference(
    model,
    processor,
    image,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
):
    """
    Run vision-language inference on an image.

    Args:
        model: Loaded Gemma-3 model
        processor: Model processor
        image: PIL Image
        prompt: User prompt/question about the image
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 recommended by Google)
        top_p: Top-p sampling (0.95 recommended)
        top_k: Top-k sampling (64 recommended)

    Returns:
        str: Generated response
    """
    # Build messages in the format Unsloth expects
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process image and text
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to("cuda")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
        )

    # Decode response (skip the input tokens)
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    return response.strip()


def save_output(output: str, output_path: str, format: str = "txt", metadata: dict = None):
    """
    Save the output to a file.

    Args:
        output: Text output to save
        output_path: Path to output file
        format: Output format (txt or json)
        metadata: Optional metadata to include in JSON output
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "text": output,
            "model": metadata.get("model", "gemma-3") if metadata else "gemma-3",
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
        description="Run Gemma-3 vision-language inference"
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
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID (default: unsloth/gemma-3-4b-it)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["describe", "ocr", "caption", "analyze", "query"],
        default="describe",
        help="Task to perform"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (overrides task preset, required for 'query' task)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (1.0 recommended by Google)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses more VRAM)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "json"],
        default="txt",
        help="Output format"
    )

    args = parser.parse_args()

    # Resolve paths for CVL dual-mode support
    INP = get_input_dir()
    OUT = get_output_dir()

    if args.output is None:
        args.output = "result.txt"

    image_path = resolve_input_path(args.image, INP)
    output_path = resolve_output_path(args.output, OUT)

    # Load model
    model, processor = load_model(
        args.model_id,
        load_in_4bit=not args.no_4bit
    )

    # Load image
    print(f"\n{'='*80}")
    print("INPUT")
    print('='*80)
    print(f"Image: {image_path}")

    image = load_image(image_path)
    print(f"Size:  {image.size[0]}x{image.size[1]} pixels")
    print('='*80 + '\n')

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    elif args.task == "query":
        print("Error: --prompt is required for query task")
        return
    else:
        prompt = TASK_PROMPTS[args.task]

    print(f"Task: {args.task}")
    print(f"Prompt: {prompt}\n")

    # Run inference
    print("Running inference...")
    output = run_inference(
        model,
        processor,
        image,
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Print output
    print("\n" + "="*80)
    print(f"{args.task.upper()} OUTPUT:")
    print("="*80)
    print(output)
    print("="*80 + "\n")

    # Save output
    metadata = {
        "model": args.model_id,
        "task": args.task,
        "prompt": prompt,
    }
    save_output(output, output_path, args.format, metadata)
    print("Done!")


if __name__ == "__main__":
    main()
