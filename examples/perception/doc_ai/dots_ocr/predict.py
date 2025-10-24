#!/usr/bin/env python3
"""
dots.ocr Inference Script

This script demonstrates document OCR and layout parsing using the dots.ocr model.
It supports both local images and URLs, with flexible output formats (JSON, Markdown).
Dual-mode execution: standalone or via CVL with --inputs/--outputs.
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


# Default prompt for document layout parsing
DEFAULT_PROMPT = """Parse the image and extract all content including text, tables, and formulas.
Return the result as structured JSON with layout information and reading order."""

# More detailed prompt for comprehensive parsing
DETAILED_PROMPT = """Analyze this document image and extract:
1. All text content with reading order
2. Table structures in markdown format
3. Mathematical formulas in LaTeX
4. Layout information (bounding boxes and categories)

Provide output in structured JSON format."""


def load_model(model_path: str, device: str = "auto"):
    """Load the dots.ocr model and processor."""
    print(f"Loading model from {model_path}...")

    # Check if flash attention is available
    try:
        attn_implementation = "flash_attention_2"
    except:
        attn_implementation = "eager"

    # For HuggingFace model IDs with periods, download snapshot first
    if "/" in model_path and "." in model_path.split("/")[-1]:
        print("Detected model ID with period in name, downloading snapshot...")
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(
            repo_id=model_path,
            local_files_only=False
        )
        print(f"Using snapshot at: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True if not "/" in model_path else False
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True if not "/" in model_path else False
    )

    print("Model loaded successfully!")
    return model, processor


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


def run_inference(model, processor, image_path: str, prompt: str, max_new_tokens: int = 4096):
    """Run OCR inference on an image."""
    print("Running inference...")

    # Prepare messages in chat format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Process inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    # Decode output (trim the input from output)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    return output_text


def save_output(output: str, output_path: str, format: str = "txt"):
    """Save the output to a file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Try to parse as JSON and pretty-print
        try:
            json_data = json.loads(output)
            with open(output_file, "w") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            # If not valid JSON, save as text
            with open(output_file, "w") as f:
                f.write(output)
    else:
        with open(output_file, "w") as f:
            f.write(output)


def main():
    parser = argparse.ArgumentParser(
        description="Run dots.ocr inference on document images"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/sample.jpg",
        help="Path to input image or URL (default: examples/sample.jpg)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="rednote-hilab/dots.ocr",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom prompt for OCR task"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use detailed prompt for comprehensive parsing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/result.{format})"
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
        default=4096,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    # Resolve paths for CVL dual-mode support
    INP = get_input_dir()
    OUT = get_output_dir()

    # Smart default for output path
    if args.output is None:
        ext = {"json": "json", "txt": "txt"}[args.format]
        args.output = f"result.{ext}"

    # Resolve paths using cvlization utilities
    input_path = resolve_input_path(args.image, INP) if not args.image.startswith("http") else args.image
    output_path = Path(resolve_output_path(args.output, OUT))

    # Validate input file (if not URL)
    if not args.image.startswith("http") and not Path(input_path).exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    # Use detailed prompt if requested
    if args.detailed:
        prompt = DETAILED_PROMPT
    else:
        prompt = args.prompt

    # Load model
    model, processor = load_model(args.model_path, args.device)

    # Load image (just to verify it exists and get size)
    print(f"\n{'='*80}")
    print("INPUT")
    print('='*80)
    print(f"Image: {input_path}")
    print('='*80 + '\n')

    print(f"Loading image...")
    image = load_image(str(input_path))
    print(f"Image loaded: {image.size}")

    # Run inference (pass path for process_vision_info)
    output = run_inference(model, processor, str(input_path), prompt, args.max_tokens)

    # Print output
    print("\n" + "="*80)
    print(f"{args.format.upper()} OUTPUT (preview):")
    print("="*80)
    preview = output[:500] + ("..." if len(output) > 500 else "")
    print(preview)
    print("="*80 + "\n")

    # Save output
    save_output(output, str(output_path), args.format)

    # Show container path (CVL will translate to host path)
    print(f"Output saved to {output_path}")
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
