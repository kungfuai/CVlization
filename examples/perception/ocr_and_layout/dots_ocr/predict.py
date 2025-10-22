#!/usr/bin/env python3
"""
dots.ocr Inference Script

This script demonstrates document OCR and layout parsing using the dots.ocr model.
It supports both local images and URLs, with flexible output formats (JSON, Markdown).
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info


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

    print(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run dots.ocr inference on document images"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/sample.jpg",
        help="Path to input image or URL"
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

    # Use detailed prompt if requested
    if args.detailed:
        prompt = DETAILED_PROMPT
    else:
        prompt = args.prompt

    # Load model
    model, processor = load_model(args.model_path, args.device)

    # Load image (just to verify it exists and get size)
    print(f"Loading image from {args.image}...")
    image = load_image(args.image)
    print(f"Image loaded: {image.size}")

    # Run inference (pass path for process_vision_info)
    output = run_inference(model, processor, args.image, prompt, args.max_tokens)

    # Print output
    print("\n" + "="*80)
    print("OCR OUTPUT:")
    print("="*80)
    print(output)
    print("="*80 + "\n")

    # Save output
    save_output(output, args.output, args.format)
    print("Done!")


if __name__ == "__main__":
    main()
