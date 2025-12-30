#!/usr/bin/env python3
"""
Phi-3.5-vision-instruct - Microsoft's Multimodal LLM

This script demonstrates multimodal understanding using Phi-3.5-vision-instruct,
a 4.2B parameter model from Microsoft supporting 128K context and strong reasoning.

Model: microsoft/Phi-3.5-vision-instruct
Features: VQA, image captioning, OCR, reasoning, multi-image support
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
from cvlization.paths import (
    resolve_input_path,
    resolve_output_path,
)


# Model configuration
MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
DEFAULT_SAMPLE = "examples/sample.jpg"

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
    Load the Phi-3.5-vision-instruct model and processor.

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

    # Determine attention implementation
    try:
        import flash_attn
        attn_implementation = 'flash_attention_2'
        print("Using Flash Attention 2")
    except ImportError:
        attn_implementation = 'eager'
        print("Using eager attention (Flash Attention not available)")

    # Load processor with num_crops=16 for single images
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=16  # Use 16 for single images, 4 for multi-frame
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation=attn_implementation
    )

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


def run_inference(model, processor, images, prompt: str):
    """
    Run inference on one or more images using Phi-3.5-vision-instruct.

    Args:
        model: Loaded Phi-3.5-vision model
        processor: Model processor
        images: Single PIL Image or list of PIL Images
        prompt: Text prompt/question

    Returns:
        str: Model response text
    """
    print(f"Running inference with prompt: '{prompt[:50]}...'")

    # Ensure images is a list
    if not isinstance(images, list):
        images = [images]

    print(f"Processing {len(images)} image(s)")

    # Build prompt with image placeholders
    image_placeholders = "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
    user_content = f"{image_placeholders}\n{prompt}"

    # Construct chat messages with image placeholder
    messages = [
        {"role": "user", "content": user_content},
    ]

    # Apply chat template
    prompt_text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(prompt_text, images, return_tensors="pt")

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args
        )

    # Decode (skip the input prompt)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
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
        description="Phi-3.5-vision-instruct Multimodal Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image captioning
  python predict.py --image photo.jpg --task caption

  # OCR (text extraction)
  python predict.py --image document.jpg --task ocr

  # Visual question answering
  python predict.py --image chart.png --task vqa --prompt "What is the trend in this chart?"

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

    # Validate image arguments
    if args.image and args.images:
        parser.error("Cannot specify both --image and --images")

    # Handle bundled sample vs user-provided input
    if not args.image and not args.images:
        # Use bundled sample directly
        image_paths = [DEFAULT_SAMPLE]
        print(f"No --image provided, using bundled sample: {DEFAULT_SAMPLE}")
    elif args.images:
        # Multiple user-provided images
        image_paths = [resolve_input_path(img) for img in args.images]
    else:
        # Single user-provided image
        image_paths = [resolve_input_path(args.image)]

    # Resolve output path
    output_path = resolve_output_path(args.output)

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = TASK_PROMPTS.get(args.task, TASK_PROMPTS["caption"])

    print("=" * 60)
    print(f"Phi-3.5-vision-instruct - Microsoft")
    print(f"Model: {args.model_id}")
    print(f"Task: {args.task}")
    print("=" * 60)

    # Load model
    model, processor, device = load_model(args.model_id, args.device)

    # Load images (supports single or multi-image prompts)
    loaded_images = []
    print("\nLoading image(s):")
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"  [{idx}] {image_path}")
        image = load_image(image_path)
        print(f"    Size: {image.size}")
        loaded_images.append(image)

    # Run inference
    print()
    result = run_inference(model, processor, loaded_images, prompt)

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
