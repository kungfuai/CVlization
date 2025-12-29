#!/usr/bin/env python3
"""
InternVL3-8B - OpenGVLab's Advanced Vision Language Model

This script demonstrates multimodal understanding using InternVL3-8B from OpenGVLab,
an 8B parameter model with strong multimodal reasoning capabilities.

Model: OpenGVLab/InternVL3-8B
Features: Advanced OCR, VQA, image captioning, visual reasoning, multimodal reasoning
License: MIT
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


# Model configuration
MODEL_ID = "OpenGVLab/InternVL3-8B"
DEFAULT_IMAGE = "test_images/sample.jpg"
DEFAULT_OUTPUT = "outputs/result.txt"


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
    Load the InternVL3-8B model and tokenizer.

    Args:
        model_id: HuggingFace model ID
        device: Device to use (auto-detect if None)

    Returns:
        tuple: (model, tokenizer, device)
    """
    print(f"Loading {model_id}...")

    # Auto-detect device if not specified
    if device is None:
        device = detect_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # Load model
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()

    if device == "cuda":
        model = model.cuda()

    print(f"Model loaded successfully on {device}!")
    return model, tokenizer, device


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


def run_inference(model, tokenizer, image, prompt: str):
    """
    Run inference on an image using InternVL3-8B.

    Args:
        model: Loaded InternVL3 model
        tokenizer: Model tokenizer
        image: PIL Image
        prompt: Text prompt/question

    Returns:
        str: Model response text
    """
    print(f"Running inference with prompt: '{prompt[:50]}...'")

    # Load the image directly - InternVL's load_image expects a path
    # So we'll save temporarily or use the dynamic tiling approach
    # For simplicity, we'll use the model's built-in preprocessing
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    # Use 448 as default input size for InternVL
    transform = build_transform(input_size=448)
    pixel_values = transform(image).unsqueeze(0).to(model.device).to(model.dtype)

    # Construct generation config
    generation_config = dict(
        max_new_tokens=512,
        do_sample=False
    )

    # Run inference using InternVL's chat method
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config
        )

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
        description="InternVL3-8B Vision Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image captioning
  python predict.py --image photo.jpg --prompt "Describe this image in detail."

  # OCR (text extraction)
  python predict.py --image document.jpg --prompt "Extract all text from this image."

  # Visual question answering
  python predict.py --image chart.png --prompt "What trends are visible in this chart?"

  # Save as JSON
  python predict.py --image doc.jpg --prompt "Extract text" --output result.json --format json
        """
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
        default=MODEL_ID,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt/question for the model"
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

    print("=" * 60)
    print(f"InternVL3-8B - OpenGVLab")
    print(f"Model: {args.model_id}")
    print("=" * 60)

    # Load model
    model, tokenizer, device = load_model(args.model_id, args.device)

    # Load image
    print(f"\nLoading image: {image_path}")
    image = load_image(image_path)
    print(f"Image size: {image.size}")

    # Run inference
    print()
    result = run_inference(model, tokenizer, image, args.prompt)

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
