#!/usr/bin/env python3
"""
Phi-4-multimodal-instruct - Microsoft's Advanced Multimodal Language Model

This script demonstrates multimodal understanding using Phi-4-multimodal-instruct from Microsoft,
a 5.6B parameter model with vision, speech, and text capabilities.

Model: microsoft/Phi-4-multimodal-instruct
Features: Image understanding, OCR, chart/table understanding, multi-image comparison, speech recognition
License: MIT
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


# Model configuration
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"


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
    Load the Phi-4-multimodal-instruct model and processor.

    Args:
        model_id: HuggingFace model ID
        device: Device to use (auto-detect if None)

    Returns:
        tuple: (model, processor, generation_config, device)
    """
    print(f"Loading {model_id}...")

    # Auto-detect device if not specified
    if device is None:
        device = detect_device()

    # Determine attention implementation via native PyTorch support
    attn_implementation = "eager"
    if device == "cuda":
        attn_implementation = "sdpa"
        print("Using PyTorch SDPA attention kernels")
    else:
        print("Running on CPU/MPS, using eager attention")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device if device == "cuda" else "auto",
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation=attn_implementation
    )

    if device == "cuda":
        model = model.cuda()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_id)

    print(f"Model loaded successfully on {device}!")
    return model, processor, generation_config, device


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


def run_inference(model, processor, generation_config, image, prompt: str):
    """
    Run inference on an image using Phi-4-multimodal-instruct.

    Args:
        model: Loaded Phi-4 model
        processor: Model processor
        generation_config: Generation configuration
        image: PIL Image
        prompt: Text prompt/question

    Returns:
        str: Model response text
    """
    print(f"Running inference with prompt: '{prompt[:50]}...'")

    # Format prompt with Phi-4 chat template
    # <|user|><|image_1|>{prompt}<|end|><|assistant|>
    formatted_prompt = f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"

    # Process inputs
    inputs = processor(
        text=formatted_prompt,
        images=image,
        return_tensors='pt'
    ).to(model.device)

    # Generate
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=generation_config
        )

    # Decode (trim input prompt)
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
        description="Phi-4-multimodal-instruct Vision Language Model",
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
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt/question for the model"
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

    # Resolve paths for CVL compatibility
    try:
        image_path = resolve_input_path(args.image)
        output_path = resolve_output_path(args.output)
    except:
        # Fallback to direct paths if CVL not available
        image_path = args.image
        output_path = args.output

    print("=" * 60)
    print(f"Phi-4-multimodal-instruct - Microsoft")
    print(f"Model: {args.model_id}")
    print("=" * 60)

    # Load model
    model, processor, generation_config, device = load_model(args.model_id, args.device)

    # Load image
    print(f"\nLoading image: {image_path}")
    image = load_image(image_path)
    print(f"Image size: {image.size}")

    # Run inference
    print()
    result = run_inference(model, processor, generation_config, image, args.prompt)

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
