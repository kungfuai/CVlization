#!/usr/bin/env python3
"""
CHURRO-3B - Historical Document OCR

Handwritten and printed text recognition across 22 centuries and 46 language clusters.
Specializes in historical documents with exceptional accuracy at 15.5× lower cost than Gemini 2.5 Pro.

Features:
- Historical document transcription (handwritten and printed)
- 46 language clusters including historical and dead languages
- XML-structured output with layout preservation
- Optional neural binarization for degraded documents
- Up to 20,000 tokens for complete page transcription

Model: stanford-oval/churro-3B (based on Qwen2.5-VL-3B-Instruct)
Repository: https://github.com/stanford-oval/Churro
License: qwen-research
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# CVL dual-mode execution support
try:
    from cvlization.paths import (
        get_input_dir,
        get_output_dir,
        resolve_input_path,
        resolve_output_path,
    )
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_input_dir():
        return os.getcwd()

    def get_output_dir():
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)

    def resolve_output_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)


MODEL_ID = "stanford-oval/churro-3B"


def detect_device() -> tuple:
    """Auto-detect device and dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        # Use bfloat16 for modern GPUs
        major_cc = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        print(f"Using CUDA device with {'bfloat16' if major_cc >= 8 else 'float16'}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Using MPS device with float16")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU with float32")

    return device, dtype


def load_model(model_id: str = MODEL_ID, device: Optional[str] = None):
    """Load CHURRO model and processor."""
    if device is None:
        device, dtype = detect_device()
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading {model_id}...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    if device == "cuda":
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)

    print(f"Model loaded successfully on {device}")
    return model, processor


def load_image(image_path: str) -> Image.Image:
    """Load image from file or URL."""
    if image_path.startswith(("http://", "https://")):
        from io import BytesIO
        import requests
        response = requests.get(image_path)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    return image.convert("RGB")


def strip_xml_tags(text: str) -> str:
    """
    Strip XML tags from CHURRO output.

    Args:
        text: Raw output with XML markup

    Returns:
        Plain text without XML tags
    """
    # Remove XML tags but keep content
    text = re.sub(r'<[^>]+>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def run_inference(
    model,
    processor,
    image: Image.Image,
    max_new_tokens: int = 20000,
    strip_xml: bool = False
) -> str:
    """
    Run OCR inference on historical document.

    Args:
        model: Loaded CHURRO model
        processor: Loaded processor
        image: PIL Image to process
        max_new_tokens: Maximum tokens (20000 for full pages)
        strip_xml: Whether to remove XML markup from output

    Returns:
        Transcribed text (with or without XML)
    """
    print(f"Processing image (size: {image.size})...")

    # Prepare conversation format for Qwen2VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Transcribe this historical document."}
            ]
        }
    ]

    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate transcription
    print(f"Generating transcription (max {max_new_tokens} tokens)...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None
        )

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    if strip_xml:
        output_text = strip_xml_tags(output_text)

    return output_text


def save_output(text: str, output_path: str, format: str = "txt", metadata: dict = None):
    """Save transcribed text to file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "text": text,
            "model": MODEL_ID,
            **(metadata or {})
        }
        output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        output_file.write_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="CHURRO-3B: Historical document OCR for 22 centuries and 46 languages",
        epilog="""
Examples:
  # Basic transcription
  python predict.py --image historical_manuscript.jpg

  # Strip XML for plain text
  python predict.py --image document.png --strip-xml --output result.txt

  # JSON output with metadata
  python predict.py --image scroll.jpg --format json --output result.json

  # Shorter output for testing
  python predict.py --image sample.jpg --max-new-tokens 500
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg",
        help="Path to input image (default: shared test image)"
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
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20000,
        help="Maximum tokens to generate (default: 20000 for full pages, use 500 for testing)"
    )
    parser.add_argument(
        "--strip-xml",
        action="store_true",
        help="Strip XML markup from output (default: keep XML structure)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Resolve paths
    INP = get_input_dir()
    OUT = get_output_dir()

    if args.output is None:
        ext = "json" if args.format == "json" else "txt"
        args.output = f"result.{ext}"

    input_path = resolve_input_path(args.image, INP)
    output_path = Path(resolve_output_path(args.output, OUT))

    # Validate input
    if not Path(input_path).exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    # Show input
    print(f"\n{'='*80}")
    print("INPUT")
    print('='*80)
    print(f"Image: {input_path}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Strip XML: {args.strip_xml}")
    print('='*80 + '\n')

    # Load model
    try:
        model, processor = load_model(args.model, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load image
    try:
        image = load_image(str(input_path))
    except Exception as e:
        print(f"Error loading image: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    try:
        output_text = run_inference(
            model, processor, image, args.max_new_tokens, args.strip_xml
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print preview
    print("\n" + "="*80)
    print(f"{args.format.upper()} OUTPUT (preview):")
    print("="*80)
    preview = output_text[:1000] + ("..." if len(output_text) > 1000 else "")
    print(preview)
    print("="*80 + "\n")

    # Save output
    metadata = {
        "image_size": image.size,
        "max_new_tokens": args.max_new_tokens,
        "strip_xml": args.strip_xml
    }
    save_output(output_text, str(output_path), args.format, metadata)

    print(f"Output saved to {output_path}")
    print("\nCHURRO-3B Statistics:")
    print(f"  - Output length: {len(output_text)} characters")
    print(f"  - Image size: {image.size}")
    print(f"  - Model: {args.model}")
    print(f"  - Performance: 15.5× lower cost than Gemini 2.5 Pro")
    print(f"  - Languages: 46 clusters across 22 centuries")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
