#!/usr/bin/env python3
"""
NVIDIA Nemotron Parse v1.1 - Document Structure Understanding with Spatial Grounding

This script demonstrates document parsing using NVIDIA Nemotron Parse v1.1, a <1B parameter
transformer-based vision-encoder-decoder model for document structure understanding.
Extracts text, tables, and layout elements with bounding boxes and semantic class labels.

Features:
- Text extraction with spatial grounding (bounding boxes)
- Table extraction with complex formatting support
- Object classification (titles, sections, captions, footnotes, lists, tables, etc.)
- Mathematical equation and formatting extraction (LaTeX)
- Document structure understanding
- Output formats: Markdown, JSON, plain text

Model: https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1
License: NVIDIA Community Model License
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

# CVL dual-mode execution support - make optional for branches without cvlization
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

    # Fallback functions for standalone execution
    def get_input_dir():
        """Fallback: return current working directory"""
        return os.getcwd()

    def get_output_dir():
        """Fallback: return outputs directory relative to CWD"""
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path, base_dir):
        """Fallback: resolve input path relative to base_dir"""
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)

    def resolve_output_path(path, base_dir):
        """Fallback: resolve output path relative to base_dir"""
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)


# Model configuration
MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
DEFAULT_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"


def detect_device() -> Tuple[str, torch.dtype]:
    """
    Auto-detect the best available device and appropriate dtype.

    Returns:
        Tuple of (device_name, dtype)
    """
    if torch.cuda.is_available():
        device = "cuda"
        # Use bfloat16 for modern GPUs (compute capability >= 8.0)
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


def load_model(model_id: str = MODEL_ID, device: str = None):
    """
    Load the Nemotron Parse model and processor.

    Args:
        model_id: HuggingFace model ID
        device: Device to load model on (auto-detected if None)

    Returns:
        Tuple of (model, processor, device, dtype)
    """
    if device is None:
        device, dtype = detect_device()
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_id}...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto"
    )

    print(f"Model loaded successfully on {device}")

    return model, processor, device, dtype


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from file or URL.

    Args:
        image_path: Path to image file or URL

    Returns:
        PIL Image in RGB format
    """
    if image_path.startswith(("http://", "https://")):
        from io import BytesIO
        import requests
        response = requests.get(image_path)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    return image.convert("RGB")


def parse_nemotron_output(output_text: str) -> Dict:
    """
    Parse Nemotron Parse output to extract markdown, bounding boxes, and classes.

    The model outputs text in a special format with embedded bbox and class information.
    Format: <bbox>coords</bbox><class>label</class>text

    Args:
        output_text: Raw output from the model

    Returns:
        Dict with 'markdown', 'bboxes', 'classes', and 'structured_elements'
    """
    # Extract bounding boxes (format: <bbox>x1,y1,x2,y2</bbox>)
    bbox_pattern = r'<bbox>([\d,]+)</bbox>'
    bboxes = []
    for match in re.finditer(bbox_pattern, output_text):
        coords = [int(x) for x in match.group(1).split(',')]
        bboxes.append(coords)

    # Extract classes (format: <class>label</class>)
    class_pattern = r'<class>([\w\s]+)</class>'
    classes = []
    for match in re.finditer(class_pattern, output_text):
        classes.append(match.group(1))

    # Clean markdown text (remove bbox and class tags)
    markdown_text = re.sub(r'<bbox>[\d,]+</bbox>', '', output_text)
    markdown_text = re.sub(r'<class>[\w\s]+</class>', '', markdown_text)
    markdown_text = markdown_text.strip()

    # Create structured elements combining bbox, class, and text
    structured_elements = []
    bbox_matches = list(re.finditer(bbox_pattern, output_text))
    class_matches = list(re.finditer(class_pattern, output_text))

    for i, (bbox_match, class_match) in enumerate(zip(bbox_matches, class_matches)):
        coords = [int(x) for x in bbox_match.group(1).split(',')]
        class_label = class_match.group(1)

        # Extract text following the tags (until next bbox or end)
        start_pos = class_match.end()
        if i + 1 < len(bbox_matches):
            end_pos = bbox_matches[i + 1].start()
        else:
            end_pos = len(output_text)

        element_text = output_text[start_pos:end_pos].strip()
        element_text = re.sub(r'<bbox>[\d,]+</bbox>', '', element_text)
        element_text = re.sub(r'<class>[\w\s]+</class>', '', element_text)

        structured_elements.append({
            "bbox": coords,
            "class": class_label,
            "text": element_text
        })

    return {
        "markdown": markdown_text,
        "bboxes": bboxes,
        "classes": classes,
        "structured_elements": structured_elements,
        "raw_output": output_text
    }


def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 4096
) -> str:
    """
    Run inference on the document image.

    Args:
        model: The loaded model
        processor: The loaded processor
        image: PIL Image to process
        prompt: Prompt string (default uses bbox + classes + markdown)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Raw model output text
    """
    print(f"Processing image (size: {image.size})...")

    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output
    print("Generating document parse...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
        )

    # Decode output
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True
    )[0]

    return output_text


def save_output(parsed_output: Dict, output_path: str, format: str = "md"):
    """
    Save the parsed output to a file.

    Args:
        parsed_output: Dict from parse_nemotron_output()
        output_path: Path to save output
        format: Output format ('md', 'json', 'txt')
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save complete structured output
        output_data = {
            "markdown": parsed_output["markdown"],
            "bboxes": parsed_output["bboxes"],
            "classes": parsed_output["classes"],
            "structured_elements": parsed_output["structured_elements"],
            "num_elements": len(parsed_output["structured_elements"])
        }
        output_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    elif format == "txt":
        # Save plain text (markdown without formatting)
        output_file.write_text(parsed_output["markdown"])
    else:  # md
        # Save markdown with metadata header
        content = f"# Document Parse Output\n\n"
        content += f"**Elements detected:** {len(parsed_output['structured_elements'])}\n\n"
        content += f"---\n\n{parsed_output['markdown']}\n"
        output_file.write_text(content)


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Nemotron Parse v1.1 - Document structure understanding with spatial grounding",
        epilog="""
Examples:
  # Basic usage - extract markdown with layout
  python predict.py --image document.pdf --output result.md

  # Extract with bounding boxes and classes as JSON
  python predict.py --image invoice.png --format json --output result.json

  # Plain text extraction
  python predict.py --image form.jpg --format txt --output result.txt

  # Custom prompt (without bboxes)
  python predict.py --image doc.png --prompt "</s><s><output_markdown>" --output result.md
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg",
        help="Path to input image or PDF (default: shared test image)"
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
        choices=["txt", "json", "md"],
        default="md",
        help="Output format (default: md for markdown)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Model ID on HuggingFace (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Custom prompt (default: {DEFAULT_PROMPT})"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Resolve paths for CVL dual-mode support
    INP = get_input_dir()
    OUT = get_output_dir()

    # Smart default for output path
    if args.output is None:
        ext = {"json": "json", "txt": "txt", "md": "md"}[args.format]
        args.output = f"result.{ext}"

    # Resolve paths using cvlization utilities
    input_path = resolve_input_path(args.image, INP)
    output_path = Path(resolve_output_path(args.output, OUT))

    # Validate input file
    if not Path(input_path).exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    # Show input
    print(f"\n{'='*80}")
    print("INPUT")
    print('='*80)
    print(f"Image: {input_path}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt[:50]}..." if len(args.prompt) > 50 else f"Prompt: {args.prompt}")
    print('='*80 + '\n')

    # Load model
    try:
        model, processor, device, dtype = load_model(args.model, args.device)
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
        raw_output = run_inference(
            model, processor, image, args.prompt, args.max_new_tokens
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Parse output
    parsed_output = parse_nemotron_output(raw_output)

    # Print output preview
    print("\n" + "="*80)
    print(f"{args.format.upper()} OUTPUT (preview):")
    print("="*80)
    if args.format == "json":
        preview = json.dumps({
            "num_elements": len(parsed_output["structured_elements"]),
            "sample_elements": parsed_output["structured_elements"][:3]
        }, indent=2)
        print(preview)
    else:
        preview_text = parsed_output["markdown"]
        preview = preview_text[:1000] + ("..." if len(preview_text) > 1000 else "")
        print(preview)
    print("="*80 + "\n")

    # Save output
    save_output(parsed_output, str(output_path), args.format)

    # Show statistics
    print(f"Output saved to {output_path}")
    print("\nNemotron Parse Statistics:")
    print(f"  - Elements detected: {len(parsed_output['structured_elements'])}")
    print(f"  - Classes found: {set(parsed_output['classes'])}")
    print(f"  - Output length: {len(parsed_output['markdown'])} characters")
    print(f"  - Image size: {image.size}")
    print(f"  - Model: {args.model}")

    print("\nDone!")

    return 0


if __name__ == "__main__":
    exit(main())
