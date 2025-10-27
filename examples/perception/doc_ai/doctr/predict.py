#!/usr/bin/env python3
"""
docTR OCR - Document Text Recognition

This script demonstrates OCR using docTR, an end-to-end document OCR solution
that performs both text detection and recognition.
Dual-mode execution: standalone or via CVL with --inputs/--outputs.

Model: https://github.com/mindee/doctr
"""

import argparse
import json
from pathlib import Path
from PIL import Image

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


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


def run_ocr(image_path: str, det_arch: str = 'db_resnet50', reco_arch: str = 'crnn_vgg16_bn', use_gpu: bool = True):
    """Run OCR on an image using docTR."""
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile

    print(f"Loading docTR model (det: {det_arch}, reco: {reco_arch})...")
    model = ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        pretrained=True
    )

    print(f"Running OCR on {image_path}...")
    doc = DocumentFile.from_images(image_path)
    result = model(doc)

    return result


def format_output(result, format_type: str = "text"):
    """Format OCR result."""
    if format_type == "json":
        return result.export()
    else:
        # Extract text in reading order
        text_lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [word.value for word in line.words]
                    text_lines.append(" ".join(words))
        return "\n".join(text_lines)


def save_output(output, output_path: str, format_type: str = "txt"):
    """Save the output to a file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format_type == "json":
        # Try to parse as JSON and pretty-print
        try:
            json_data = json.loads(output) if isinstance(output, str) else output
            with open(output_file, "w") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, save as text
            with open(output_file, "w") as f:
                f.write(str(output))
    else:
        with open(output_file, "w") as f:
            f.write(str(output))


def main():
    parser = argparse.ArgumentParser(
        description="Run docTR OCR on an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/sample.jpg",
        help="Path to input image or URL (default: examples/sample.jpg)"
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
        "--det-arch",
        type=str,
        default="db_resnet50",
        help="Detection architecture (default: db_resnet50). Options: db_resnet50, db_mobilenet_v3_large, linknet_resnet18"
    )
    parser.add_argument(
        "--reco-arch",
        type=str,
        default="crnn_vgg16_bn",
        help="Recognition architecture (default: crnn_vgg16_bn). Options: crnn_vgg16_bn, crnn_mobilenet_v3_small, master, sar_resnet31"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
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

    # Show input
    print(f"\n{'='*80}")
    print("INPUT")
    print('='*80)
    print(f"Image: {input_path}")
    print(f"Detection: {args.det_arch}")
    print(f"Recognition: {args.reco_arch}")
    print('='*80 + '\n')

    # Load image
    print(f"Loading image...")
    image = load_image(str(input_path))
    print(f"Image loaded: {image.size}")

    # Run OCR
    result = run_ocr(str(input_path), args.det_arch, args.reco_arch, use_gpu=not args.no_gpu)

    # Format output
    output = format_output(result, args.format)

    # Print output preview
    print("\n" + "="*80)
    print(f"{args.format.upper()} OUTPUT (preview):")
    print("="*80)
    output_str = json.dumps(output, indent=2) if isinstance(output, dict) else str(output)
    preview = output_str[:500] + ("..." if len(output_str) > 500 else "")
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
