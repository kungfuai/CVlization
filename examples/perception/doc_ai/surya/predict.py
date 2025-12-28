#!/usr/bin/env python3
"""
Surya OCR - Multilingual Document OCR

This script demonstrates OCR, layout analysis, and reading order detection
using Surya, supporting 90+ languages.
Dual-mode execution: standalone or via CVL with --inputs/--outputs.

Model: https://github.com/datalab-to/surya
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)

DEFAULT_IMAGE = "examples/sample.jpg"


def load_image(image_path: str):
    """Load an image from file path or URL."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    return image.convert("RGB")


def run_ocr(images):
    """Run OCR on images using Surya (automatically handles 90+ languages)."""
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    print("Loading models...")
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()

    print("Running OCR...")
    predictions = recognition_predictor(images, det_predictor=detection_predictor)

    return predictions


def run_layout_analysis(images):
    """Run layout analysis on images using Surya."""
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.settings import settings

    print("Loading models...")
    layout_predictor = LayoutPredictor(
        FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    )

    print("Running layout analysis...")
    predictions = layout_predictor(images)

    return predictions


def run_reading_order(images):
    """Run reading order detection on images using Surya."""
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.settings import settings

    print("Loading models...")
    layout_predictor = LayoutPredictor(
        FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    )

    print("Running reading order detection...")
    predictions = layout_predictor(images)

    return predictions


def format_ocr_output(predictions):
    """Format OCR predictions as text."""
    output = []

    for page_idx, page_pred in enumerate(predictions):
        if len(predictions) > 1:
            output.append(f"\n=== Page {page_idx + 1} ===\n")

        # Extract text from OCR predictions
        if hasattr(page_pred, 'text_lines'):
            for line in page_pred.text_lines:
                if hasattr(line, 'text'):
                    output.append(line.text)
        elif hasattr(page_pred, 'bboxes'):
            for bbox in page_pred.bboxes:
                if hasattr(bbox, 'text'):
                    output.append(bbox.text)

    return "\n".join(output)


def format_layout_output(predictions):
    """Format layout predictions as text."""
    output = []

    for page_idx, page_pred in enumerate(predictions):
        if len(predictions) > 1:
            output.append(f"\n=== Page {page_idx + 1} ===\n")

        if hasattr(page_pred, 'bboxes'):
            output.append(f"Detected {len(page_pred.bboxes)} layout regions:")
            for idx, bbox in enumerate(page_pred.bboxes, 1):
                label = bbox.label if hasattr(bbox, 'label') else 'Unknown'
                output.append(f"  {idx}. {label}")
        else:
            output.append("No layout regions detected")

    return "\n".join(output)


def save_output(output: str, output_path: str, format: str = "txt", metadata: dict = None):
    """Save the output to a file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as JSON with metadata
        data = {
            "text": output,
            "model": "surya-ocr",
            **(metadata or {})
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_file, "w") as f:
            f.write(output)


def main():
    parser = argparse.ArgumentParser(
        description="Run Surya OCR, layout analysis, or reading order detection"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Path to input image or URL (default: examples/sample.jpg)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["ocr", "layout", "order"],
        default="ocr",
        help="Task to perform: ocr, layout, or order"
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

    args = parser.parse_args()

    # Resolve paths for CVL dual-mode support
    OUT = get_output_dir()

    # Smart default for output path
    if args.output is None:
        ext = {"json": "json", "txt": "txt"}[args.format]
        args.output = f"result.{ext}"

    # Defaults are local to example dir; user-provided paths resolve to cwd
    if args.image.startswith("http"):
        input_path = args.image
    elif args.image == DEFAULT_IMAGE:
        input_path = args.image
    else:
        input_path = resolve_input_path(args.image)
    # Output always resolves to user's cwd
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
    print(f"Task: {args.task}")
    print('='*80 + '\n')

    # Load image
    print(f"Loading image...")
    image = load_image(str(input_path))
    print(f"Image loaded: {image.size}")

    # Run task
    images = [image]
    metadata = {"task": args.task}

    if args.task == "ocr":
        predictions = run_ocr(images)
        output = format_ocr_output(predictions)

    elif args.task == "layout":
        predictions = run_layout_analysis(images)
        output = format_layout_output(predictions)

    elif args.task == "order":
        predictions = run_reading_order(images)
        output = format_layout_output(predictions)  # Similar format

    # Print output preview
    print("\n" + "="*80)
    print(f"{args.format.upper()} OUTPUT (preview):")
    print("="*80)
    preview = output[:500] + ("..." if len(output) > 500 else "")
    print(preview)
    print("="*80 + "\n")

    # Save output
    save_output(output, str(output_path), args.format, metadata)

    # Show container path (CVL will translate to host path)
    print(f"Output saved to {output_path}")
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
