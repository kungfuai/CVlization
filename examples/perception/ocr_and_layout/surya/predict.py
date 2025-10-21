#!/usr/bin/env python3
"""
Surya OCR - Multilingual Document OCR

This script demonstrates OCR, layout analysis, and reading order detection
using Surya, supporting 90+ languages.

Model: https://github.com/datalab-to/surya
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO


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

    print(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Surya OCR, layout analysis, or reading order detection"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/sample.jpg",
        help="Path to input image or URL"
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

    args = parser.parse_args()

    # Load image
    print(f"Loading image from {args.image}...")
    image = load_image(args.image)
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

    # Print output
    print("\n" + "="*80)
    print(f"{args.task.upper()} OUTPUT:")
    print("="*80)
    print(output)
    print("="*80 + "\n")

    # Save output
    save_output(output, args.output, args.format, metadata)
    print("Done!")


if __name__ == "__main__":
    main()
