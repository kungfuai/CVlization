#!/usr/bin/env python3
"""
PaddleOCR-VL Inference Script

This script runs PaddleOCR-VL for document OCR with 109 language support.
Supports both standalone execution and CVlization integration.

Usage:
    python predict.py --image path/to/image.jpg --output output.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image

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


def run_ocr(image_path: str):
    """
    Run OCR using PaddleOCR-VL.

    Args:
        image_path: Path to the image file

    Returns:
        dict: OCR output with markdown and JSON content
    """
    from paddleocr import PaddleOCRVL

    print(f"Initializing PaddleOCR-VL pipeline...")
    print("This may take a few minutes on first run (downloading model weights)...")

    # Initialize pipeline
    pipeline = PaddleOCRVL()

    print(f"Processing image: {image_path}")

    # Run prediction
    output = pipeline.predict(image_path)

    # Extract results - PaddleOCR-VL returns result objects with markdown content
    results = []
    markdown_texts = []

    for res in output:
        # The result object should have a dictionary structure with markdown_texts
        # According to PaddleOCR-VL documentation, results have markdown attribute
        markdown_content = ""

        # Try multiple ways to extract markdown content
        if hasattr(res, 'markdown'):
            # If result has markdown attribute (dict)
            markdown_dict = res.markdown
            if isinstance(markdown_dict, dict) and 'markdown_texts' in markdown_dict:
                md_texts = markdown_dict['markdown_texts']
                # Only join if it's a list, otherwise use as-is
                markdown_content = "\n\n".join(md_texts) if isinstance(md_texts, list) else str(md_texts)
            else:
                markdown_content = str(markdown_dict)
        elif isinstance(res, dict):
            # If result is a dict
            if 'markdown' in res and isinstance(res['markdown'], dict):
                if 'markdown_texts' in res['markdown']:
                    md_texts = res['markdown']['markdown_texts']
                    markdown_content = "\n\n".join(md_texts) if isinstance(md_texts, list) else str(md_texts)
            elif 'markdown_texts' in res:
                md_texts = res['markdown_texts']
                markdown_content = "\n\n".join(md_texts) if isinstance(md_texts, list) else str(md_texts)
            elif 'vl_rec_res' in res and isinstance(res['vl_rec_res'], dict):
                # Check in vl_rec_res sub-dictionary
                vl_res = res['vl_rec_res']
                if 'md_content' in vl_res:
                    markdown_content = vl_res['md_content']
                elif 'markdown' in vl_res:
                    markdown_content = str(vl_res['markdown'])

        # Fallback: if still no content, convert to string
        if not markdown_content:
            markdown_content = str(res)

        markdown_texts.append(markdown_content)

        result_data = {
            "content": markdown_content,
            "metadata": {
                "model": "PaddleOCR-VL",
                "languages": "109 languages supported",
                "model_size": "0.9B parameters"
            }
        }
        results.append(result_data)

    # Combine all results
    combined_content = "\n\n".join(markdown_texts)

    return {
        "markdown": combined_content,
        "results": results,
        "metadata": {
            "model": "PaddleOCR-VL",
            "model_size": "0.9B parameters",
            "languages_supported": 109
        }
    }


def save_output(output: dict, output_path: str, format: str = "md"):
    """Save the output to a file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as JSON with all metadata
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    elif format == "md":
        # Save as Markdown
        markdown_content = output.get("markdown", output.get("content", ""))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
    elif format == "txt":
        # Save as plain text
        text_content = output.get("markdown", output.get("content", ""))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text_content)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Output saved to {output_file}")


def print_output_preview(output: dict, max_length: int = 500):
    """Print a preview of the output."""
    content = output.get("markdown", "")

    print("\n" + "=" * 80)
    print("MD OUTPUT (preview):")
    print("=" * 80)

    if len(content) > max_length:
        print(content[:max_length] + "...")
    else:
        print(content)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL - Ultra-Efficient Document OCR with 109 Language Support"
    )

    # Default to shared test image
    parser.add_argument(
        "--image",
        type=str,
        default="/cvlization_repo/examples/doc_ai/leaderboard/test_data/sample.jpg",
        help="Path to input image (default: shared test image)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.md",
        help="Output file path (default: outputs/result.md)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["md", "json", "txt"],
        default="md",
        help="Output format (default: md)"
    )

    args = parser.parse_args()

    # Resolve paths
    input_dir = get_input_dir()
    output_dir = get_output_dir()

    input_path = resolve_input_path(args.image, input_dir)
    output_path = resolve_output_path(args.output, output_dir)

    # Verify input exists
    if not os.path.exists(input_path):
        print(f"Error: Input image not found: {input_path}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("INPUT")
    print("=" * 80)
    print(f"Image: {input_path}")
    print("=" * 80)
    print()

    try:
        # Run OCR
        output = run_ocr(str(input_path))

        # Print preview
        print_output_preview(output)

        # Save output
        save_output(output, str(output_path), format=args.format)

        # Print statistics
        print()
        print("PaddleOCR-VL Statistics:")
        print(f"  - Output length: {len(output.get('markdown', ''))} characters")
        print(f"  - Model size: 0.9B parameters (ultra-efficient)")
        print(f"  - Languages supported: 109")
        print(f"  - HuggingFace trending: #1 for 5 consecutive days")
        print()
        print("Done!")

    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
