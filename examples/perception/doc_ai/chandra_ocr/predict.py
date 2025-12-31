#!/usr/bin/env python3
"""
Chandra OCR - High-Accuracy Document OCR

This script demonstrates OCR using Chandra, which achieves 83.1 score on OmniDocBench,
beating DeepSeek-OCR, dots.ocr, and olmOCR. Supports tables, forms, handwriting,
math, and 40+ languages with full layout preservation.

Model: https://github.com/datalab-to/chandra
HuggingFace: datalab-to/chandra
"""

import argparse
import json
import os
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


def run_ocr(image_path: str, prompt_type: str = "ocr_layout", model_name: str = "datalab-to/chandra"):
    """
    Run OCR using Chandra CLI tool.

    Args:
        image_path: Path to the image file
        prompt_type: Prompt type - "ocr_layout", "ocr", or "plain_ocr"
        model_name: HuggingFace model name

    Returns:
        dict: OCR output with markdown, html, and metadata
    """
    import subprocess
    import tempfile
    import shutil

    print(f"Running Chandra OCR with model: {model_name}")
    print(f"Prompt type: {prompt_type}")
    print("This may take a few minutes on first run (downloading ~7GB model)...")

    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Set environment variables for chandra
        env = os.environ.copy()
        env["CHANDRA_CHECKPOINT"] = model_name
        env["CHANDRA_PROMPT_TYPE"] = prompt_type

        # Run chandra CLI tool
        print(f"Processing image: {image_path}")
        print(f"Temporary output directory: {temp_output_dir}")

        cmd = [
            "chandra",
            str(image_path),
            str(temp_output_dir),
            "--method", "hf"
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error running chandra CLI:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Chandra CLI failed with return code {result.returncode}")

        # Read the output file
        # Chandra CLI creates a subdirectory with the input filename
        # and puts markdown files inside it
        temp_path = Path(temp_output_dir)

        # Find all markdown files recursively
        output_files = list(temp_path.rglob("*.md"))

        if not output_files:
            # If no .md files, list what we have for debugging
            all_files = list(temp_path.rglob("*"))
            print(f"No .md files found. Contents of {temp_output_dir}:")
            for f in all_files:
                print(f"  {f}")
            raise RuntimeError(f"No markdown output files found in {temp_output_dir}")

        # Read the first markdown file
        output_file = output_files[0]
        print(f"Reading output from: {output_file}")

        with open(output_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # Get image size for metadata
        from PIL import Image
        image = Image.open(image_path)

        return {
            "raw": markdown_content,
            "markdown": markdown_content,
            "html": "",  # CLI doesn't provide HTML directly
            "metadata": {
                "prompt_type": prompt_type,
                "model": model_name,
                "image_size": image.size
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
    elif format == "html":
        # Save as HTML
        html_content = output.get("html", output.get("markdown", output.get("raw", "")))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
    else:  # md or txt
        # Save as markdown/text
        markdown_content = output.get("markdown", output.get("raw", ""))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)


def main():
    parser = argparse.ArgumentParser(
        description="Run Chandra OCR for high-accuracy document OCR with layout preservation"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg",
        help="Path to input image (default: shared test image)"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["ocr_layout", "ocr", "plain_ocr"],
        default="ocr_layout",
        help="Prompt type: ocr_layout (default, with layout), ocr (structured), or plain_ocr (simple text)"
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
        choices=["txt", "json", "md", "html"],
        default="md",
        help="Output format (default: md for markdown)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="datalab-to/chandra",
        help="Model name on HuggingFace (default: datalab-to/chandra)"
    )

    args = parser.parse_args()

    # Resolve paths for CVL dual-mode support
    INP = get_input_dir()
    OUT = get_output_dir()

    # Smart default for output path
    if args.output is None:
        ext = {"json": "json", "txt": "txt", "md": "md", "html": "html"}[args.format]
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
    print(f"Prompt Type: {args.prompt_type}")
    print('='*80 + '\n')

    # Run OCR
    try:
        output = run_ocr(str(input_path), args.prompt_type, args.model)
    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print output preview
    print("\n" + "="*80)
    print(f"{args.format.upper()} OUTPUT (preview):")
    print("="*80)
    if output:
        preview_text = output.get("markdown", output.get("raw", ""))
        preview = preview_text[:1000] + ("..." if len(preview_text) > 1000 else "")
        print(preview)
    else:
        print("(empty output)")
    print("="*80 + "\n")

    # Save output
    if output:
        save_output(output, str(output_path), args.format)

        # Show container path (CVL will translate to host path)
        print(f"Output saved to {output_path}")
        print("\nChandra OCR Statistics:")
        output_text = output.get("markdown", output.get("raw", ""))
        print(f"  - Output length: {len(output_text)} characters")
        print(f"  - Image size: {output.get('metadata', {}).get('image_size', 'N/A')}")
        print(f"  - OmniDocBench Score: 83.1 (beats DeepSeek-OCR, dots.ocr, olmOCR)")
        print(f"  - Model: {args.model}")
    else:
        print("Warning: No output generated")

    print("\nDone!")

    return 0


if __name__ == "__main__":
    exit(main())
