#!/usr/bin/env python3
"""
olmOCR-2 - Production-Ready OCR from AllenAI

This script runs olmOCR-2-7B-1025, a 7B parameter model fine-tuned from Qwen2.5-VL
with GRPO RL training for improved math equations, tables, and complex OCR scenarios.

Achieves 82.4±1.1 score on olmOCR-Bench, with particularly strong performance on
mathematical content and old scans. Designed for large-scale production pipelines
processing millions of documents.

Model: https://github.com/allenai/olmocr
HuggingFace: allenai/olmOCR-2-7B-1025-FP8
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import shutil
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


def run_ocr(input_path: str):
    """
    Run OCR using olmOCR-2 pipeline.

    Args:
        input_path: Path to the image or PDF file

    Returns:
        dict: OCR output with markdown content and metadata
    """
    print(f"Initializing olmOCR-2 pipeline...")
    print("Model: allenai/olmOCR-2-7B-1025-FP8")
    print("This may take a few minutes on first run (downloading model weights)...")

    # Create a temporary workspace directory
    with tempfile.TemporaryDirectory() as temp_workspace:
        print(f"Processing: {input_path}")
        print(f"Temporary workspace: {temp_workspace}")

        # Determine file type
        input_path_obj = Path(input_path)
        is_pdf = input_path_obj.suffix.lower() == '.pdf'

        # Build olmOCR command
        # python -m olmocr.pipeline ./workspace --markdown --pdfs/--pngs input_file
        cmd = [
            sys.executable, "-m", "olmocr.pipeline",
            temp_workspace,
            "--markdown"
        ]

        if is_pdf:
            cmd.extend(["--pdfs", str(input_path)])
        else:
            # For images (PNG, JPG, JPEG), use --pngs
            cmd.extend(["--pngs", str(input_path)])

        print(f"Running command: {' '.join(cmd)}")

        # Run olmOCR pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("ERROR: olmOCR pipeline failed")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"olmOCR pipeline failed with return code {result.returncode}")

        print("Pipeline completed successfully")
        print(f"STDOUT: {result.stdout}")

        # Read the markdown output
        # olmOCR creates output in workspace/markdown/ directory
        markdown_dir = Path(temp_workspace) / "markdown"

        if not markdown_dir.exists():
            print(f"WARNING: Expected markdown directory not found: {markdown_dir}")
            print(f"Workspace contents: {list(Path(temp_workspace).iterdir())}")
            # Try to find markdown files anywhere in the workspace
            markdown_files = list(Path(temp_workspace).rglob("*.md"))
            if not markdown_files:
                raise FileNotFoundError(f"No markdown output found in {temp_workspace}")
            markdown_file = markdown_files[0]
        else:
            # Get the first markdown file (should match input filename)
            markdown_files = list(markdown_dir.glob("*.md"))
            if not markdown_files:
                raise FileNotFoundError(f"No markdown files found in {markdown_dir}")
            markdown_file = markdown_files[0]

        print(f"Reading output from: {markdown_file}")

        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        return {
            "markdown": markdown_content,
            "metadata": {
                "model": "olmOCR-2-7B-1025-FP8",
                "source": "AllenAI",
                "model_size": "7B parameters",
                "base_model": "Qwen2.5-VL-7B-Instruct",
                "training": "GRPO RL for math, tables, complex OCR",
                "benchmark_score": "82.4±1.1 on olmOCR-Bench"
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
        markdown_content = output.get("markdown", "")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
    elif format == "txt":
        # Save as plain text
        text_content = output.get("markdown", "")
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
        description="olmOCR-2 - Production-Ready OCR with 7B Parameter Model"
    )

    # Default to shared test image
    parser.add_argument(
        "--image",
        type=str,
        default="/cvlization_repo/examples/doc_ai/leaderboard/test_data/sample.jpg",
        help="Path to input image or PDF (default: shared test image)"
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
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("INPUT")
    print("=" * 80)
    print(f"File: {input_path}")
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
        print("olmOCR-2 Statistics:")
        print(f"  - Output length: {len(output.get('markdown', ''))} characters")
        print(f"  - Model: olmOCR-2-7B-1025-FP8 (7B parameters)")
        print(f"  - Base model: Qwen2.5-VL-7B-Instruct with GRPO RL training")
        print(f"  - Benchmark score: 82.4±1.1 on olmOCR-Bench")
        print(f"  - Specialty: Math equations, tables, large-scale pipelines")
        print()
        print("Done!")

    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
