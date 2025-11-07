#!/usr/bin/env python3
"""
DeepSeek-OCR - Context Compression & Document OCR

This script demonstrates OCR using DeepSeek-OCR, which uses optical context
compression to achieve 7-20x token reduction while maintaining 97% accuracy.
Dual-mode execution: standalone or via CVL with --inputs/--outputs.

Model: https://github.com/deepseek-ai/DeepSeek-OCR
HuggingFace: deepseek-ai/DeepSeek-OCR
"""

import argparse
import json
import os
from pathlib import Path
import torch

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


def run_ocr(image_path: str, task: str = "markdown", model_name: str = "deepseek-ai/DeepSeek-OCR"):
    """
    Run OCR using DeepSeek-OCR.

    Args:
        image_path: Path to the image file
        task: Task type - "markdown", "free_ocr", or "grounding"
        model_name: HuggingFace model name

    Returns:
        str: OCR output text
    """
    from transformers import AutoModel, AutoTokenizer
    import sys
    from io import StringIO

    print(f"Loading DeepSeek-OCR model from HuggingFace: {model_name}")
    print("This may take a few minutes on first run (downloading ~3GB model)...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"  # Use eager attention
    )

    model = model.eval()

    # Task-specific prompts based on DeepSeek-OCR documentation
    task_prompts = {
        "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "free_ocr": "<image>\nFree OCR.",
        "grounding": "<image>\n<|grounding|>Extract all text with locations."
    }

    prompt = task_prompts.get(task, task_prompts["markdown"])

    print(f"Running OCR with task: {task}")
    print(f"Prompt: {prompt}")

    # Create temporary output directory for model.infer()
    temp_output_dir = "/tmp/deepseek_ocr_output"
    os.makedirs(temp_output_dir, exist_ok=True)

    # Capture stdout since model.infer() prints output instead of returning it
    captured_output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        # Run inference using model's custom .infer() method
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=temp_output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True
        )
    finally:
        # Restore stdout
        sys.stdout = original_stdout

    # Get captured output
    stdout_text = captured_output.getvalue()

    # Parse the OCR output from stdout
    # The model prints the OCR result before "===============save results:==============="
    if "===============save results:===============" in stdout_text:
        ocr_output = stdout_text.split("===============save results:===============")[0].strip()
    else:
        ocr_output = stdout_text.strip()

    # Remove debug messages if any
    lines = ocr_output.split('\n')
    # Filter out lines with debug info (torch.Size, etc.)
    clean_lines = []
    skip_next = False
    for line in lines:
        if 'torch.Size' in line or '=====' in line or 'BASE:' in line or 'PATCHES:' in line:
            skip_next = True
            continue
        if skip_next and not line.strip():
            skip_next = False
            continue
        skip_next = False
        clean_lines.append(line)

    ocr_output = '\n'.join(clean_lines).strip()

    # If still empty, try the return value
    if not ocr_output and res:
        ocr_output = res

    return ocr_output if ocr_output else ""


def save_output(output: str, output_path: str, format: str = "txt", metadata: dict = None):
    """Save the output to a file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as JSON with metadata
        data = {
            "text": output,
            "model": "deepseek-ocr",
            **(metadata or {})
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR for document OCR with context compression"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg",
        help="Path to input image (default: shared test image)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["markdown", "free_ocr", "grounding"],
        default="markdown",
        help="Task to perform: markdown (default), free_ocr, or grounding"
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
        default="deepseek-ai/DeepSeek-OCR",
        help="Model name on HuggingFace (default: deepseek-ai/DeepSeek-OCR)"
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
    print(f"Task: {args.task}")
    print('='*80 + '\n')

    # Run OCR
    try:
        output = run_ocr(str(input_path), args.task, args.model)
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
        preview = output[:1000] + ("..." if len(output) > 1000 else "")
        print(preview)
    else:
        print("(empty output)")
    print("="*80 + "\n")

    # Save output
    if output:
        metadata = {
            "task": args.task,
            "model": args.model
        }
        save_output(output, str(output_path), args.format, metadata)

        # Show container path (CVL will translate to host path)
        print(f"Output saved to {output_path}")
        print("\nDeepSeek-OCR Statistics:")
        print(f"  - Output length: {len(output)} characters")
        print(f"  - Context compression: 7-20x token reduction vs traditional OCR")
        print(f"  - Model: {args.model}")
    else:
        print("Warning: No output generated")

    print("\nDone!")

    return 0


if __name__ == "__main__":
    exit(main())
