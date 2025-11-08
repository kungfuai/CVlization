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
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

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


# Global model and processor (loaded once)
_model = None
_processor = None


def load_model():
    """Load the olmOCR-2 model and processor."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    print("Loading olmOCR-2-7B-1025-FP8 model...")
    print("This may take a few minutes on first run (downloading model weights)...")

    model_name = "allenai/olmOCR-2-7B-1025-FP8"

    # Load model using AutoModel to handle qwen2.5-vl architecture
    _model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load processor
    _processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    print("Model loaded successfully!")
    return _model, _processor


def run_ocr(input_path: str):
    """
    Run OCR using olmOCR-2 model directly via transformers.

    Args:
        input_path: Path to the image or PDF file

    Returns:
        dict: OCR output with markdown content and metadata
    """
    print(f"Processing: {input_path}")

    # Load model
    model, processor = load_model()

    # Open image
    image = Image.open(input_path).convert("RGB")

    # Prepare messages for Qwen2.5-VL format
    # olmOCR-2 uses a specific prompt for OCR
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": input_path,
                },
                {
                    "type": "text",
                    "text": "Convert the content of this document image to markdown format. Preserve all text, structure, tables, and mathematical equations."
                },
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate output
    print("Running inference...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,  # Greedy decoding for deterministic OCR
        )

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return {
        "markdown": output_text,
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
