#!/usr/bin/env python3
"""
Document extraction using IBM Granite-Docling-258M VLM model.
End-to-end document understanding with a single 258M parameter model.
"""
import argparse
import json
import os
from pathlib import Path
import torch
from cvlization.paths import resolve_input_path, resolve_output_path
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

def main():
    parser = argparse.ArgumentParser(description="Extract content from document images using Granite-Docling-258M")
    parser.add_argument("input_file", type=str, nargs='?', default=None,
                       help="Path to input image file (PNG, JPG, etc.) (default: uses bundled sample)")
    parser.add_argument("--output", type=str, default="output.md",
                       help="Output file path (default: output.md)")
    parser.add_argument("--format", type=str, choices=["markdown", "json", "docling"], default="markdown",
                       help="Output format: markdown (default), json, or docling")
    parser.add_argument("--model", type=str, default="ibm-granite/granite-docling-258M",
                       help="Model to use (default: ibm-granite/granite-docling-258M)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda",
                       help="Device to run inference on (default: cuda)")

    args = parser.parse_args()

    # Handle bundled sample vs user-provided input
    DEFAULT_SAMPLE = "examples/sample.jpg"
    if args.input_file is None:
        input_path = Path(DEFAULT_SAMPLE)
        print(f"No input provided, using bundled sample: {DEFAULT_SAMPLE}")
    else:
        # Resolve user-provided path against CVL_INPUTS
        input_path = Path(resolve_input_path(args.input_file))
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    DEVICE = args.device

    # Load model and processor
    print(f"Loading Granite-Docling model: {args.model}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        _attn_implementation="sdpa",
    ).to(DEVICE)

    # Load image
    print(f"Processing: {input_path.name}", flush=True)
    image = load_image(str(input_path))

    # Prepare prompt based on output format
    if args.format == "docling":
        prompt = "Convert this page to docling."
    elif args.format == "markdown":
        prompt = "Convert this page to markdown format."
    else:  # json
        prompt = "Convert this page to docling."  # Docling format can be parsed as JSON

    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    # Generate output
    print("Generating extraction...", flush=True)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    # Trim input tokens from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Format output if needed
    if args.format == "json":
        # Try to parse as JSON if it's in docling format
        try:
            import re
            # Extract JSON from markdown code block if present
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                output_data = json.loads(json_str)
                output_text = json.dumps(output_data, indent=2, ensure_ascii=False)
            else:
                # Try direct parsing
                output_data = json.loads(output_text)
                output_text = json.dumps(output_data, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, AttributeError):
            # If not valid JSON, wrap it
            output_data = {
                "input_file": str(input_path),
                "model": args.model,
                "content": output_text
            }
            output_text = json.dumps(output_data, indent=2, ensure_ascii=False)

    # Save output to file
    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding='utf-8')

    # Print output
    print("\n" + "="*80)
    print(output_text)
    print("="*80)
    print(f"\n✓ Output saved to: {output_path.resolve()}")

    return 0

if __name__ == "__main__":
    exit(main())
