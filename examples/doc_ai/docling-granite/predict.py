#!/usr/bin/env python3
"""
Document extraction using IBM Granite-Docling-258M VLM model.
End-to-end document understanding with a single 258M parameter model.
"""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    parser = argparse.ArgumentParser(description="Extract content from document images using Granite-Docling-258M")
    parser.add_argument("input_file", type=str, help="Path to input image file (PNG, JPG, etc.)")
    parser.add_argument("--output", type=str, help="Output file path (optional, prints to stdout if not specified)")
    parser.add_argument("--format", type=str, choices=["markdown", "json"], default="markdown",
                       help="Output format: markdown (default) or json")
    parser.add_argument("--model", type=str, default="ibm-granite/granite-docling-258M",
                       help="Model to use (default: ibm-granite/granite-docling-258M)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                       help="Device to run inference on")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found")
        return 1

    # Load model and processor
    print(f"Loading Granite-Docling model: {args.model}...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
        device_map=args.device
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # Load image
    print(f"Processing: {input_path.name}", flush=True)
    image = Image.open(input_path).convert("RGB")

    # Prepare prompt based on output format
    if args.format == "markdown":
        prompt = "Convert the document to Markdown format, preserving all structure, tables, and formatting."
    else:  # json
        prompt = "Extract the document structure and content as JSON, including headings, paragraphs, tables, and lists."

    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(input_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(args.device)

    # Generate output
    print("Generating extraction...", flush=True)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False
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

    # Format output
    if args.format == "json":
        try:
            # Try to parse as JSON if model returned JSON
            output_data = json.loads(output_text)
            output_text = json.dumps(output_data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            # If not valid JSON, wrap it
            output_data = {
                "input_file": str(input_path),
                "model": args.model,
                "content": output_text
            }
            output_text = json.dumps(output_data, indent=2, ensure_ascii=False)

    # Write or print output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding='utf-8')
        print(f"Output written to: {output_path}")
    else:
        print("\n" + "="*80)
        print(output_text)
        print("="*80)

    return 0

if __name__ == "__main__":
    exit(main())
