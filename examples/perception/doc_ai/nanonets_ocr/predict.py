#!/usr/bin/env python3
"""
Document OCR using Nanonets-OCR2-3B VLM model.
Supports complex document extraction with tables, forms, equations, and visual question answering.
Dual-mode execution: standalone or via CVL with --inputs/--outputs.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from PIL import Image
from pdf2image import convert_from_path

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


def load_image(image_path):
    """Load image from file path, handling PDFs."""
    path = Path(image_path)

    # Handle PDF files
    if path.suffix.lower() == '.pdf':
        # Convert first page of PDF to image
        images = convert_from_path(str(path), first_page=1, last_page=1)
        if images:
            return images[0].convert('RGB')
        else:
            raise ValueError(f"Could not extract image from PDF: {image_path}")

    # Handle regular image files
    return Image.open(image_path).convert('RGB')


def process_document(image_path, model, processor, device, prompt=None, max_tokens=4096):
    """Process a document image with the model."""
    # Load image
    image = load_image(image_path)

    # Default prompt for OCR
    if prompt is None:
        prompt = "Convert this document to markdown format, preserving all structure, tables, equations, and content."

    # Create messages following official format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    # Trim input tokens
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

    # Decode
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="Document OCR using Nanonets-OCR2-3B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic OCR to markdown
  python predict.py --input document.jpg

  # Save output to file
  python predict.py --input document.jpg --output output.md

  # Visual Question Answering
  python predict.py --input chart.png --mode vqa --question "What is the total revenue?"

  # JSON output
  python predict.py --input form.jpg --format json --output result.json
        """
    )
    parser.add_argument("--input", type=str, default="examples/sample.jpg",
                       help="Path to input image file (default: examples/sample.jpg)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (default: outputs/result.{format})")
    parser.add_argument("--format", type=str, choices=["markdown", "json"], default="markdown",
                       help="Output format: markdown (default) or json")
    parser.add_argument("--mode", type=str, choices=["ocr", "vqa"], default="ocr",
                       help="Processing mode: ocr (document extraction) or vqa (visual question answering)")
    parser.add_argument("--question", type=str, help="Question for VQA mode")
    parser.add_argument("--model", type=str, default="nanonets/Nanonets-OCR2-3B",
                       help="Model to use (default: nanonets/Nanonets-OCR2-3B)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda",
                       help="Device to run inference on (default: cuda)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Maximum tokens to generate (default: 4096)")

    args = parser.parse_args()

    # Resolve paths for CVL dual-mode support
    INP = get_input_dir()
    OUT = get_output_dir()

    # Smart default for output path
    if args.output is None:
        ext = {"json": "json", "markdown": "md"}[args.format]
        args.output = f"result.{ext}"

    # Resolve paths using cvlization utilities
    input_path = Path(resolve_input_path(args.input, INP))
    output_path = Path(resolve_output_path(args.output, OUT))

    # Validate input file
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    # Validate VQA mode
    if args.mode == "vqa" and not args.question:
        print("Error: --question is required for VQA mode")
        return 1

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model and processor
    print(f"Loading Nanonets-OCR2-3B model: {args.model}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    if device == "cpu":
        model = model.to(device)

    # Prepare prompt
    if args.mode == "vqa":
        prompt = args.question
    else:
        prompt = "Convert this document to markdown format, preserving all structure, tables, equations, and content."

    # Show input
    print(f"\n{'='*80}")
    print("INPUT")
    print('='*80)
    print(f"Image: {input_path}")
    print(f"Mode: {args.mode}")
    if args.mode == "vqa":
        print(f"Question: {args.question}")
    print('='*80 + '\n')

    # Process document
    print(f"Processing document...", flush=True)
    output_text = process_document(
        str(input_path),
        model,
        processor,
        device,
        prompt=prompt,
        max_tokens=args.max_tokens
    )

    # Format output if needed
    if args.format == "json":
        output_data = {
            "input_file": str(input_path),
            "model": args.model,
            "mode": args.mode,
            "content": output_text
        }
        if args.mode == "vqa":
            output_data["question"] = args.question
        output_text = json.dumps(output_data, indent=2, ensure_ascii=False)

    # Print output preview
    print("\n" + "="*80)
    print(f"{args.format.upper()} OUTPUT (preview):")
    print("="*80)
    preview = output_text[:500] + ("..." if len(output_text) > 500 else "")
    print(preview)
    print("="*80 + "\n")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding='utf-8')

    # Show container path (CVL will translate to host path)
    print(f"Output saved to {output_path}")
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
