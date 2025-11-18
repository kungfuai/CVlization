#!/usr/bin/env python3
"""
Batch inference for Pixtral 12B Vision Language Model.

Processes batches of images with custom prompts from JSONL input.
Pixtral 12B from Mistral AI with 4-bit quantization via Unsloth.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image
from unsloth import FastVisionModel

MODEL_ID = "unsloth/Pixtral-12B-2409"


def load_model(model_id: str = MODEL_ID):
    """Load Pixtral 12B model using Unsloth with 4-bit quantization."""
    print(f"Loading {model_id} with 4-bit quantization...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=True,
    )

    # Prepare for inference
    FastVisionModel.for_inference(model)

    print(f"Model loaded successfully!")
    return model, tokenizer


def run_inference_batch(
    model,
    tokenizer,
    image_paths: List[str],
    prompt: str,
    max_tokens: int = 512
) -> str:
    """Run inference on multiple images (Pixtral supports multi-image)."""
    # Load all images
    images = [Image.open(path).convert("RGB") for path in image_paths]

    # Prepare messages with support for multiple images
    content = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    # Tokenize inputs with all images
    inputs = tokenizer(
        images,  # Pass all images
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate with tight constraints for concise answers
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        repetition_penalty=1.2,
        do_sample=False,  # Greedy decoding for consistency
    )

    # Decode only the new tokens (skip the input)
    input_length = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0][input_length:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return result


def process_batch(
    model,
    tokenizer,
    batch_input_file: Path,
    output_dir: Path,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """Process a batch of images from JSONL input."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read batch input
    with open(batch_input_file) as f:
        requests = [json.loads(line) for line in f]

    print(f"\nProcessing {len(requests)} requests...")
    print(f"Max tokens: {max_tokens}")
    print()

    total = len(requests)
    for idx, request in enumerate(requests, 1):
        request_id = request.get("id", f"request_{idx}")
        image_paths = request["images"]
        prompt = request["prompt"]
        output_file = request.get("output", f"{request_id}.txt")

        print(f"[{idx}/{total}] Processing {request_id}...", end=" ", flush=True)

        # Run inference
        response = run_inference_batch(
            model=model,
            tokenizer=tokenizer,
            image_paths=image_paths,
            prompt=prompt,
            max_tokens=max_tokens
        )

        # Save output
        output_path = output_dir / output_file
        output_path.write_text(response, encoding='utf-8')

        print(f"✓ Saved to {output_file}")

    print(f"\n✓ Completed {total} requests")
    print(f"  Output directory: {output_dir}")

    return {
        "total_requests": total,
        "output_dir": str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pixtral 12B batch inference from JSONL (Unsloth 4-bit)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
JSONL Input Format:
Each line should be a JSON object with:
  {
    "images": ["path/to/image1.png", "path/to/image2.png"],  // Supports multiple images
    "prompt": "Your question here",
    "id": "unique_request_id",
    "output": "output_filename.txt"
  }

Example:
  python batch_predict.py \\
    --batch-input batch_requests.jsonl \\
    --output-dir outputs/
"""
    )

    parser.add_argument(
        "--batch-input",
        type=Path,
        required=True,
        help="Path to JSONL file with batch inputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})"
    )

    args = parser.parse_args()

    print("="*60)
    print("Pixtral 12B Batch Inference (Unsloth 4-bit)")
    print("="*60)
    print(f"Model: {args.model_id}")
    print(f"Batch input: {args.batch_input}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)

    # Load model
    model, tokenizer = load_model(args.model_id)

    # Process batch
    process_batch(
        model=model,
        tokenizer=tokenizer,
        batch_input_file=args.batch_input,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
