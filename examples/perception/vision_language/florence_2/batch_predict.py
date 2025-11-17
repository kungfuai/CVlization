#!/usr/bin/env python3
"""
Batch inference for Florence-2 Vision Language Model.

Supports both base Florence-2 models and fine-tuned variants (e.g., DocVQA).
Processes batches of images with custom prompts from JSONL input.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def load_model(model_id: str, device: str = "cuda"):
    """
    Load Florence-2 model and processor.

    Args:
        model_id: HuggingFace model ID
        device: Device to use (cuda, mps, or cpu)

    Returns:
        tuple: (model, processor)
    """
    print(f"Loading model: {model_id}...")

    # Determine dtype based on device
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    print(f"Model loaded successfully on {device}!")
    return model, processor


def load_image(image_path: str, max_size: int = None) -> Image.Image:
    """
    Load an image from path with optional resizing.

    Args:
        image_path: Path to image file
        max_size: Maximum dimension (width or height). If specified, resize maintaining aspect ratio.

    Returns:
        PIL.Image: Loaded image in RGB format
    """
    image = Image.open(image_path).convert("RGB")

    # Resize if max_size specified
    if max_size is not None:
        width, height = image.size
        if width > max_size or height > max_size:
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    task_prompt: str = None,
    device: str = "cuda",
    max_new_tokens: int = 1024
) -> str:
    """
    Run inference on a single image.

    Args:
        model: Loaded Florence-2 model
        processor: Model processor
        image: PIL Image
        prompt: Text prompt/question (used as-is if no task_prompt, or as text_input for certain tasks)
        task_prompt: Optional task prefix (e.g., "<CAPTION>", "<DocVQA>")
        device: Device being used
        max_new_tokens: Maximum tokens to generate

    Returns:
        str: Generated text response
    """
    # Construct full prompt
    # For Florence-2 base models: task_prompt must be ALONE (e.g., just "<CAPTION>")
    # For fine-tuned models (e.g., DocVQA): task_prompt + question works
    if task_prompt:
        # For base Florence-2, most tasks don't accept additional text
        # Only use task_prompt alone unless it's a grounding/VQA task
        if task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>",
                           "<OCR>", "<OCR_WITH_REGION>", "<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>"]:
            # Base tasks: use task prompt alone
            full_prompt = task_prompt
        else:
            # Fine-tuned tasks (e.g., <DocVQA>): append the question
            full_prompt = task_prompt + prompt
    else:
        # No task prompt: use prompt as-is
        full_prompt = prompt

    # Process inputs
    inputs = processor(text=full_prompt, images=image, return_tensors="pt")

    # Move to device with proper dtype
    inputs = {
        k: v.to(device).to(model.dtype)
        if v.dtype.is_floating_point
        else v.to(device)
        for k, v in inputs.items()
    }

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3
        )

    # Decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-process if task_prompt was used
    if task_prompt:
        try:
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height)
            )
            # Extract the actual text from parsed answer
            if isinstance(parsed_answer, dict):
                task_key = list(parsed_answer.keys())[0]
                result = parsed_answer[task_key]
                # Handle different result types
                if isinstance(result, str):
                    return result
                elif isinstance(result, dict) and 'text' in result:
                    return result['text']
                else:
                    return str(result)
            return str(parsed_answer)
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}. Using raw output.")
            # Fallback: extract text after the prompt
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if full_prompt in generated_text:
                return generated_text[len(full_prompt):].strip()
            return generated_text.strip()
    else:
        # No task prompt - just clean up and return
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if full_prompt in generated_text:
            return generated_text[len(full_prompt):].strip()
        return generated_text.strip()


def process_batch(
    model,
    processor,
    batch_input_file: Path,
    output_dir: Path,
    device: str = "cuda",
    task_prompt: str = None,
    max_new_tokens: int = 1024,
    max_image_size: int = None
) -> Dict[str, Any]:
    """
    Process a batch of images from JSONL input.

    Args:
        model: Loaded model
        processor: Model processor
        batch_input_file: Path to JSONL file with batch inputs
        output_dir: Directory to save outputs
        device: Device to use
        task_prompt: Optional task prefix (e.g., "<CAPTION>", "<DocVQA>")
        max_new_tokens: Maximum tokens to generate per request
        max_image_size: Optional max image dimension for resizing

    Returns:
        dict: Summary statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read batch input
    with open(batch_input_file) as f:
        requests = [json.loads(line) for line in f]

    print(f"\nProcessing {len(requests)} requests...")
    print(f"Task prompt: {task_prompt or 'None (using prompt as-is)'}")
    if max_image_size:
        print(f"Max image size: {max_image_size}px")
    print()

    total = len(requests)
    for idx, request in enumerate(requests, 1):
        request_id = request.get("id", f"request_{idx}")
        image_paths = request["images"]
        prompt = request["prompt"]
        output_file = request.get("output", f"{request_id}.txt")

        print(f"[{idx}/{total}] Processing {request_id}...", end=" ", flush=True)

        # For Florence-2, we process each image separately
        # (Florence-2 doesn't support multi-image input in base form)
        responses = []
        for img_path in image_paths:
            # Load image
            image = load_image(img_path, max_image_size)

            # Run inference
            response = run_inference(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                task_prompt=task_prompt,
                device=device,
                max_new_tokens=max_new_tokens
            )
            responses.append(response)

        # Combine responses if multiple images
        if len(responses) > 1:
            final_response = "\n\n".join(f"Image {i+1}:\n{resp}" for i, resp in enumerate(responses))
        else:
            final_response = responses[0]

        # Save output
        output_path = output_dir / output_file
        output_path.write_text(final_response, encoding='utf-8')

        print(f"✓ Saved to {output_file}")

    print(f"\n✓ Completed {total} requests")
    print(f"  Output directory: {output_dir}")

    return {
        "total_requests": total,
        "output_dir": str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Florence-2 batch inference from JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
JSONL Input Format:
Each line should be a JSON object with:
  {
    "images": ["path/to/image1.png"],  // Single image (multi-image processes separately)
    "prompt": "Your question or prompt here",
    "id": "unique_request_id",
    "output": "output_filename.txt"
  }

Task Prompts:
  Base Florence-2 models use task prefixes:
    --task-prompt "<CAPTION>"              # Image captioning
    --task-prompt "<DETAILED_CAPTION>"     # Detailed caption
    --task-prompt "<OCR>"                  # Text extraction
    --task-prompt "<OCR_WITH_REGION>"      # OCR with bounding boxes

  Fine-tuned DocVQA models use:
    --task-prompt "<DocVQA>"               # Document question answering

  Or omit --task-prompt to use the prompt as-is (for pre-prompted inputs)

Example:
  python batch_predict.py \\
    --model-id microsoft/Florence-2-large \\
    --batch-input batch_requests.jsonl \\
    --output-dir outputs/ \\
    --task-prompt "<CAPTION>"
"""
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/Florence-2-large",
        help="HuggingFace model ID (default: microsoft/Florence-2-large)"
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
        "--task-prompt",
        type=str,
        default=None,
        help="Task prompt prefix (e.g., '<CAPTION>', '<OCR>', '<DocVQA>'). Omit to use prompt as-is."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Maximum image dimension (width or height). Images larger than this will be resized maintaining aspect ratio. Default: no resizing"
    )

    args = parser.parse_args()

    print("="*60)
    print("Florence-2 Batch Inference")
    print("="*60)
    print(f"Model: {args.model_id}")
    print(f"Batch input: {args.batch_input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*60)

    # Load model
    model, processor = load_model(args.model_id, args.device)

    # Process batch
    process_batch(
        model=model,
        processor=processor,
        batch_input_file=args.batch_input,
        output_dir=args.output_dir,
        device=args.device,
        task_prompt=args.task_prompt,
        max_new_tokens=args.max_new_tokens,
        max_image_size=args.max_image_size
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
