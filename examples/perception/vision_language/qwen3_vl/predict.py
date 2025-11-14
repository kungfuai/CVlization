#!/usr/bin/env python3
"""
Unified Qwen3-VL runner covering 2B / 4B / 8B Instruct checkpoints.

Usage examples:
  python predict.py --variant 2b --image test_images/sample.jpg --task caption
  python predict.py --variant 4b --images page1.png page2.png --task ocr
  python predict.py --model-id Qwen/Qwen3-VL-8B-Instruct --task vqa --prompt "What is checked?"
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from cvlization.paths import (
    resolve_input_path,
    resolve_output_path,
)

MODEL_VARIANTS = {
    "2b": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "max_new_tokens": 128,
        "vram_gb": "≈4GB",
    },
    "4b": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "max_new_tokens": 512,
        "vram_gb": "≈8GB",
    },
    "8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "max_new_tokens": 512,
        "vram_gb": "≈16GB",
    },
}

TASK_PROMPTS = {
    "caption": "Describe this image in detail.",
    "ocr": "Extract all text from this image.",
    "vqa": None,
}


def detect_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS)")
    else:
        device = "cpu"
        print("Using CPU (no GPU detected)")
    return device


def load_model(model_id: str, device: Optional[str]) -> tuple:
    if device is None:
        device = detect_device()

    print(f"Loading {model_id} on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id)

    if device == "cuda":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
        ).to(device)

    return model, processor


def load_images(image_paths: List[str]):
    images = []
    for path in image_paths:
        if path.startswith("http://") or path.startswith("https://"):
            images.append(path)
        else:
            images.append(Image.open(path).convert("RGB"))
    return images


def prepare_messages(images, prompt: str):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_inference(model, processor, images, prompt, max_new_tokens: int) -> str:
    messages = prepare_messages(images, prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated)
    ]
    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response


def save_output(text: str, path: Path, fmt: str, model_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "text": text,
            "model": model_id,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Output saved to {path} (JSON)")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"Output saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL unified inference script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --variant 2b --task caption
  python predict.py --variant 4b --images page1.png page2.png --task ocr
  python predict.py --model-id Qwen/Qwen3-VL-8B-Instruct --task vqa --prompt "What is checked?"
        """,
    )
    parser.add_argument("--variant", choices=MODEL_VARIANTS.keys(), default="2b",
                        help="Predefined Qwen3-VL model size to use.")
    parser.add_argument("--model-id", default=None,
                        help="Override HuggingFace model ID (takes precedence over --variant).")
    parser.add_argument("--image", help="Path/URL to a single image.")
    parser.add_argument("--images", nargs="+", help="Paths/URLs to multiple images.")
    parser.add_argument("--task", choices=list(TASK_PROMPTS.keys()), default="caption")
    parser.add_argument("--prompt", help="Custom prompt (required for VQA).")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override max tokens when generating.")
    parser.add_argument("--output", default="outputs/result.txt", help="Output path.")
    parser.add_argument("--format", choices=["txt", "json"], default="txt")
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=os.environ.get("QWEN3_VL_DEVICE"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "vqa" and not args.prompt:
        raise SystemExit("--prompt is required for VQA tasks.")
    if args.image and args.images:
        raise SystemExit("Specify either --image or --images, not both.")
    if not args.image and not args.images:
        args.image = "test_images/sample.jpg"

    variant = MODEL_VARIANTS[args.variant]
    model_id = args.model_id or variant["model_id"]
    max_tokens = args.max_new_tokens or variant["max_new_tokens"]

    print("=" * 60)
    print(f"Qwen3-VL Unified Runner")
    print(f"Variant: {args.variant} ({variant['vram_gb']} VRAM)")
    print(f"Model: {model_id}")
    print(f"Task: {args.task}")
    print("=" * 60)

    try:
        if args.image:
            image_paths = [resolve_input_path(args.image)]
        else:
            image_paths = [resolve_input_path(path) for path in args.images]
        output_path = Path(resolve_output_path(args.output))
    except Exception:
        image_paths = [args.image] if args.image else args.images
        output_path = Path(args.output)

    prompt = args.prompt or TASK_PROMPTS[args.task]

    model, processor = load_model(model_id, args.device)
    images = load_images(image_paths)
    response = run_inference(model, processor, images, prompt, max_tokens)

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")

    save_output(response, output_path, args.format, model_id)


if __name__ == "__main__":
    main()
