#!/usr/bin/env python3
"""
Transformers-based runner for Llama 3.2 Vision Instruct (single-image, single-turn).

Example:
  python predict.py --image test_images/checkbox_page.png --prompt "Describe the scene."
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor

DEFAULT_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"


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


def load_model(model_id: str, device: Optional[str]):
    if device is None:
        device = detect_device()

    print(f"Loading {model_id} on {device} ...")
    processor = MllamaProcessor.from_pretrained(model_id)

    if device == "cuda":
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
    else:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to(device)

    return model, processor, device


def build_inputs(processor, image: Image.Image, prompt: str, device: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt.strip()},
            ],
        }
    ]
    # Build template text then tokenize with image
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def generate(model, processor, inputs, max_new_tokens: int):
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    # Remove prompt tokens from generated sequence
    prompt_len = inputs["input_ids"].shape[1]
    generated_text = processor.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return generated_text.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Llama 3.2 Vision (transformers) inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID to load.")
    parser.add_argument(
        "--image",
        default="shared_test_images/sample.jpg",
        help="Path to an image.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in one sentence.",
        help="User instruction appended after the image.",
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="Generation limit.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Force a specific device (otherwise auto-detect).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to save the generated text.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = None if args.device == "auto" else args.device

    image = Image.open(args.image).convert("RGB")
    model, processor, device = load_model(args.model, device)
    inputs = build_inputs(processor, image, args.prompt, device)
    text = generate(model, processor, inputs, max_new_tokens=args.max_tokens)

    print("\nPrompt:", args.prompt)
    print("Model :", args.model)
    print("\nGenerated:\n", text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
