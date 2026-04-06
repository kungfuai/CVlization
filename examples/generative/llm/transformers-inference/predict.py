#!/usr/bin/env python3
"""
HuggingFace Transformers inference for chat/instruct LLMs and VLMs.

Loads any causal LM or image-text-to-text model via AutoModel and runs
chat-template generation. Supports optional image input for VLMs (Gemma 4,
Granite 4.0 Vision, etc.) via --image.

Works with standard transformer models and hybrid architectures like OLMo-Hybrid
that require trust_remote_code and are not yet well-supported by vLLM/SGLang.
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except ImportError:
    def resolve_input_path(path, base=None):
        return path
    def resolve_output_path(path, base=None):
        return path


def load_image(src: str, max_size: int = 1280) -> Image.Image:
    if src.startswith(("http://", "https://")):
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(src).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def build_messages(prompt: str, system: str, image: Image.Image | None = None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if image is not None:
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def load_processor(model_id: str):
    """Load AutoProcessor (handles both text-only and multimodal models)."""
    try:
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def load_model(model_id: str, dtype: torch.dtype, has_image: bool):
    """Load model, preferring ImageTextToText for VLM mode, CausalLM otherwise."""
    kwargs = dict(dtype=dtype, device_map="auto", trust_remote_code=True)
    if has_image:
        try:
            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        except Exception:
            pass
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def run_inference(args) -> str:
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)

    image = None
    if args.image:
        src = resolve_input_path(args.image) if not args.image.startswith(("http://", "https://")) else args.image
        image = load_image(src)
        print(f"Loaded image: {args.image} ({image.size[0]}x{image.size[1]})")

    processor = load_processor(args.model)
    model = load_model(args.model, dtype, image is not None)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    messages = build_messages(args.prompt, args.system, image)
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    if image is not None:
        inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=text, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else processor.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)


def save_output(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Saved output to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HuggingFace Transformers chat inference (text LLMs and VLMs)"
    )
    parser.add_argument("--prompt", default="Give me one fun fact about language models.",
                        help="User message.")
    parser.add_argument("--system", default=os.getenv("SYSTEM_PROMPT", ""),
                        help="Optional system prompt.")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "allenai/Olmo-Hybrid-Instruct-DPO-7B"),
                        help="HuggingFace model ID.")
    parser.add_argument("--image", default=os.getenv("IMAGE_PATH"),
                        help="Path or URL to an image (enables VLM mode).")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens to generate (default: 256).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0 = greedy (default: 0.0).")
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"),
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (default: bfloat16).")
    parser.add_argument("--output", default="outputs/result.txt",
                        help="Path to save output text.")
    return parser.parse_args()


def main():
    args = parse_args()

    response = run_inference(args)
    print("Response:\n")
    print(response.strip())

    out_path = Path(resolve_output_path(args.output))
    save_output(response.strip(), out_path)


if __name__ == "__main__":
    main()
