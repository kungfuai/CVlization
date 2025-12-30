#!/usr/bin/env python3
"""
Transformers-based captioning for fancyfeast/llama-joycaption-beta-one-hf-llava.
"""
import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except Exception:
    resolve_input_path = None  # type: ignore
    resolve_output_path = None  # type: ignore


# Default bundled sample
DEFAULT_IMAGE = "examples/sample.jpg"
DEFAULT_MODEL = os.environ.get("JOYCAPTION_MODEL_ID", "fancyfeast/llama-joycaption-beta-one-hf-llava")


def fetch_to_tmp(url: str, suffix: str = ".img") -> Path:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    fd, tmp_path = tempfile.mkstemp(prefix="joycaption_", suffix=suffix, dir="/tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return Path(tmp_path)


def load_image(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        path = fetch_to_tmp(src, Path(src).suffix or ".img")
    else:
        path = Path(src)
    return Image.open(path).convert("RGB")


def save_output(text: str, path: Path, fmt: str, model_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "text": text,
            "model": model_id,
            "timestamp": datetime.now().isoformat(),
        }
        import json

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")
    print(f"Saved to {path}")


def resolve_in(path: str) -> str:
    if resolve_input_path:
        try:
            return resolve_input_path(path)
        except Exception:
            return path
    return path


def resolve_out(path: str) -> Path:
    if resolve_output_path:
        try:
            return Path(resolve_output_path(path))
        except Exception:
            return Path(path)
    return Path(path)


def run_caption(
    model_id: str,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt},
    ]
    convo_str = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

    # Processor returns pixel_values in float32; cast to bf16 when on CUDA
    inputs = processor(text=[convo_str], images=[image], return_tensors="pt").to(device)
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
        )[0]
    trimmed = generate_ids[inputs["input_ids"].shape[1]:]
    caption = processor.tokenizer.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return caption


def parse_args():
    parser = argparse.ArgumentParser(description="JoyCaption (LLaVA) inference")
    parser.add_argument("--image", help="Path or URL to image.")
    parser.add_argument("--prompt", default="Write a long descriptive caption for this image in a formal tone.", help="Prompt text.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="Model to load.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation cap.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--output", default="joycaption.txt", help="Save path.")
    parser.add_argument("--format", choices=["txt", "json"], default="txt")
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle bundled sample vs user-provided input
    if not args.image:
        img_path = DEFAULT_IMAGE
        print(f"No --image provided, using bundled sample: {img_path}")
    else:
        img_path = resolve_in(args.image)

    img = load_image(img_path)

    caption = run_caption(
        args.model_id,
        img,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    print("\n=== Caption ===")
    print(caption)
    print("===============")

    save_output(caption, resolve_out(args.output), args.format, args.model_id)


if __name__ == "__main__":
    main()
