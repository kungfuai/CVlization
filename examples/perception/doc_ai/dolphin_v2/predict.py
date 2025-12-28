#!/usr/bin/env python3
"""
Run Dolphin-v2 (ByteDance) for document parsing on a single image.
"""
import argparse
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)

MODEL_ID = "ByteDance/Dolphin-v2"
DEFAULT_IMAGE = "/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg"
DEFAULT_PROMPT = (
    "Parse this document. Return structured text: keep tables as HTML, formulas as "
    "LaTeX, lists and headings marked, and preserve reading order."
)


def load_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load image at {path}") from exc


def move_to_device(batch: Dict, device: str) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dolphin-v2 document parsing inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image (default: bundled sample).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Instruction for the model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.txt",
        help="Where to write the generated text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run on (default: cuda if available).",
    )
    args = parser.parse_args()

    input_dir = get_input_dir()
    output_dir = get_output_dir()

    # Resolve paths: None means use bundled sample, otherwise resolve to user's cwd
    if args.image is None:
        image_path = Path(DEFAULT_IMAGE)
        print(f"No --image provided, using bundled sample: {image_path}")
    else:
        image_path = Path(resolve_input_path(args.image, input_dir))
    if not image_path.exists() and args.image is None:
        fallbacks = [
            Path("/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg"),
            Path(__file__).resolve().parent.parent / "leaderboard" / "test_data" / "sample.jpg",
        ]
        for cand in fallbacks:
            if cand.exists():
                print(f"Input not found at {image_path}, using sample: {cand}")
                image_path = cand
                break
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path = Path(resolve_output_path(args.output, output_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_ID} on {device} (dtype={dtype})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model.to(device)

    image = load_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = move_to_device(inputs, model.device)

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    output_path.write_text(output_text, encoding="utf-8")

    print(f"\nPrompt: {args.prompt}")
    print(f"Image: {image_path}")
    preview = output_text[:800]
    print(f"\nOutput preview (truncated to 800 chars):\n{preview}")
    print(f"\nSaved full output to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
