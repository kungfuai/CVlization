#!/usr/bin/env python3
"""
GLM-5V-Turbo inference runner via the Z.ai API.

GLM-5V-Turbo is a native multimodal agent foundation model from Z.ai,
designed for visual reasoning, GUI grounding, visual tool use, and
multimodal coding. Unlike standard VLMs that bolt vision onto a text
model, GLM-5V-Turbo integrates multimodal perception as a core part
of reasoning, planning, and execution.

This example uses the Z.ai API (OpenAI-compatible) -- no local GPU required.

Usage examples:
  # Caption an image
  python predict.py --image test_images/sample.jpg --task caption

  # GUI grounding on a screenshot
  python predict.py --image screenshot.png --task gui_grounding

  # Visual reasoning with thinking mode
  python predict.py --image diagram.png --task reasoning --thinking

  # Custom prompt
  python predict.py --image chart.png --task vqa --prompt "What trend is visible?"

Requires:
  ZAI_API_KEY environment variable (get one at https://open.z.ai)
"""

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from cvlization.paths import resolve_input_path, resolve_output_path

# Z.ai API endpoint (OpenAI-compatible)
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_ID = "glm-5v-turbo"

TASK_PROMPTS = {
    "caption": "Describe this image in detail.",
    "ocr": "Extract all text from this image. Preserve the layout as closely as possible.",
    "vqa": None,  # requires --prompt
    "gui_grounding": (
        "Identify all interactive UI elements in this screenshot. "
        "For each element, describe its type (button, text field, checkbox, etc.), "
        "its label or text content, and its approximate location in the image."
    ),
    "reasoning": (
        "Analyze this image step by step. Describe what you observe, "
        "identify key elements, and explain any relationships, patterns, "
        "or conclusions you can draw."
    ),
}


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


def build_messages(image_paths: list, prompt: str) -> list:
    content = []
    for path in image_paths:
        if path.startswith("http://") or path.startswith("https://"):
            content.append({
                "type": "image_url",
                "image_url": {"url": path},
            })
        else:
            mime = detect_mime_type(path)
            b64 = encode_image_to_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_inference(
    client: OpenAI,
    image_paths: list,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    thinking: bool = False,
) -> str:
    messages = build_messages(image_paths, prompt)

    kwargs = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def save_output(text: str, path: Path, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "text": text,
            "model": MODEL_ID,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Output saved to {path} (JSON)")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"Output saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GLM-5V-Turbo inference via Z.ai API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image test_images/sample.jpg --task caption
  python predict.py --image screenshot.png --task gui_grounding
  python predict.py --image chart.png --task vqa --prompt "What trend is shown?"
  python predict.py --image diagram.png --task reasoning --thinking
        """,
    )
    parser.add_argument("--image", help="Path/URL to a single image.")
    parser.add_argument("--images", nargs="+", help="Paths/URLs to multiple images.")
    parser.add_argument(
        "--task",
        choices=list(TASK_PROMPTS.keys()),
        default="caption",
        help="Task to run (default: caption).",
    )
    parser.add_argument("--prompt", help="Custom prompt (required for vqa task).")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens (default: 1024).")
    parser.add_argument("--output", default="result.txt", help="Output path.")
    parser.add_argument("--format", choices=["txt", "json"], default="txt")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2).")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking/reasoning mode for step-by-step analysis.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ZAI_API_KEY"),
        help="Z.ai API key (default: ZAI_API_KEY env var).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZAI_BASE_URL", DEFAULT_BASE_URL),
        help="API base URL (default: Z.ai endpoint).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        raise SystemExit(
            "ZAI_API_KEY is required. Set it via --api-key or the ZAI_API_KEY "
            "environment variable.\nGet an API key at https://open.z.ai"
        )
    if args.task == "vqa" and not args.prompt:
        raise SystemExit("--prompt is required for vqa task.")
    if args.image and args.images:
        raise SystemExit("Specify either --image or --images, not both.")

    DEFAULT_IMAGE = "test_images/sample.jpg"
    using_bundled_sample = False
    if not args.image and not args.images:
        args.image = DEFAULT_IMAGE
        using_bundled_sample = True
        print(f"No --image provided, using bundled sample: {DEFAULT_IMAGE}")

    print("=" * 60)
    print("GLM-5V-Turbo Inference (Z.ai API)")
    print(f"Model: {MODEL_ID}")
    print(f"Task: {args.task}")
    print(f"Thinking: {'enabled' if args.thinking else 'disabled'}")
    print("=" * 60)

    # Resolve paths
    if using_bundled_sample:
        image_paths = [args.image]
    elif args.image:
        image_paths = [resolve_input_path(args.image)]
    else:
        image_paths = [resolve_input_path(p) for p in args.images]
    output_path = Path(resolve_output_path(args.output))

    prompt = args.prompt or TASK_PROMPTS[args.task]
    if args.thinking and args.task != "reasoning":
        prompt = f"Think step by step.\n\n{prompt}"

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    print(f"\nProcessing {len(image_paths)} image(s)...")
    response = run_inference(
        client,
        image_paths,
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
    )

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")

    save_output(response, output_path, args.format)


if __name__ == "__main__":
    main()
