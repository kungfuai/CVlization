#!/usr/bin/env python3
"""
Send a multimodal chat completion request to an OpenAI-compatible (vLLM) endpoint.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import List

import requests


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{path.suffix.lstrip('.').lower() or 'png'};base64,{b64}"


def build_content(images: List[Path], prompt: str):
    content = [{"type": "image_url", "image_url": {"url": encode_image(img)}} for img in images]
    content.append({"type": "text", "text": prompt})
    return content


def extract_text(choice_content):
    if isinstance(choice_content, list):
        parts = []
        for part in choice_content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(text)
        return "".join(parts).strip()
    if isinstance(choice_content, str):
        return choice_content.strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description="Send multimodal request to OpenAI-compatible API")
    parser.add_argument("--api-base", required=True, help="Base URL of the API (e.g., http://localhost:8000/v1)")
    parser.add_argument("--model", required=True, help="Model name served by vLLM")
    parser.add_argument("--prompt", required=True, help="User prompt/question")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths to include (ordered)")
    parser.add_argument("--output", required=True, help="Where to write the model response")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))

    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    url = f"{api_base}/chat/completions" if api_base.endswith("/v1") else f"{api_base}/v1/chat/completions"

    images = [Path(p) for p in args.images]
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({
        "role": "user",
        "content": build_content(images, args.prompt)
    })

    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]["message"]
    content = extract_text(choice.get("content", ""))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
