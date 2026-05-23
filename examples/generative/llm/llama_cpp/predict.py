#!/usr/bin/env python3
"""Minimal chat client for the llama.cpp example.

Spins up llama-server in this process (using serve.py's launcher), polls until
the OpenAI-compatible HTTP endpoint is ready, sends one chat completion via
the OpenAI Python client, then tears the server down.

Supports both text-only and vision-language GGUF models via `--image`
(llama-server's OpenAI API accepts `image_url` content parts).
"""

from __future__ import annotations

import argparse
import base64
import os
import signal
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
from cvlization.paths import resolve_input_path, resolve_output_path
from openai import OpenAI
from PIL import Image

from serve import build_cmd  # reuse the auto-tuned llama-server cmd builder


def load_image(src: str, max_size: int = 1280) -> Image.Image:
    if src.startswith(("http://", "https://")):
        # Set a UA — some hosts (e.g. wikipedia) 403 the default requests UA.
        headers = {"User-Agent": "cvl-llama-cpp/0.1 (+https://github.com/kungfuai/CVlization)"}
        resp = requests.get(src, timeout=30, headers=headers)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(src).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def image_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(prompt: str, system: str, image: Optional[Image.Image]) -> List[dict]:
    msgs: List[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if image is not None:
        msgs.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        msgs.append({"role": "user", "content": prompt})
    return msgs


def wait_for_server(port: int, timeout: int = 600):
    """llama-server's first-run can download a multi-GB GGUF; allow generous timeout."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"llama-server did not become ready within {timeout}s")


def stop_server(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGKILL)


def save_output(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Saved output to {path}")


def parse_args():
    p = argparse.ArgumentParser(description="llama.cpp inference (chat; supports VLMs via --image)")
    p.add_argument("--prompt", default="Give me one bullet on why llama.cpp is fast.")
    p.add_argument("--system", default=os.getenv("SYSTEM_PROMPT", ""))
    p.add_argument("--model", default=os.getenv("MODEL_ID", "Qwen/Qwen3-8B-GGUF:Q4_K_M"),
                   help="HF repo + quant tag (e.g. Qwen/Qwen3-8B-GGUF:Q4_K_M) — served name only; serve.py reads MODEL_ID for the actual launch.")
    p.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", type=Path, default=Path("outputs/result.txt"))
    p.add_argument("--api-key", default=os.getenv("LLAMA_API_KEY", "sk-noauth"))
    p.add_argument("--image", default=os.getenv("IMAGE_PATH"),
                   help="Path or URL to an image (enables VLM mode).")
    p.add_argument("--max-image-size", type=int, default=1280)
    return p.parse_args()


def main():
    args = parse_args()

    image = None
    if args.image:
        src = resolve_input_path(args.image) if not args.image.startswith(("http://", "https://")) else args.image
        image = load_image(src, args.max_image_size)
        print(f"Loaded image: {args.image} ({image.size[0]}x{image.size[1]})")

    # Make sure serve.build_cmd sees the same host/port the client will hit.
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    cmd = build_cmd()
    print("Spawning llama-server...")
    server = subprocess.Popen(cmd, env=os.environ.copy())
    try:
        wait_for_server(args.port)
        client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key=args.api_key)
        messages = build_messages(args.prompt, args.system, image)
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        text = resp.choices[0].message.content or ""
        reasoning = getattr(resp.choices[0].message, "reasoning", None) \
            or getattr(resp.choices[0].message, "reasoning_content", None)
        print("Response:\n")
        if reasoning:
            print(f"[reasoning ({len(reasoning)} chars)]\n{reasoning}\n")
        print(text.strip())
        save_output(text.strip(), Path(resolve_output_path(str(args.output))))
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
