#!/usr/bin/env python3
"""
Minimal chat client for the SGLang example.

Spins up a local SGLang HTTP server (OpenAI-compatible) inside this process,
hits it with the OpenAI Python client, then tears it down.

Supports both text-only LLMs and vision-language models (VLMs) via --image flag.
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
from gpu_utils import get_optimal_attention_backend
from openai import OpenAI
from PIL import Image


def load_image(src: str, max_size: int = 1280) -> Image.Image:
    """Load an image from a local path or URL, optionally resizing."""
    if src.startswith(("http://", "https://")):
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(src).convert("RGB")

    # Resize if too large (helps with memory)
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_messages(user_prompt: str, system_prompt: str, image: Optional[Image.Image] = None) -> List[dict]:
    """Build chat messages, optionally including an image for VLMs."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    if image is not None:
        # Multimodal message format for VLMs (OpenAI-compatible)
        b64 = image_to_base64(image)
        msgs.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user_prompt},
            ],
        })
    else:
        msgs.append({"role": "user", "content": user_prompt})
    return msgs


def wait_for_server(port: int, timeout: int = 180):
    url = f"http://127.0.0.1:{port}/get_model_info"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"SGLang server did not become ready within {timeout}s")


def launch_server(args) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--tokenizer-path",
        args.tokenizer or args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--context-length",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.extra_args:
        cmd.extend(args.extra_args.split())
    # Auto-detect attention backend if not explicitly specified
    if "--attention-backend" not in " ".join(cmd):
        backend = get_optimal_attention_backend()
        cmd.extend(["--attention-backend", backend])
        print(f"Auto-selected attention backend: {backend}")
    print("Starting SGLang server:", " ".join(cmd))
    return subprocess.Popen(cmd, env=os.environ.copy())


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
    parser = argparse.ArgumentParser(description="SGLang inference (chat, supports VLMs via --image)")
    parser.add_argument("--prompt", default="Give me one bullet on why SGLang is fast.",
                        help="User message to send.")
    parser.add_argument("--system", default=os.getenv("SYSTEM_PROMPT", ""),
                        help="Optional system prompt.")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "allenai/Olmo-3-7B-Instruct"),
                        help="Model ID / served model name.")
    parser.add_argument("--tokenizer", default=os.getenv("TOKENIZER_PATH"),
                        help="Tokenizer path (defaults to model id).")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "30000")),
                        help="Port for the local server.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                        help="Host for the local server.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--output", type=Path, default=Path("outputs/result.txt"),
                        help="Path to save output text.")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "sk-noauth"),
                        help="Dummy API key for OpenAI-compatible endpoint.")
    # VLM support
    parser.add_argument("--image", default=os.getenv("IMAGE_PATH"),
                        help="Path or URL to an image (enables VLM mode).")
    parser.add_argument("--max-image-size", type=int, default=1280,
                        help="Max dimension for image resizing.")
    # Server knobs
    parser.add_argument("--dtype", default=os.getenv("SGLANG_DTYPE", "bfloat16"),
                        help="dtype passed to sglang.launch_server.")
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=int(os.getenv("SGLANG_TP_SIZE", "1")),
                        help="Tensor parallelism.")
    parser.add_argument("--max-model-len", type=int,
                        default=int(os.getenv("SGLANG_CONTEXT_LENGTH", "4096")),
                        help="Max context length.")
    parser.add_argument("--mem-fraction-static", type=float,
                        default=float(os.getenv("SGLANG_MEM_FRACTION_STATIC", "0.9")),
                        help="Memory fraction for weights+KV.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        default=os.getenv("TRUST_REMOTE_CODE", "1") != "0",
                        help="Allow custom modeling files.")
    parser.add_argument("--extra-args", default=os.getenv("SGLANG_EXTRA_ARGS", ""),
                        help="Extra args forwarded to sglang.launch_server.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image if provided (VLM mode)
    image = None
    if args.image:
        image_path = resolve_input_path(args.image) if not args.image.startswith(("http://", "https://")) else args.image
        image = load_image(image_path, args.max_image_size)
        print(f"Loaded image: {args.image} ({image.size[0]}x{image.size[1]})")

    proc = launch_server(args)
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
        text = resp.choices[0].message.content
        print("Response:\n")
        print(text.strip())
        save_output(text.strip(), Path(resolve_output_path(str(args.output))))
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
