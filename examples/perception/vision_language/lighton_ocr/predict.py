#!/usr/bin/env python3
"""
Local LightOnOCR-1B inference with vLLM.generate (image/PDF inputs).
"""
import os
# Set spawn method early to avoid CUDA fork issues with vLLM
# Must be set BEFORE importing vllm
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
import pypdfium2 as pdfium
from transformers import AutoProcessor, set_seed
from vllm import LLM, SamplingParams

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except Exception:
    resolve_input_path = None  # type: ignore
    resolve_output_path = None  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = "/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg"
# DEFAULT_MODEL_ID = os.environ.get("LIGHTON_OCR_MODEL_ID", "lightonai/LightOnOCR-1B-1025")
DEFAULT_MODEL_ID = os.environ.get("LIGHTON_OCR_MODEL_ID", "lightonai/LightOnOCR-2-1B")
MAX_LONG_SIDE = 1540


def fetch_to_tmp(url: str, suffix: str) -> Path:
    """Download a remote file to /tmp."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    fd, tmp_path = tempfile.mkstemp(prefix="lighton_ocr_", suffix=suffix, dir="/tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return Path(tmp_path)


def load_image(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        tmp_path = fetch_to_tmp(src, Path(src).suffix or ".img")
        path = tmp_path
    else:
        path = Path(src)
    img = Image.open(path).convert("RGB")
    return resize_longest(img, MAX_LONG_SIDE)


def render_pdf(src: str, page_index: int = 0, dpi: int = 200) -> Image.Image:
    if src.startswith(("http://", "https://")):
        path = fetch_to_tmp(src, ".pdf")
    else:
        path = Path(src)
    pdf = pdfium.PdfDocument(str(path))
    if page_index < 0 or page_index >= len(pdf):
        raise ValueError(f"PDF has {len(pdf)} pages; page_index {page_index} is out of range.")
    page = pdf[page_index]
    # 200 DPI ≈ 1540px on the longest side for A4
    pil = page.render(scale=dpi / 72.0).to_pil()
    return resize_longest(pil, MAX_LONG_SIDE)


def resize_longest(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= target:
        return img
    scale = target / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def save_output(text: str, path: Path, fmt: str, model_id: str, source: str = "local-vllm"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "text": text,
            "model": model_id,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
        path.write_text(json_dumps(payload), encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")
    print(f"Saved to {path}")


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="LightOnOCR local inference with vLLM.generate")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image", help="Path or URL to an image.")
    input_group.add_argument("--pdf", help="Path or URL to a PDF (first page by default).")
    parser.add_argument("--page", type=int, default=0, help="PDF page index (0-based).")
    parser.add_argument("--prompt", default="Extract all text from this image.", help="Custom prompt.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model to load.")
    parser.add_argument("--tp-size", type=int, default=int(os.environ.get("TENSOR_PARALLEL_SIZE", 1)), help="Tensor parallel shards.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override max model length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="GPU memory utilization (0.0-1.0).")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Generation cap.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--output", default="outputs/lighton_ocr.txt", help="Where to save the result.")
    parser.add_argument("--format", choices=["txt", "json"], default="txt")
    return parser.parse_args()


def resolve_path(path: str) -> str:
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


def run_request_local(img: Image.Image, prompt: str, model_id: str, tp_size: int, max_model_len: Optional[int], gpu_memory_utilization: float, max_tokens: int, temperature: float, top_p: float) -> str:
    # Initialize LLM FIRST to spawn worker processes before any CUDA init
    # (AutoProcessor with trust_remote_code may trigger CUDA, causing fork issues)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # vLLM 0.11+ API: pass multi_modal_data inside the prompt dict
    outputs = llm.generate(
        {
            "prompt": prompt_text,
            "multi_modal_data": {"image": img},
        },
        sampling_params=sampling,
    )
    return outputs[0].outputs[0].text


def main():
    args = parse_args()

    if not args.image and not args.pdf:
        args.image = str(DEFAULT_IMAGE)

    if args.image:
        src = resolve_path(args.image)
        img = load_image(src)
    else:
        src = resolve_path(args.pdf)
        img = render_pdf(src, page_index=args.page)

    set_seed(42)

    print(f"Running LightOnOCR locally with vLLM")
    print(f"Model: {args.model_id}")
    print(f"Prompt: {args.prompt}")
    text = run_request_local(
        img,
        args.prompt,
        args.model_id,
        args.tp_size,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    print("\n=== Response ===")
    print(text)
    print("================\n")

    save_output(text, resolve_out(args.output), args.format, args.model_id, "local-vllm")


if __name__ == "__main__":
    main()
