#!/usr/bin/env python3
"""
Minimal client for the vLLM example.

Modes:
- chat: run vLLM.LLM inside this process (no server needed). Supports both text LLMs and VLMs.
- embed: compute embeddings locally with transformers (encoder or decoder models).
- rerank: run a simple cross-encoder scoring via sentence-transformers/transformers if available.
"""

from __future__ import annotations

import os
# Set spawn method early to avoid CUDA fork issues with VLMs
# Must be set BEFORE importing vllm or torch
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
from pathlib import Path
from typing import List, Optional

from cvlization.paths import resolve_input_path, resolve_output_path
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForSequenceClassification
from PIL import Image
import torch

# Configure Flash Attention before importing vLLM
from gpu_utils import configure_flash_attn_for_gpu
configure_flash_attn_for_gpu()

from vllm import LLM, SamplingParams


def load_image(src: str, max_size: int = 1280) -> Image.Image:
    """Load an image from a local path or URL, optionally resizing."""
    if src.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
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


def build_messages(user_prompt: str, system_prompt: str, image: Optional[Image.Image] = None) -> List[dict]:
    """Build chat messages, optionally including an image for VLMs."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if image is not None:
        # Multimodal message format for VLMs
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": user_prompt})
    return messages


def run_local_chat(args, messages: List[dict], image: Optional[Image.Image] = None) -> str:
    """Run chat inference with vLLM. Supports both text-only and vision-language models."""
    # Try AutoProcessor first (needed for VLMs), fall back to AutoTokenizer
    try:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Initialize vLLM engine
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Use multimodal API if image provided, otherwise text-only
    if image is not None:
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
    else:
        inputs = prompt

    outputs = llm.generate(inputs, sampling_params=sampling)
    return outputs[0].outputs[0].text


def save_output(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Saved output to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM inference (chat/embedding/rerank, text or VLM)")
    parser.add_argument("--prompt", default="Give me one bullet on why vLLM is fast.",
                        help="User message to send.")
    parser.add_argument("--system", default=os.getenv("SYSTEM_PROMPT", ""),
                        help="Optional system prompt.")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "allenai/Olmo-3-7B-Instruct"),
                        help="Model ID / served model name.")
    parser.add_argument("--mode", choices=["chat", "embed", "rerank"],
                        default=os.getenv("VLLM_MODE", "chat"),
                        help="chat|embed|rerank")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--output", type=Path, default=Path("outputs/result.txt"),
                        help="Path to save output text.")
    # VLM support
    parser.add_argument("--image", default=os.getenv("IMAGE_PATH"),
                        help="Path or URL to an image (enables VLM mode).")
    parser.add_argument("--max-image-size", type=int, default=1280,
                        help="Max dimension for image resizing.")
    # Local chat knobs
    parser.add_argument("--dtype", default=os.getenv("VLLM_DTYPE", "bfloat16"),
                        help="Local vLLM dtype (e.g., bfloat16, float16, auto).")
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=int(os.getenv("VLLM_TP_SIZE", "1")),
                        help="Local vLLM tensor parallelism.")
    parser.add_argument("--max-model-len", type=int,
                        default=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
                        help="Local vLLM max context length.")
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
                        help="Fraction of GPU memory to use (local mode).")
    parser.add_argument("--enforce-eager", action="store_true",
                        default=os.getenv("VLLM_ENFORCE_EAGER", "1") != "0",
                        help="Disable torch.compile to reduce memory/time on small GPUs.")
    # Embedding / rerank knobs
    parser.add_argument("--text-a", default=None, help="Text A (for rerank or embedding)")
    parser.add_argument("--text-b", default=None, help="Text B (for rerank); if not set, falls back to prompt.")
    parser.add_argument("--doc", dest="docs", action="append",
                        help="Candidate document (repeatable) for rerank mode.")
    parser.add_argument("--docs-file", type=Path,
                        help="Optional file with one candidate document per line for rerank mode.")
    parser.add_argument("--normalize", action="store_true", help="L2 normalize embeddings.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image if provided (VLM mode)
    image = None
    if args.image:
        image_path = resolve_input_path(args.image) if not args.image.startswith(("http://", "https://")) else args.image
        image = load_image(image_path, args.max_image_size)
        print(f"Loaded image: {args.image} ({image.size[0]}x{image.size[1]})")

    if args.mode == "chat":
        messages = build_messages(args.prompt, args.system, image)
        text = run_local_chat(args, messages, image)
        print("Response:\n")
        print(text.strip())
        save_output(text.strip(), Path(resolve_output_path(str(args.output))))
    elif args.mode == "embed":
        if image:
            print("Warning: --image is ignored in embed mode")
        # Simple embedding pipeline: mean pooling on last hidden states (CLS if available)
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        mod = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mod.to(device)
        txt = args.text_a or args.prompt
        inputs = tok(txt, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = mod(**inputs)
            if hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state.mean(dim=1)
            elif hasattr(out, "pooler_output"):
                emb = out.pooler_output
            else:
                raise RuntimeError("No usable embeddings from model output")
        if args.normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vec = emb[0].cpu().tolist()
        text = f"embedding_dim={len(vec)} sample={vec[:8]}"
        print("Embedding:", text)
        save_output(text, Path(resolve_output_path(str(args.output))))
    elif args.mode == "rerank":
        if image:
            print("Warning: --image is ignored in rerank mode")
        # Minimal cross-encoder scoring: score query (text_a/prompt) against one or more documents
        query = args.text_a or args.prompt
        docs: List[str] = []
        if args.docs:
            docs.extend(args.docs)
        if args.docs_file:
            docs_file_path = Path(resolve_input_path(str(args.docs_file)))
            file_docs = [line.strip() for line in docs_file_path.read_text().splitlines() if line.strip()]
            docs.extend(file_docs)
        if not docs and args.text_b:
            docs.append(args.text_b)
        if not docs:
            docs.append(args.prompt)
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        mod = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mod.to(device)
        inputs = tok([query] * len(docs), docs, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = mod(**inputs).logits
        scores = logits.view(-1).tolist()
        lines = []
        for idx, (doc, score) in enumerate(zip(docs, scores)):
            preview = (doc[:160] + "...") if len(doc) > 160 else doc
            lines.append(f"{idx}\t{score:.4f}\t{preview}")
        text = "rerank_scores:\n" + "\n".join(lines)
        print("Rerank:\n", text)
        save_output(text, Path(resolve_output_path(str(args.output))))
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
