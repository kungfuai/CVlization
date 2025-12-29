#!/usr/bin/env python3
"""
Minimal client for the vLLM example.

Modes:
- chat: run vLLM.LLM inside this process (no server needed).
- embed: compute embeddings locally with transformers (encoder or decoder models).
- rerank: run a simple cross-encoder scoring via sentence-transformers/transformers if available.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from cvlization.paths import resolve_input_path, resolve_output_path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# Configure Flash Attention before importing vLLM
from gpu_utils import configure_flash_attn_for_gpu
configure_flash_attn_for_gpu()

from vllm import LLM, SamplingParams


def build_messages(user_prompt: str, system_prompt: str) -> List[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def run_local_chat(args, messages: List[dict]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    outputs = llm.generate(
        prompt,
        sampling_params=SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
    )
    return outputs[0].outputs[0].text


def save_output(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Saved output to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test the vLLM example (chat/embedding/rerank)")
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
    if args.mode == "chat":
        messages = build_messages(args.prompt, args.system)
        text = run_local_chat(args, messages)
        print("Response:\n")
        print(text.strip())
        save_output(text.strip(), Path(resolve_output_path(str(args.output))))
    elif args.mode == "embed":
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
