#!/usr/bin/env python3
"""
HuggingFace Transformers inference for chat/instruct LLMs.

Loads any causal LM via AutoModelForCausalLM and runs chat-template generation.
Works with standard transformer models and hybrid architectures like OLMo-Hybrid
that require trust_remote_code and are not yet well-supported by vLLM/SGLang.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except ImportError:
    def resolve_input_path(path, base=None):
        return path
    def resolve_output_path(path, base=None):
        return path


def build_messages(prompt: str, system: str) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def run_inference(args) -> str:
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    messages = build_messages(args.prompt, args.system)
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def save_output(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Saved output to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HuggingFace Transformers chat inference"
    )
    parser.add_argument("--prompt", default="Give me one fun fact about language models.",
                        help="User message.")
    parser.add_argument("--system", default=os.getenv("SYSTEM_PROMPT", ""),
                        help="Optional system prompt.")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "allenai/Olmo-Hybrid-Instruct-DPO-7B"),
                        help="HuggingFace model ID.")
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
