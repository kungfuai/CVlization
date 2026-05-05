#!/usr/bin/env python3
"""Inference with a TIDE-distilled diffusion language model.

Loads a BD3LM checkpoint (from HuggingFace or a local path) and generates
text via iterative block-diffusion demasking using the dllm library.

Pre-trained TIDE checkpoints on HuggingFace:
  TIDE-dllm/distill-WeDLM-TIDE_Shared   (Pipeline B, shared tokenizer)
  TIDE-dllm/distill-LLaDA2-TIDE_Cross   (Pipeline A, cross tokenizer)

The base (undistilled) model also works:
  dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1
"""

import os
import sys
import logging
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.ERROR)
for _name in ("transformers", "torch", "accelerate"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

import torch
import transformers
import dllm

try:
    from cvlization.paths import get_output_dir, resolve_output_path
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_output_dir():
        d = os.path.join(os.getcwd(), "outputs")
        os.makedirs(d, exist_ok=True)
        return d

    def resolve_output_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)


MODELS = {
    "tide-wedlm-shared": {
        "id": "TIDE-dllm/distill-WeDLM-TIDE_Shared",
        "desc": "Pipeline B (shared tokenizer, TIDAL+CompDemo)",
    },
    "tide-llada-cross": {
        "id": "TIDE-dllm/distill-LLaDA2-TIDE_Cross",
        "desc": "Pipeline A (cross tokenizer, Reverse CALM)",
    },
    "base": {
        "id": "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
        "desc": "Base Qwen3-0.6B BD3LM (undistilled)",
    },
}


class _ModelArgs:
    def __init__(self, path):
        self.model_name_or_path = path


def load_model(model_key_or_path: str):
    """Load a BD3LM model and return (model, tokenizer, sampler)."""
    if model_key_or_path in MODELS:
        model_id = MODELS[model_key_or_path]["id"]
    else:
        model_id = model_key_or_path

    print(f"Loading model: {model_id}")
    args = _ModelArgs(model_id)
    model = dllm.utils.get_model(model_args=args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=args)
    sampler = dllm.core.samplers.BD3LMSampler(model=model, tokenizer=tokenizer)
    return model, tokenizer, sampler


def run_inference(sampler, tokenizer, prompt, steps=128,
                  max_new_tokens=256, block_size=32, temperature=0.0):
    messages = [[{"role": "user", "content": prompt}]]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    config = dllm.core.samplers.BD3LMSamplerConfig(
        steps=steps,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        remasking="low_confidence",
        right_shift_logits=False,
    )
    outputs = sampler.sample(inputs, config, return_dict=True)
    seqs = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
    return seqs[0].strip() or "<empty>"


def main():
    p = argparse.ArgumentParser(
        description="Inference with TIDE-distilled dLLM",
        epilog="""Available models:
  tide-wedlm-shared : TIDE Pipeline B distilled (TIDAL+CompDemo)
  tide-llada-cross  : TIDE Pipeline A distilled (Reverse CALM)
  base              : Undistilled Qwen3-0.6B BD3LM
  <path/or/hf-id>   : Any BD3LM-compatible checkpoint
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="base",
                   help="Model key or HF ID / local path")
    p.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    p.add_argument("--output", default=None)
    p.add_argument("--format", choices=["txt", "json"], default="txt")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    if args.list_models:
        print("Available models:")
        for k, v in MODELS.items():
            print(f"  {k:20} : {v['id']}")
            print(f"  {' '*20}   {v['desc']}")
        return 0

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for name in ("transformers", "torch", "accelerate"):
            logging.getLogger(name).setLevel(logging.INFO)

    transformers.set_seed(args.seed)

    OUT = get_output_dir()
    if args.output is None:
        ext = "json" if args.format == "json" else "txt"
        args.output = f"result.{ext}"
    output_path = resolve_output_path(args.output, OUT)

    model, tokenizer, sampler = load_model(args.model)

    print(f"\nPrompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    print(f"Diffusion steps: {args.steps}, block_size: {args.block_size}")

    text = run_inference(
        sampler, tokenizer, args.prompt,
        steps=args.steps, max_new_tokens=args.max_tokens,
        block_size=args.block_size, temperature=args.temperature,
    )

    print("\n=== Output ===")
    print(text)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        data = {
            "text": text, "model": args.model, "prompt": args.prompt,
            "steps": args.steps, "max_tokens": args.max_tokens,
            "block_size": args.block_size, "temperature": args.temperature,
        }
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        out.write_text(text)
    print(f"Output saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
