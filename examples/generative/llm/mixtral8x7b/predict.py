#!/usr/bin/env python
"""Simple text-generation CLI for Mixtral 8x7B (or any causal LM).

The previous script relied on a custom offloading repository and the Hugging
Face CLI.  This version uses the standard transformers API with optional 4-bit
loading so it can run either as a quick smoke test or a full inference job.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:  # noqa: F401 - ensure dependency is present for LLaMA tokenizers
    import sentencepiece  # type: ignore
except ImportError:  # pragma: no cover - fallback for lean images
    logging.getLogger(__name__).info("Installing sentencepiece at runtime...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
    import sentencepiece  # type: ignore  # noqa: E401

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mixtral 8x7B inference with optional 4-bit loading")
    parser.add_argument("--model_id", default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model repository id")
    parser.add_argument("--prompt", default="Tell me a joke about databases.", help="Prompt text")
    parser.add_argument("--prompt_file", type=Path, help="Optional path to a file containing the prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--output_file", type=Path, default=Path("outputs/mixtral-output.txt"))
    parser.add_argument("--json_metadata", type=Path, default=None, help="Optional JSON metadata output path")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=False)

    parser.add_argument("--device", default=None, help="Force device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--trust_remote_code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)

    return parser.parse_args()


def maybe_read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        text = args.prompt_file.read_text().strip()
        if text:
            return text
        LOGGER.warning("Prompt file %s was empty; falling back to command-line prompt", args.prompt_file)
    return args.prompt


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(args: argparse.Namespace):
    hf_token = os.environ.get("HF_TOKEN")
    LOGGER.info("Loading tokenizer from %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=hf_token,
        use_fast=False,
        legacy=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            LOGGER.warning("bitsandbytes not available; falling back to full precision. Install bitsandbytes or rerun with --no_load_in_4bit.")
            args.load_in_4bit = False

    if args.load_in_4bit:
        LOGGER.info("Loading model in 4-bit mode")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
            use_auth_token=hf_token,
        )
    else:
        LOGGER.info("Loading model in full precision")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            use_auth_token=hf_token,
        )
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Moving model to %s", device)
        model.to(device)

    return model, tokenizer


def main() -> None:
    args = parse_args()
    prompt = maybe_read_prompt(args)
    set_seed(args.seed)

    model, tokenizer = load_model(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else model.device)

    LOGGER.info("Generating text (%s)", args.model_id)
    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

    with torch.no_grad():
        generated = model.generate(**input_tokens, generation_config=generation_config)

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    # Extract only the newly generated text (everything after the prompt)
    output_text = output_text[len(prompt) :].strip()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(output_text + "\n")
    LOGGER.info("Saved generations to %s", args.output_file)

    if args.json_metadata:
        args.json_metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_id": args.model_id,
            "prompt": prompt,
            "output_file": str(args.output_file),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        }
        args.json_metadata.write_text(json.dumps(metadata, indent=2))
        LOGGER.info("Saved generation metadata to %s", args.json_metadata)

    print("=== Prompt ===")
    print(prompt)
    print("\n=== Completion ===")
    print(output_text)


if __name__ == "__main__":
    main()
