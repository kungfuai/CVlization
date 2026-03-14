#!/usr/bin/env python3
"""Generate text with a trained tiny diffusion or GPT language model.

Two generation strategies:
- diffusion: confidence-based parallel decoding (all tokens denoised at once)
- gpt: standard autoregressive left-to-right generation

Based on: https://github.com/nathan-barry/tiny-diffusion
"""

import argparse
import json
import os
import sys
import time

import torch

from model import TinyLM, CharTokenizer, generate_diffusion, generate_gpt

try:
    from cvlization.paths import get_input_dir, get_output_dir, resolve_output_path
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_input_dir():
        return os.getcwd()

    def get_output_dir():
        d = os.path.join(os.getcwd(), "outputs")
        os.makedirs(d, exist_ok=True)
        return d

    def resolve_output_path(path, base):
        return path if os.path.isabs(path) else os.path.join(base, path)


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_data(path):
    if os.path.exists(path):
        return
    import urllib.request
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with tiny diffusion or GPT LM",
    )
    parser.add_argument("--model", choices=["diffusion", "gpt"],
                        default="diffusion",
                        help="Model type (default: diffusion)")
    parser.add_argument("--weights", default=None,
                        help="Path to model weights (default: auto)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer (default: auto)")
    parser.add_argument("--data", default="data.txt",
                        help="Path to text data for seed/tokenizer fallback")
    parser.add_argument("--prompt", default=None,
                        help="Seed text for generation (default: first 16 chars of data)")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Number of characters to generate (default: 2000)")
    parser.add_argument("--prompt-len", type=int, default=16,
                        help="Prompt context length (default: 16)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence threshold for diffusion decoding (default: 0.95)")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Top-k sampling for diffusion (default: 2)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed (default: 1337)")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: auto)")
    parser.add_argument("--format", choices=["txt", "json"], default="txt",
                        help="Output format (default: txt)")
    args = parser.parse_args()

    mode = args.model
    is_diffusion = mode == "diffusion"

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Mode: {mode}")
    torch.manual_seed(args.seed)

    # Resolve paths
    OUT = get_output_dir()
    weights_path = args.weights or os.path.join(OUT, f"{mode}.pt")
    tokenizer_path = args.tokenizer or os.path.join(OUT, f"tokenizer_{mode}.pt")

    # Load tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = CharTokenizer.load(tokenizer_path)
    else:
        print(f"No saved tokenizer at {tokenizer_path}, rebuilding from data")
        download_data(args.data)
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = CharTokenizer(text, add_mask_token=is_diffusion)

    print(f"Vocabulary: {tokenizer.vocab_size} characters")

    # Load model
    if not os.path.exists(weights_path):
        print(f"Error: weights not found at {weights_path}")
        print(f"Run train.py --model {mode} first or specify --weights")
        return 1

    model = TinyLM(tokenizer.vocab_size, is_causal=not is_diffusion).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({param_count / 1e6:.1f}M params) from {weights_path}")

    # Seed text
    if args.prompt:
        seed_text = args.prompt
    else:
        download_data(args.data)
        with open(args.data, "r", encoding="utf-8") as f:
            seed_text = f.read()[:args.prompt_len]

    print(f"Seed: {repr(seed_text[:50])}")
    print(f"Generating {args.max_tokens} characters (temp={args.temperature})")

    # Generate
    start = time.time()
    if is_diffusion:
        output = generate_diffusion(
            model, tokenizer, device,
            seed_text=seed_text,
            max_new_tokens=args.max_tokens,
            prompt_len=args.prompt_len,
            temp=args.temperature,
            confidence_threshold=args.confidence,
            top_k=args.top_k,
        )
    else:
        output = generate_gpt(
            model, tokenizer, device,
            seed_text=seed_text,
            max_new_tokens=args.max_tokens,
            prompt_len=args.prompt_len,
            temp=args.temperature,
        )
    elapsed = time.time() - start
    print(f"Generation time: {elapsed:.2f}s")

    # Display
    print("\n=== Output ===")
    print(output)

    # Save
    if args.output is None:
        ext = "json" if args.format == "json" else "txt"
        output_path = resolve_output_path(f"result_{mode}.{ext}", OUT)
    else:
        output_path = resolve_output_path(args.output, OUT)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if args.format == "json":
        data = {
            "model": mode,
            "text": output,
            "seed": seed_text[:args.prompt_len],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "generation_time_s": round(elapsed, 2),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, "w") as f:
            f.write(output)

    print(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
