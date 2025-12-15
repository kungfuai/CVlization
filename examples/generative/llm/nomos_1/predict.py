#!/usr/bin/env python3
"""
Nomos-1: Mathematical reasoning and proof-writing model.

A 31B parameter MoE model (~3B active) fine-tuned for mathematical problem-solving.
Based on Qwen3-30B-A3B-Thinking, achieves 87/120 on Putnam 2025 benchmark.

License: Apache 2.0
Model: NousResearch/nomos-1
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Configure logging - only show errors by default
logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "torch"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "NousResearch/nomos-1"


def detect_device() -> tuple:
    """Auto-detect device and dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        # Use bfloat16 for modern GPUs (compute capability >= 8)
        major_cc = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        print(f"Using CUDA device with {'bfloat16' if major_cc >= 8 else 'float16'}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Using MPS device with float16")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU with float32")

    return device, dtype


def load_model(model_id: str, device: str = None):
    """Load Nomos-1 model and tokenizer."""
    if device is None:
        device, dtype = detect_device()
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading {model_id}...")
    print("This may take a few minutes for the 31B parameter model...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model loaded successfully")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    """Run inference on Nomos-1 model."""
    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        if do_sample:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Decode only the generated part
    generated_ids = outputs[0][input_ids.shape[1] :]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return output_text


def save_output(text: str, output_path: str, format: str = "txt", prompt: str = None):
    """Save output to file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "model": MODEL_ID,
            "prompt": prompt,
            "response": text,
        }
        output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        output_file.write_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="Nomos-1: Mathematical reasoning and proof-writing model",
        epilog="""Examples:
  python predict.py --prompt "Prove that sqrt(2) is irrational"
  python predict.py --prompt "Solve: x^2 - 5x + 6 = 0" --format json
  python predict.py --prompt "What is the sum of 1+2+...+100?" --temperature 0.3""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Prove that there are infinitely many prime numbers.",
        help="Math problem or question to solve",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/result.{format})",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "json"],
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Re-enable verbose output if requested
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for logger_name in ["transformers", "torch"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    # Set output path
    if args.output is None:
        ext = "json" if args.format == "json" else "txt"
        args.output = f"outputs/result.{ext}"

    output_path = Path(args.output)

    # Load model
    try:
        model, tokenizer = load_model(MODEL_ID, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Run inference
    print(f"\n=== Prompt ===")
    print(args.prompt)
    print(f"\n=== Generating (max {args.max_new_tokens} tokens) ===")

    try:
        output_text = run_inference(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.greedy,
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print(f"\n=== Response ===")
    print(output_text)

    # Save output
    save_output(output_text, str(output_path), args.format, args.prompt)
    print(f"\nOutput saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
