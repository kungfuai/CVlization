#!/usr/bin/env python3
"""RNJ-1-Instruct: 8B parameter code and STEM model by Essential AI.

A dense language model optimized for code generation, tool calling, and STEM tasks.
- 8.3B parameters, 32K context window
- Strong on SWE-bench, HumanEval+, MBPP+, math benchmarks
- Apache 2.0 license

Reference: https://huggingface.co/EssentialAI/rnj-1-instruct
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default (BEFORE heavy imports)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging - only show errors by default
logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "torch", "accelerate"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "EssentialAI/rnj-1-instruct"

# CVL dual-mode execution support
try:
    from cvlization.paths import (
        get_input_dir,
        get_output_dir,
        resolve_input_path,
        resolve_output_path,
    )
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_input_dir():
        return os.getcwd()

    def get_output_dir():
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)

    def resolve_output_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)


def detect_device():
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
    """Load RNJ-1-Instruct model and tokenizer."""
    if device is None:
        device, dtype = detect_device()
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)

    print("Model loaded successfully")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
):
    """Run inference on RNJ-1-Instruct."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        if temperature > 0:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Decode only the generated tokens
    generated_ids = output_ids[0][input_ids.shape[-1] :]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return output_text


def save_output(text: str, output_path: str, format: str = "txt", metadata: dict = None):
    """Save output to file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "text": text,
            "model": MODEL_ID,
            **(metadata or {}),
        }
        output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        output_file.write_text(text)

    print(f"Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RNJ-1-Instruct: Code and STEM inference",
        epilog="""Examples:
  python predict.py --prompt "Write a Python function to check if a number is prime"
  python predict.py --prompt "Explain quicksort" --temperature 0.5
  python predict.py --prompt "def fibonacci(n):" --system "Complete the code" --format json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a Python function that checks if a string is a palindrome.",
        help="Input prompt for the model",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt (default: 'You are a helpful assistant.')",
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
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature, 0 for greedy (default: 0.2)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
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
        for logger_name in ["transformers", "torch", "accelerate"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"

    # Resolve output path
    OUT = get_output_dir()
    if args.output is None:
        ext = "json" if args.format == "json" else "txt"
        args.output = f"result.{ext}"
    output_path = resolve_output_path(args.output, OUT)

    # Load model
    try:
        model, tokenizer = load_model(MODEL_ID, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    print(f"Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    try:
        output_text = run_inference(
            model,
            tokenizer,
            args.prompt,
            system_prompt=args.system,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Display output
    print("\n=== Output ===")
    print(output_text)

    # Save output
    metadata = {
        "prompt": args.prompt,
        "system_prompt": args.system,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    save_output(output_text, output_path, args.format, metadata)

    return 0


if __name__ == "__main__":
    sys.exit(main())
