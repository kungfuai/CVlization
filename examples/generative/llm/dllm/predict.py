#!/usr/bin/env python3
"""dLLM - Diffusion Language Models inference.

Supports multiple diffusion language models:
- Qwen3-0.6B-diffusion-bd3lm: Block diffusion (BD3LM), 0.6B params, ~2GB VRAM
- ModernBERT-large-chat: BERT for chat via diffusion (MDLM), 395M params, ~1GB VRAM
- LLaDA-8B-Instruct: Large diffusion LLM (MDLM), 8B params, ~16GB VRAM

Unlike autoregressive models, diffusion LLMs generate text through iterative
denoising - all tokens are refined in parallel over multiple steps.

References:
- dLLM: https://github.com/ZHZisZZ/dllm
- BD3LM: https://arxiv.org/abs/2503.09573
- LLaDA: https://arxiv.org/abs/2502.09992
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default (BEFORE heavy imports)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "torch", "accelerate"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

import torch
import transformers

# Import dllm (installed in container)
import dllm

# Available models
MODELS = {
    "qwen-bd3lm": {
        "id": "dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1",
        "sampler": "bd3lm",
        "params": "0.6B",
        "vram": "~2GB",
    },
    "bert-chat": {
        "id": "dllm-collection/ModernBERT-large-chat-v0.1",
        "sampler": "mdlm",
        "params": "395M",
        "vram": "~1GB",
    },
    "llada": {
        "id": "GSAI-ML/LLaDA-8B-Instruct",
        "sampler": "mdlm",
        "params": "8B",
        "vram": "~16GB",
    },
}

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


class ModelArgs:
    """Arguments for dllm model loading."""
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path


def load_model(model_key: str):
    """Load diffusion model and tokenizer using dllm utilities."""
    model_info = MODELS[model_key]
    model_id = model_info["id"]
    sampler_type = model_info["sampler"]

    print(f"Loading {model_key}: {model_id}")
    print(f"  Parameters: {model_info['params']}, VRAM: {model_info['vram']}")

    model_args = ModelArgs(model_id)
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Create appropriate sampler based on model type
    if sampler_type == "bd3lm":
        sampler = dllm.core.samplers.BD3LMSampler(model=model, tokenizer=tokenizer)
    else:  # mdlm
        sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)

    print("Model loaded successfully")
    return model, tokenizer, sampler, sampler_type


def run_inference(
    sampler,
    tokenizer,
    sampler_type: str,
    prompt: str,
    steps: int = 128,
    max_new_tokens: int = 256,
    block_size: int = 32,
    temperature: float = 0.0,
):
    """Run diffusion inference."""
    # Prepare message
    messages = [[{"role": "user", "content": prompt}]]

    # Tokenize with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    # Create sampler config based on type
    if sampler_type == "bd3lm":
        sampler_config = dllm.core.samplers.BD3LMSamplerConfig(
            steps=steps,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            remasking="low_confidence",
            right_shift_logits=False,
        )
    else:  # mdlm
        sampler_config = dllm.core.samplers.MDLMSamplerConfig(
            steps=steps,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            remasking="low_confidence",
        )

    # Generate
    outputs = sampler.sample(inputs, sampler_config, return_dict=True)

    # Decode output
    sequences = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)
    output_text = sequences[0].strip() if sequences[0].strip() else "<empty>"

    return output_text


def save_output(text: str, output_path: str, format: str = "txt", metadata: dict = None):
    """Save output to file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "text": text,
            **(metadata or {}),
        }
        output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        output_file.write_text(text)

    print(f"Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="dLLM - Diffusion Language Model inference",
        epilog="""Examples:
  # Use default model (qwen-bd3lm, smallest)
  python predict.py --prompt "What is 2+2?"

  # Use BERT-Chat (very fast, small)
  python predict.py --model bert-chat --prompt "Hello!"

  # Use LLaDA (best quality, needs 16GB VRAM)
  python predict.py --model llada --prompt "Write a poem"

Available models:
  qwen-bd3lm  : Qwen3-0.6B with BD3LM (0.6B, ~2GB VRAM) [default]
  bert-chat   : ModernBERT-large-chat (395M, ~1GB VRAM)
  llada       : LLaDA-8B-Instruct (8B, ~16GB VRAM)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default="qwen-bd3lm",
        help="Model to use (default: qwen-bd3lm)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is 2 + 2? Answer briefly.",
        help="Input prompt for the model",
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
        "--steps",
        type=int,
        default=128,
        help="Number of diffusion denoising steps (default: 128)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for diffusion (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        print("Available models:")
        for key, info in MODELS.items():
            print(f"  {key:12} : {info['id']}")
            print(f"               Params: {info['params']}, VRAM: {info['vram']}, Sampler: {info['sampler']}")
        return 0

    # Re-enable verbose output if requested
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for logger_name in ["transformers", "torch", "accelerate"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"

    # Set seed
    transformers.set_seed(args.seed)

    # Resolve output path
    OUT = get_output_dir()
    if args.output is None:
        ext = "json" if args.format == "json" else "txt"
        args.output = f"result.{ext}"
    output_path = resolve_output_path(args.output, OUT)

    # Load model
    try:
        model, tokenizer, sampler, sampler_type = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    print(f"\nPrompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    print(f"Diffusion steps: {args.steps}, block_size: {args.block_size}")
    try:
        output_text = run_inference(
            sampler,
            tokenizer,
            sampler_type,
            args.prompt,
            steps=args.steps,
            max_new_tokens=args.max_tokens,
            block_size=args.block_size,
            temperature=args.temperature,
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
        "model": args.model,
        "model_id": MODELS[args.model]["id"],
        "prompt": args.prompt,
        "steps": args.steps,
        "max_tokens": args.max_tokens,
        "block_size": args.block_size,
        "temperature": args.temperature,
    }
    save_output(output_text, output_path, args.format, metadata)

    return 0


if __name__ == "__main__":
    sys.exit(main())
