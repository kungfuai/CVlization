#!/usr/bin/env python3
"""
QED-Nano: 4B-parameter model for Olympiad-level mathematical proof generation.

Generates natural-language proofs for competition math problems using a
Qwen3-4B-based model trained with SFT + GRPO reinforcement learning.

Model: lm-provers/QED-Nano (Apache 2.0)
HF:    https://huggingface.co/lm-provers/QED-Nano
"""

import os
import sys

# Must be set before importing vllm/torch (prevents CUDA fork issues)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TORCH_DISTRIBUTED_DESTROY_PROCESS_GROUP_ON_EXIT", "0")

import argparse
import json
from pathlib import Path

try:
    from cvlization.paths import resolve_output_path
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def resolve_output_path(path: str) -> str:
        if path.startswith("/"):
            return path
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        return str(out / path)


MODEL_ID = os.getenv("MODEL_ID", "lm-provers/QED-Nano")

# Prompt template used during QED-Nano training (from conf/qed_nano_rl.yaml)
PROOF_PROMPT_TEMPLATE = "Generate a rigorous proof to the following question:\n\n{problem}"

PROBLEMS = {
    "amgm": (
        "Let a, b, c be positive real numbers. "
        "Prove that (a + b)(b + c)(c + a) ≥ 8abc."
    ),
    "imo1988p6": (
        "Let a and b be positive integers such that ab + 1 divides a² + b². "
        "Prove that (a² + b²) / (ab + 1) is the square of an integer. "
        "(IMO 1988, Problem 6)"
    ),
    "imo2000p2": (
        "Let a, b, c be positive real numbers such that abc = 1. "
        "Prove that (a − 1 + 1/b)(b − 1 + 1/c)(c − 1 + 1/a) ≤ 1. "
        "(IMO 2000, Problem 2)"
    ),
}

SAMPLE_PROBLEM = PROBLEMS["amgm"]


def configure_flash_attn() -> None:
    """Use FA2 on consumer Blackwell GPUs (SM120+) where FA3 is unsupported."""
    import torch
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    sm = major * 10 + minor
    if sm > 100 and "VLLM_FLASH_ATTN_VERSION" not in os.environ:
        os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
        print(f"Auto-set VLLM_FLASH_ATTN_VERSION=2 for SM{sm}")


def build_prompt(model_id: str, problem: str) -> str:
    """Apply the Qwen3 chat template to format the proof request."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    messages = [
        {"role": "user", "content": PROOF_PROMPT_TEMPLATE.format(problem=problem)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def split_thinking(text: str) -> tuple[str, str]:
    """Split a Qwen3 thinking response into (chain_of_thought, proof).

    The model outputs <think>...</think> followed by the proof text.
    """
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        thinking = before.replace("<think>", "").strip()
        proof = after.strip()
    else:
        thinking = ""
        proof = text.strip()
    return thinking, proof


def run_inference(
    problem: str,
    model_id: str = MODEL_ID,
    max_tokens: int = 8192,
    temperature: float = 0.8,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
) -> dict:
    from vllm import LLM, SamplingParams

    prompt = build_prompt(model_id, problem)

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(prompt, sampling_params=sampling)
    raw = outputs[0].outputs[0].text
    thinking, proof = split_thinking(raw)
    return {"problem": problem, "thinking": thinking, "proof": proof}


def save_output(result: dict, output_path: Path, fmt: str, show_thinking: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parts = []
        if show_thinking and result["thinking"]:
            parts.append("=== Chain of Thought ===")
            parts.append(result["thinking"])
            parts.append("")
        parts.append("=== Proof ===")
        parts.append(result["proof"])
        output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Output saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="QED-Nano: Olympiad-level mathematical proof generation (4B model).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python predict.py
  python predict.py --problem "Prove that sqrt(2) is irrational."
  python predict.py --problem "..." --show-thinking --format json
  python predict.py --problem "..." --max-tokens 16384
""",
    )
    parser.add_argument(
        "--problem",
        default=SAMPLE_PROBLEM,
        help="Mathematical problem statement to prove.",
    )
    parser.add_argument(
        "--preset",
        choices=list(PROBLEMS.keys()),
        help=f"Use a named preset problem ({', '.join(PROBLEMS.keys())}).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max new tokens to generate (default: 8192).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8, matches training config).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Max context length for vLLM (default: 16384).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory allocated to vLLM (default: 0.9).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile (useful on unusual GPU architectures).",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Include the chain-of-thought trace in the output file.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "json"],
        default="txt",
        help="Output format (default: txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Default: outputs/proof.{format}.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from transformers and vLLM.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.verbose:
        import logging
        logging.basicConfig(level=logging.ERROR)
        for name in ["transformers", "vllm", "torch"]:
            logging.getLogger(name).setLevel(logging.ERROR)
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    if args.preset:
        args.problem = PROBLEMS[args.preset]

    configure_flash_attn()

    ext = "json" if args.format == "json" else "txt"
    if args.output is None:
        args.output = Path(resolve_output_path(f"proof.{ext}"))

    print(f"Model  : {args.model}")
    print(f"Problem: {args.problem[:120]}{'...' if len(args.problem) > 120 else ''}")
    print()

    result = run_inference(
        problem=args.problem,
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    print("=== Proof ===")
    print(result["proof"])
    print()

    save_output(result, args.output, args.format, args.show_thinking)
    return 0


if __name__ == "__main__":
    sys.exit(main())
