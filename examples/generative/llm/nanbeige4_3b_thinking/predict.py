#!/usr/bin/env python3
"""Run Nanbeige4-3B-Thinking-2510 chat/tool inference."""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress noisy defaults; re-enable with --verbose
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("MLIR_ENABLE_DUMP", "0")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
logging.basicConfig(level=logging.ERROR)
for _name in ["transformers", "torch", "triton"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
)

# CVL dual-mode support
try:  # pragma: no cover - optional dependency
    from cvlization.paths import (  # type: ignore
        get_input_dir,
        get_output_dir,
        resolve_input_path,
        resolve_output_path,
    )

    CVL_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when cvlization is absent
    CVL_AVAILABLE = False

    def get_input_dir() -> str:
        return os.getcwd()

    def get_output_dir() -> str:
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def resolve_input_path(path: str, base_dir: str) -> str:
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    def resolve_output_path(path: str, base_dir: str) -> str:
        return path if os.path.isabs(path) else os.path.join(base_dir, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nanbeige4-3B-Thinking chat and tool-call generation"
    )
    parser.add_argument(
        "--model-id",
        default="Nanbeige/Nanbeige4-3B-Thinking-2510",
        help="Hugging Face model id or local path",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="User message text. Use --prompt-file to load from file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file containing the user prompt.",
    )
    parser.add_argument(
        "--system-message",
        default=(
            "You are Nanbeige4-3B-Thinking, a concise reasoning assistant. "
            "Do your internal reasoning inside <think>...</think>, close the think block, "
            "and then provide the final answer in plain text after the think block."
        ),
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--tools-json",
        type=Path,
        help="Optional path to a JSON file containing tool/function definitions.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0 for greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling cutoff.",
    )
    parser.add_argument(
        "--truncate-input-tokens",
        type=int,
        default=None,
        help="If set, keep only the last N tokens of the prompt to avoid OOM.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/nanbeige4_3b_output.txt"),
        help="Where to save generated text.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Force a device or auto-detect.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code if you want to match upstream README; not required for this model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs and framework debug output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def enable_verbose():
    logging.basicConfig(level=logging.INFO, force=True)
    for _name in ["transformers", "torch", "triton"]:
        logging.getLogger(_name).setLevel(logging.INFO)
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    os.environ["MLIR_ENABLE_DUMP"] = "1"


def detect_device(preference: str) -> Tuple[str, torch.dtype]:
    if preference != "auto":
        device = preference
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda":
        major_cc = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def load_tools(path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("tools-json must be a JSON array of tool definitions")
    return data


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text().strip()
    if args.prompt:
        return args.prompt.strip()
    raise ValueError("Provide either --prompt or --prompt-file")


def apply_truncation(input_ids: torch.Tensor, limit: Optional[int]) -> torch.Tensor:
    if limit is None or input_ids.shape[1] <= limit:
        return input_ids
    return input_ids[:, -limit:]


def build_inputs(
    tokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    truncate_tokens: Optional[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(
        chat_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = apply_truncation(encoded.input_ids, truncate_tokens)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def run_generation(
    model,
    tokenizer,
    model_inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    do_sample = temperature > 0
    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=eos,
            pad_token_id=pad,
        )
    new_tokens = generated[0, model_inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def strip_think_blocks(text: str) -> Tuple[str, Optional[str]]:
    """Remove <think>...</think> from text, returning (cleaned, reasoning)."""
    start = text.find("<think>")
    end = text.find("</think>")
    if start != -1:
        if end != -1 and end > start:
            reasoning = text[start + len("<think>") : end].strip()
            cleaned = (text[:start] + text[end + len("</think>") :]).strip()
            return cleaned, reasoning
        reasoning = text[start + len("<think>") :].strip()
        return reasoning, reasoning
    return text.strip(), None


def main():
    args = parse_args()
    if args.verbose:
        enable_verbose()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device_str, dtype = detect_device(args.device)
    device = torch.device(device_str)
    logging.info("Using device %s with dtype %s", device, dtype)

    prompt_text = load_prompt(args)
    tools = load_tools(args.tools_json)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=args.trust_remote_code,
    )
    if device.type != "cuda":
        model = model.to(device)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": args.system_message},
        {"role": "user", "content": prompt_text},
    ]

    model_inputs = build_inputs(
        tokenizer=tokenizer,
        messages=messages,
        tools=tools,
        truncate_tokens=args.truncate_input_tokens,
        device=device,
    )

    output_text = run_generation(
        model=model,
        tokenizer=tokenizer,
        model_inputs=model_inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    final_text, reasoning = strip_think_blocks(output_text)

    output_dir = Path(resolve_output_path(str(args.output_file), get_output_dir())).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(resolve_output_path(str(args.output_file), get_output_dir()))
    to_save = final_text
    if reasoning:
        to_save = f"{final_text}\n\n[reasoning]\n{reasoning}"
    output_path.write_text(to_save.strip() + "\n")

    print("\n=== Prompt ===")
    print(prompt_text)
    if tools:
        print(f"\n=== Tools ({len(tools)}) ===")
        print(json.dumps(tools, ensure_ascii=False, indent=2))
    if reasoning:
        print("\n=== Reasoning (think) ===")
        print(reasoning)
    print("\n=== Output ===")
    print(final_text.strip())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
