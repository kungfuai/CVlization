#!/usr/bin/env python3
"""
SFT fine-tuning for QED-Nano on FineProofs-SFT.

Trains Qwen3-4B-Instruct (or any compatible model) on curated Olympiad proof
demonstrations using supervised fine-tuning with LoRA.

Dataset: lm-provers/FineProofs-SFT
  Columns: problem, reasoning_content, proof, messages, category, competition, ...

Each example is formatted as a Qwen3 chat turn:
  User:      PROOF_PROMPT_TEMPLATE.format(problem=...)
  Assistant: <think>{reasoning_content}</think>\n\n{proof}
"""

import os
import sys
import logging
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

PROOF_PROMPT_TEMPLATE = "Generate a rigorous proof to the following question:\n\n{problem}"
DEFAULT_MODEL = os.getenv("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
DEFAULT_DATASET = "lm-provers/FineProofs-SFT"


def format_row(row: dict, tokenizer) -> dict:
    """Format a dataset row into a packed chat string for SFT."""
    # Prefer the pre-built messages column if present
    if row.get("messages"):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    else:
        # Construct from components: thinking trace + proof
        thinking = row.get("reasoning_content", "")
        proof = row.get("proof", "") or row.get("solution", "")
        if thinking:
            assistant_content = f"<think>\n{thinking}\n</think>\n\n{proof}"
        else:
            assistant_content = proof
        messages = [
            {"role": "user", "content": PROOF_PROMPT_TEMPLATE.format(problem=row["problem"])},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return {"text": text}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning for QED-Nano proof generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Quick smoke test (100 examples, 30 steps)
  python train_sft.py --max-samples 100 --max-steps 30

  # Full training run
  python train_sft.py --max-steps 2000 --output-dir outputs/sft_full

  # Multi-GPU (run via torchrun or accelerate launch)
  accelerate launch train_sft.py --max-steps 2000

  # Disable LoRA (full fine-tune, needs more VRAM)
  python train_sft.py --no-lora --max-steps 500
""",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset ID")
    parser.add_argument("--output-dir", default="outputs/sft", help="Checkpoint directory")
    parser.add_argument("--max-steps", type=int, default=30, help="Training steps (default: 30 for smoke test)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Max sequence length")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tune (more VRAM)")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    for name in ["transformers", "datasets", "peft", "trl"]:
        logging.getLogger(name).setLevel(log_level)

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    print(f"Model  : {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Steps  : {args.max_steps}")
    print(f"LoRA   : {'disabled' if args.no_lora else f'r={args.lora_r} alpha={args.lora_alpha}'}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.map(
        lambda x: format_row(x, tokenizer),
        remove_columns=ds.column_names,
        num_proc=4,
    )
    print(f"Training examples: {len(ds)}")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if not args.no_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.enable_input_require_grads()  # required for gradient checkpointing + LoRA
        model.print_trainable_parameters()

    training_cfg = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_steps=max(args.max_steps // 3, 10),
        save_total_limit=3,
        dataloader_num_workers=2,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=True,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
        report_to="wandb" if args.wandb_project else "none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_cfg,
        train_dataset=ds,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving checkpoint to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
