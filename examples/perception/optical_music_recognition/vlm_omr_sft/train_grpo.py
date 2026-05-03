#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for OMR.

Starts from an SFT-trained adapter and uses RL rewards to improve:
1. Part structure accuracy (correct number of parts)
2. Pitch similarity (matching reference pitches)
3. Output length control (penalize overgeneration)

Usage:
    python train_grpo.py --config config_grpo.yaml
    python train_grpo.py --config config_grpo.yaml --max-samples 100  # smoke test
"""

import argparse
import os
import re
import sys
import yaml
import torch

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from unsloth import FastVisionModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(__file__))
from train import strip_musicxml_header, INSTRUCTION_MXC2
from mxc2 import xml_to_mxc2

# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _extract_parts(text):
    """Extract part IDs from MXC2 text."""
    parts = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("P") and not line.startswith("print"):
            parts.append(line.split()[0])
    return parts


def _extract_pitches(text):
    """Extract pitch sequence from MXC2 text."""
    pitches = []
    for line in text.split("\n"):
        m = re.match(r"[+]?N\s+(\S+)", line.strip())
        if m:
            pitches.append(m.group(1))
    return pitches


def _count_notes(text):
    """Count note events in MXC2 text."""
    return sum(1 for line in text.split("\n")
               if re.match(r"\s*[+]?N\s", line))


def _lcs_length(a, b):
    """Longest common subsequence length (O(n*m) but capped for speed)."""
    # Cap to prevent slow reward computation
    a = a[:80]
    b = b[:80]
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Space-optimized LCS
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


def combined_reward(completions, ref_mxc2, **kwargs):
    """Single combined reward to reduce reward hacking.

    Components (weighted sum, not separate optimizable targets):
    - Pitch LCS similarity: 60% weight (primary metric)
    - Part count accuracy: 20% weight
    - Length control: 20% weight
    """
    scores = []
    for pred, ref in zip(completions, ref_mxc2):
        pred_pitches = _extract_pitches(pred)
        ref_pitches = _extract_pitches(ref)
        pred_parts = _extract_parts(pred)
        ref_parts = _extract_parts(ref)
        pred_notes = _count_notes(pred)
        ref_notes = _count_notes(ref)

        # ── Pitch LCS (60% weight) ────────────────────────────────────
        if pred_pitches and ref_pitches:
            lcs = _lcs_length(pred_pitches, ref_pitches)
            # Normalize by reference length (recall-oriented)
            pitch_score = lcs / min(len(ref_pitches), 80)
        else:
            pitch_score = 0.0

        # ── Part count (20% weight) ───────────────────────────────────
        if ref_parts:
            part_score = 1.0 if len(pred_parts) == len(ref_parts) else 0.0
        else:
            part_score = 0.5

        # ── Length control (20% weight) ───────────────────────────────
        if ref_notes > 0:
            coverage = pred_notes / ref_notes
            if 0.7 <= coverage <= 1.3:
                length_score = 1.0
            elif coverage > 2.0 or coverage < 0.3:
                length_score = 0.0
            else:
                length_score = 0.5
        else:
            length_score = 0.5

        # Combined: scale to [-1, +3] range for GRPO
        score = (pitch_score * 0.6 + part_score * 0.2 + length_score * 0.2) * 4.0 - 1.0
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_grpo.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    lora_config = config["lora"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    wandb_config = config.get("wandb", {})

    # ── WandB ──────────────────────────────────────────────────────────────
    use_wandb = bool(os.environ.get("WANDB_API_KEY") and wandb_config.get("project"))
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_config["project"],
            name=wandb_config.get("run_name"),
            config=config,
        )
        print(f"WandB: {wandb.run.url}")

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading model: {model_config['name']} ...")
    model, processor = FastVisionModel.from_pretrained(
        model_config["name"],
        load_in_4bit=model_config.get("load_in_4bit", True),
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_config.get("finetune_vision_layers", False),
        finetune_language_layers=lora_config.get("finetune_language_layers", True),
        finetune_attention_modules=lora_config.get("finetune_attention_modules", True),
        finetune_mlp_modules=lora_config.get("finetune_mlp_modules", True),
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config.get("dropout", 0),
        bias="none",
        random_state=training_config.get("seed", 3407),
        use_rslora=False,
        loftq_config=None,
    )

    # Load SFT adapter weights if specified
    sft_adapter = model_config.get("sft_adapter")
    if sft_adapter:
        print(f"Loading SFT adapter: {sft_adapter} ...")
        import safetensors.torch
        from peft import set_peft_model_state_dict
        state = safetensors.torch.load_file(f"{sft_adapter}/adapter_model.safetensors")
        set_peft_model_state_dict(model, state)
        print(f"  Loaded {len(state)} tensors")

    # ── Dataset ────────────────────────────────────────────────────────────
    repo = dataset_config["repo"]
    cfg = dataset_config["config"]
    split = dataset_config.get("split", "train")
    print(f"Loading dataset: {repo} config={cfg} split={split} ...")
    dataset = load_dataset(repo, cfg, split=split)

    if "corpora" in dataset_config:
        corpora = dataset_config["corpora"]
        dataset = dataset.filter(lambda r: r.get("corpus") in corpora)
        print(f"  Filtered to corpora {corpora}: {len(dataset)} samples")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")

    drop_beams = dataset_config.get("drop_beams", True)

    # Convert to GRPO format
    def make_conversation(example):
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": INSTRUCTION_MXC2},
                ],
            },
        ]

        # Prepare reference MXC2 for reward functions
        ref_xml = strip_musicxml_header(example["musicxml"])
        try:
            ref = xml_to_mxc2(ref_xml, drop_beams=drop_beams)
        except Exception:
            ref = ref_xml

        return {
            "prompt": prompt,
            "image": example["image"],
            "ref_mxc2": ref,
        }

    print("Converting to GRPO format ...")
    train_dataset = make_conversation(dataset[0])  # test one
    train_dataset = dataset.map(make_conversation)

    # Apply chat template
    train_dataset = train_dataset.map(
        lambda example: {
            "prompt": processor.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
        }
    )

    print(f"Dataset: {len(train_dataset)} samples")

    # ── GRPO Training ──────────────────────────────────────────────────────
    training_args = GRPOConfig(
        output_dir=training_config.get("output_dir", "outputs/grpo"),
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        num_generations=training_config.get("num_generations", 4),
        max_prompt_length=training_config.get("max_prompt_length", 512),
        max_completion_length=training_config.get("max_completion_length", 4096),
        max_steps=training_config.get("max_steps", 200),
        save_steps=training_config.get("save_steps", 50),
        logging_steps=training_config.get("logging_steps", 1),
        warmup_ratio=training_config.get("warmup_ratio", 0.05),
        max_grad_norm=training_config.get("max_grad_norm", 0.1),
        weight_decay=training_config.get("weight_decay", 0.001),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        optim=training_config.get("optim", "adamw_torch_fused"),
        beta=training_config.get("beta", 0.1),
        loss_type=training_config.get("loss_type", "dr_grpo"),
        seed=training_config.get("seed", 3407),
        bf16=True,
        report_to="wandb" if use_wandb else "none",
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    max_mem = round(gpu_stats.total_memory / 1024 ** 3, 3)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    print(f"\nGPU: {gpu_stats.name}  |  {max_mem} GB total  |  {start_mem} GB reserved")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        reward_funcs=[combined_reward],
        train_dataset=train_dataset,
    )

    print("\nStarting GRPO training ...")
    trainer.train()

    used_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    print(f"\n{'='*70}")
    print(f"GRPO TRAINING COMPLETE")
    print(f"  Memory: {used_mem} GB peak ({round(used_mem / max_mem * 100, 1)}%)")
    print(f"{'='*70}")

    # Save
    output_dir = training_config.get("output_dir", "outputs/grpo")
    final_dir = f"{output_dir}/final_model"
    print(f"\nSaving to {final_dir} ...")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
