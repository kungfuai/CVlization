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
from grpo_rewards import combined_reward


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
    sft_adapter = model_config.get("sft_adapter")

    # Step 1: Load base model
    print(f"Loading base model: {model_config['name']} ...")
    model, processor = FastVisionModel.from_pretrained(
        model_config["name"],
        load_in_4bit=model_config.get("load_in_4bit", True),
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
    )

    # Step 2: Create LoRA structure for GRPO
    # finetune_vision_layers=False is REQUIRED — vLLM (used by GRPO for
    # generation) does not support LoRA on vision/encoder layers.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config.get("dropout", 0),
        bias="none",
        random_state=training_config.get("seed", 3407),
        use_rslora=False,
        loftq_config=None,
    )

    # Step 3: Load SFT adapter weights (if provided)
    # Uses PeftModel.from_pretrained which properly loads saved LoRA weights
    # into the existing model structure. Vision LoRA weights from SFT are
    # loaded but frozen during GRPO (language layers only are trainable).
    if sft_adapter:
        from peft import PeftModel
        print(f"Loading SFT adapter: {sft_adapter} ...")
        model = PeftModel.from_pretrained(model, sft_adapter)
        print(f"  SFT adapter loaded via PeftModel.from_pretrained")

    # ── Step-0 verification: check SFT quality before GRPO ──────────────
    print("\n=== Step-0 verification: testing SFT quality ===")
    FastVisionModel.for_inference(model)
    _verify_ds = load_dataset(
        dataset_config["repo"], dataset_config["config"],
        split=dataset_config.get("split", "train"),
    )
    _sample = _verify_ds[0]
    from train import prepare_inference_inputs
    _inputs = prepare_inference_inputs(processor, _sample["image"], INSTRUCTION_MXC2).to("cuda")
    with torch.no_grad():
        _out = model.generate(**_inputs, max_new_tokens=2048, use_cache=True, do_sample=False)
    _pred = processor.decode(_out[0][_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    _ref_xml = strip_musicxml_header(_sample["musicxml"])
    try:
        _ref = xml_to_mxc2(_ref_xml, drop_beams=dataset_config.get("drop_beams", True))
    except Exception:
        _ref = _ref_xml
    from grpo_rewards import _extract_pitches, _lcs_length
    _pred_p = _extract_pitches(_pred)
    _ref_p = _extract_pitches(_ref)
    if _pred_p and _ref_p:
        _lcs = _lcs_length(_pred_p, _ref_p)
        _sim = _lcs / min(len(_ref_p), 80)
        print(f"  Pred pitches: {len(_pred_p)}, Ref pitches: {len(_ref_p)}")
        print(f"  LCS pitch similarity: {_sim:.1%}")
        print(f"  Pred first 10: {_pred_p[:10]}")
        print(f"  Ref first 10:  {_ref_p[:10]}")
        if _sim < 0.1:
            print(f"  WARNING: SFT quality is very low ({_sim:.1%}). "
                  f"The adapter may not be loaded correctly!")
    else:
        print(f"  WARNING: No pitches extracted. Pred={len(_pred_p)}, Ref={len(_ref_p)}")
        print(f"  Pred first 200 chars: {_pred[:200]}")
    print("=== End step-0 verification ===\n")
    del _verify_ds, _sample, _inputs, _out, _pred, _ref

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
