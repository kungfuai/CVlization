#!/usr/bin/env python3
"""
Miles GRPO Training for Qwen3-0.6B

This script wraps the Miles RL training framework to enable single-GPU
GRPO training with Qwen3-0.6B. It uses FSDP backend with colocate mode
for training and inference on the same GPU.

Miles GitHub: https://github.com/radixark/miles
"""
import os
import sys
import logging
import warnings
import argparse
import subprocess
from pathlib import Path

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_model(model_name: str) -> Path:
    """Download model from HuggingFace if not cached."""
    from huggingface_hub import snapshot_download

    print(f"Ensuring model is downloaded: {model_name}")
    local_path = snapshot_download(repo_id=model_name)
    return Path(local_path)


def download_dataset(dataset_name: str) -> Path:
    """Download dataset from HuggingFace if not cached."""
    from datasets import load_dataset

    print(f"Ensuring dataset is downloaded: {dataset_name}")
    # Load to trigger download and cache
    ds = load_dataset(dataset_name, split="train[:100]")

    # Return cache path
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    return cache_dir


def prepare_dataset(config: dict) -> Path:
    """Prepare dataset in JSONL format for Miles."""
    from datasets import load_dataset
    import json

    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name", "zhuzilin/dapo-math-17k")
    max_samples = dataset_config.get("max_samples", 100)

    print(f"Loading dataset: {dataset_name} (max {max_samples} samples)")

    ds = load_dataset(dataset_name, split=f"train[:{max_samples}]")

    # Prepare output path
    output_dir = Path(config.get("training", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "train_data.jsonl"

    # Write JSONL
    with open(dataset_path, "w") as f:
        for example in ds:
            # Adapt to Miles format: needs 'prompt' and 'label' fields
            record = {
                "prompt": example.get("prompt", example.get("question", "")),
                "label": example.get("label", example.get("answer", "")),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(ds)} examples to {dataset_path}")
    return dataset_path


def build_miles_command(config: dict, dataset_path: Path, model_path: Path) -> list:
    """Build the Miles training command."""
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    grpo_config = config.get("grpo", {})

    output_dir = Path(training_config.get("output_dir", "outputs"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build command arguments
    # Miles train.py is at /root/miles/train.py in the container
    miles_train_script = "/root/miles/train.py"
    cmd = [
        sys.executable, miles_train_script,

        # Model checkpoints
        "--hf-checkpoint", str(model_path),
        "--ref-load", str(model_path),  # Reference model for KL divergence
        "--save", str(checkpoint_dir),
        "--save-interval", str(training_config.get("save_interval", 50)),

        # Dataset
        "--prompt-data", str(dataset_path),
        "--input-key", "prompt",
        "--label-key", "label",
        "--apply-chat-template",
        "--rollout-shuffle",

        # Reward model
        "--rm-type", training_config.get("rm_type", "math"),

        # Training loop
        "--num-rollout", str(training_config.get("num_rollout", 10)),
        "--rollout-batch-size", str(training_config.get("rollout_batch_size", 4)),
        "--n-samples-per-prompt", str(training_config.get("n_samples_per_prompt", 4)),
        "--rollout-max-response-len", str(training_config.get("max_response_len", 512)),
        "--rollout-temperature", str(training_config.get("temperature", 0.8)),
        "--global-batch-size", str(training_config.get("global_batch_size", 16)),

        # GRPO algorithm
        "--advantage-estimator", "grpo",
        "--use-kl-loss",
        "--kl-loss-coef", str(grpo_config.get("kl_coef", 0.0)),
        "--kl-loss-type", grpo_config.get("kl_type", "low_var_kl"),
        "--entropy-coef", str(grpo_config.get("entropy_coef", 0.0)),
        "--eps-clip", str(grpo_config.get("eps_clip", 0.2)),
        "--eps-clip-high", str(grpo_config.get("eps_clip_high", 0.28)),

        # Optimizer
        "--optimizer", "adam",
        "--lr", str(training_config.get("learning_rate", 1e-6)),
        "--lr-decay-style", "constant",
        "--weight-decay", str(training_config.get("weight_decay", 0.1)),
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",

        # Backend: FSDP for simpler single-GPU setup
        "--train-backend", "fsdp",
        "--attn-implementation", "flash_attention_2",
        "--gradient-checkpointing",
        "--fsdp-state-dict-cpu-offload",  # Offload state dict to CPU during save to avoid OOM

        # GPU allocation
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(training_config.get("num_gpus", 1)),
        "--colocate",  # Train + inference on same GPUs

        # SGLang inference - reduced memory fraction to leave room for FSDP save
        "--rollout-num-gpus-per-engine", "1",
        "--sglang-mem-fraction-static", str(training_config.get("sglang_mem_fraction", 0.35)),

        # Memory optimization for small GPU
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(training_config.get("max_tokens_per_gpu", 1024)),
    ]

    return cmd


def run_training(config: dict, verbose: bool = False):
    """Run Miles GRPO training."""
    print("=" * 60)
    print("Miles GRPO Training - Qwen3-0.6B")
    print("=" * 60)

    # Download/cache model
    model_name = config.get("model", {}).get("name", "Qwen/Qwen3-0.6B")
    model_path = download_model(model_name)
    print(f"Model path: {model_path}")

    # Prepare dataset
    dataset_path = prepare_dataset(config)

    # Build command
    cmd = build_miles_command(config, dataset_path, model_path)

    if verbose:
        print("\nMiles command:")
        print(" ".join(cmd))

    print("\nStarting Miles training...")
    print("-" * 60)

    # Run training
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
        )
        print("\nTraining completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("\nError: Miles not found. Ensure Miles is installed:")
        print("  pip install -e /path/to/miles")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Miles GRPO Training for Qwen3-0.6B"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    # Load config
    config = load_config(args.config)

    if args.dry_run:
        model_name = config.get("model", {}).get("name", "Qwen/Qwen3-0.6B")
        model_path = Path("/path/to/model")  # Placeholder
        dataset_path = Path("outputs/train_data.jsonl")
        cmd = build_miles_command(config, dataset_path, model_path)
        print("Dry run - would execute:")
        print(" ".join(cmd))
        return 0

    return run_training(config, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
