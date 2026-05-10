#!/usr/bin/env python3
"""
Miles GRPO for VLM OMR (Optical Music Recognition)

Trains a VLM (Qwen3-VL) with GRPO on sheet music images → MXC2 transcription.
Uses Miles' decoupled architecture: FSDP training + SGLang inference.

Usage:
    python train.py                    # uses config.yaml
    python train.py --config my.yaml
    python train.py --dry-run          # print command without running
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)

import yaml

INSTRUCTION = "Transcribe this sheet music page to MXC2 (compact MusicXML)."


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_model(name: str) -> Path:
    from huggingface_hub import snapshot_download
    print(f"Ensuring model downloaded: {name}")
    return Path(snapshot_download(repo_id=name))


def prepare_dataset(config: dict) -> Path:
    """Prepare dataset as JSONL for Miles.

    Miles expects JSONL with 'prompt' and 'label' fields.
    For VLM: prompt includes image reference, label is MusicXML.
    """
    from datasets import load_dataset

    dc = config.get("dataset", {})
    name = dc.get("name", "zzsi/synthetic-scores")
    cfg = dc.get("config", "level9")
    split = dc.get("split", "train")
    max_samples = dc.get("max_samples", 200)

    print(f"Loading {name} config={cfg} split={split} (max {max_samples})")
    ds = load_dataset(name, cfg, split=f"{split}[:{max_samples}]")

    output_dir = Path(config.get("training", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save images to disk (Miles needs file paths, not PIL objects)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    dataset_path = output_dir / "train_data.jsonl"
    with open(dataset_path, "w") as f:
        for i, example in enumerate(ds):
            # Save image
            img = example["image"]
            img_path = image_dir / f"{i:06d}.png"
            if not img_path.exists():
                img.save(img_path)

            # Build prompt with image reference
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": INSTRUCTION},
                    ],
                }
            ]

            record = {
                "prompt": json.dumps(prompt_messages),
                "label": example.get("musicxml", ""),
                "images": [str(img_path)],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(ds)} examples to {dataset_path}")
    return dataset_path


def build_command(config: dict, dataset_path: Path, model_path: Path) -> list:
    """Build Miles training command for VLM GRPO."""
    tc = config.get("training", {})
    gc = config.get("grpo", {})

    output_dir = Path(tc.get("output_dir", "outputs"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Path to the reward module
    reward_path = str(Path(__file__).parent / "reward.py")

    miles_script = "/root/miles/train.py"
    cmd = [
        sys.executable, miles_script,

        # Model
        "--hf-checkpoint", str(model_path),
        "--ref-load", str(model_path),
        "--save", str(checkpoint_dir),
        "--save-interval", str(tc.get("save_interval", 10)),

        # Dataset
        "--prompt-data", str(dataset_path),
        "--input-key", "prompt",
        "--label-key", "label",
        "--apply-chat-template",
        "--rollout-shuffle",

        # Multimodal — tell Miles to pass images
        "--multimodal-keys", '{"image": "images"}',

        # Reward
        "--rm-type", "custom",
        "--custom-rm-path", reward_path,

        # Training loop
        "--num-rollout", str(tc.get("num_rollout", 20)),
        "--rollout-batch-size", str(tc.get("rollout_batch_size", 2)),
        "--n-samples-per-prompt", str(tc.get("n_samples_per_prompt", 8)),
        "--rollout-max-response-len", str(tc.get("max_response_len", 4096)),
        "--rollout-temperature", str(tc.get("temperature", 0.7)),
        "--global-batch-size", str(tc.get("global_batch_size", 16)),

        # GRPO
        "--advantage-estimator", "grpo",
        "--use-kl-loss",
        "--kl-loss-coef", str(gc.get("kl_coef", 0.5)),
        "--kl-loss-type", gc.get("kl_type", "low_var_kl"),
        "--entropy-coef", str(gc.get("entropy_coef", 0.0)),
        "--eps-clip", str(gc.get("eps_clip", 0.2)),
        "--eps-clip-high", str(gc.get("eps_clip_high", 0.28)),

        # Optimizer
        "--optimizer", "adam",
        "--lr", str(tc.get("learning_rate", 1e-6)),
        "--lr-decay-style", "constant",
        "--weight-decay", str(tc.get("weight_decay", 0.1)),
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",

        # Backend
        "--train-backend", "fsdp",
        "--gradient-checkpointing",
        "--fsdp-state-dict-cpu-offload",

        # GPU
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(tc.get("num_gpus", 1)),
        "--colocate",

        # SGLang inference
        "--rollout-num-gpus-per-engine", "1",
        "--sglang-mem-fraction-static", str(tc.get("sglang_mem_fraction", 0.4)),

        # Memory
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(tc.get("max_tokens_per_gpu", 4096)),
    ]

    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        model_path = Path("/path/to/model")
        dataset_path = Path("outputs/train_data.jsonl")
        cmd = build_command(config, dataset_path, model_path)
        print("Dry run — would execute:")
        print(" ".join(cmd))
        return

    print("=" * 60)
    print("Miles GRPO — VLM OMR")
    print("=" * 60)

    model_name = config.get("model", {}).get("name", "Qwen/Qwen3-VL-8B-Instruct")
    model_path = download_model(model_name)
    print(f"Model: {model_path}")

    dataset_path = prepare_dataset(config)

    cmd = build_command(config, dataset_path, model_path)
    if args.verbose:
        print("\nCommand:")
        print(" ".join(cmd))

    print("\nStarting Miles GRPO training...")
    print("-" * 60)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        subprocess.run(cmd, env=env, check=True)
        print("\nTraining completed!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed: exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("\nMiles not found. Ensure the Miles Docker image is used.")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
