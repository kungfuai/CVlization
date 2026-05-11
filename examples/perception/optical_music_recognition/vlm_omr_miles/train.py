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
    """Resolve model path: local directory or HuggingFace repo ID."""
    local = Path(name)
    if local.is_absolute() and local.exists():
        print(f"Using local model: {local}")
        return local
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
    max_samples = dc.get("max_samples")  # None → use all

    slice_spec = f"{split}[:{max_samples}]" if max_samples else split
    print(f"Loading {name} config={cfg} split={slice_spec}")
    ds = load_dataset(name, cfg, split=slice_spec)
    print(f"  Loaded {len(ds)} samples")

    output_dir = Path(config.get("training", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save images to disk (Miles needs file paths, not PIL objects)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    dataset_path = output_dir / "train_data.jsonl"
    with open(dataset_path, "w") as f:
        for i, example in enumerate(ds):
            img = example["image"]
            img_path = image_dir / f"{i:06d}.png"
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


QWEN35_9B_MODEL_ARGS = [
    "--spec", "miles_plugins.models.qwen3_5", "get_qwen3_5_spec",
    "--disable-bias-linear",
    "--qk-layernorm",
    "--group-query-attention",
    "--num-attention-heads", "16",
    "--num-query-groups", "4",
    "--kv-channels", "256",
    "--num-layers", "32",
    "--hidden-size", "4096",
    "--ffn-hidden-size", "12288",
    "--normalization", "RMSNorm",
    "--apply-layernorm-1p",
    "--position-embedding-type", "rope",
    "--norm-epsilon", "1e-6",
    "--rotary-percent", "0.25",
    "--swiglu",
    "--untie-embeddings-and-output-weights",
    "--vocab-size", "248320",
    "--rotary-base", "10000000",
    "--attention-output-gate",  # qwen3.5 specific
]


def build_command(config: dict, dataset_path: Path, model_path: Path) -> list:
    """Build Miles training command for VLM GRPO (Megatron backend)."""
    tc = config.get("training", {})
    gc = config.get("grpo", {})
    num_gpus = tc.get("num_gpus", 2)

    output_dir = Path(tc.get("output_dir", "outputs"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reward_path = str(Path(__file__).parent / "reward.py")

    cmd = [
        sys.executable, "/root/miles/train.py",

        "--hf-checkpoint", str(model_path),
        "--ref-load", str(model_path),
        "--load", str(model_path),
        "--save", str(checkpoint_dir),
        "--save-interval", str(tc.get("save_interval", 10)),

        "--prompt-data", str(dataset_path),
        "--input-key", "prompt",
        "--label-key", "label",
        "--apply-chat-template",
        "--rollout-shuffle",

        "--multimodal-keys", '{"image": "images"}',

        "--rm-type", "custom",
        "--custom-rm-path", reward_path,

        "--num-rollout", str(tc.get("num_rollout", 20)),
        "--rollout-batch-size", str(tc.get("rollout_batch_size", 2)),
        "--n-samples-per-prompt", str(tc.get("n_samples_per_prompt", 8)),
        "--rollout-max-response-len", str(tc.get("max_response_len", 4096)),
        "--rollout-temperature", str(tc.get("temperature", 0.7)),
        "--global-batch-size", str(tc.get("global_batch_size", 16)),

        "--advantage-estimator", "grpo",
        "--use-kl-loss",
        "--kl-loss-coef", str(gc.get("kl_coef", 0.5)),
        "--kl-loss-type", gc.get("kl_type", "low_var_kl"),
        "--entropy-coef", str(gc.get("entropy_coef", 0.0)),
        "--eps-clip", str(gc.get("eps_clip", 0.2)),
        "--eps-clip-high", str(gc.get("eps_clip_high", 0.28)),

        "--optimizer", "adam",
        "--lr", str(tc.get("learning_rate", 1e-6)),
        "--lr-decay-style", "constant",
        "--weight-decay", str(tc.get("weight_decay", 0.1)),
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",

        # Megatron backend with in-memory bridge conversion
        "--train-backend", "megatron",
        "--tensor-model-parallel-size", str(num_gpus),
        "--sequence-parallel",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--megatron-to-hf-mode", "bridge",

        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(num_gpus),
        "--colocate",

        "--rollout-num-gpus-per-engine", str(num_gpus),
        "--sglang-mem-fraction-static", str(tc.get("sglang_mem_fraction", 0.4)),

        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(tc.get("max_tokens_per_gpu", 4096)),
    ] + QWEN35_9B_MODEL_ARGS

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
