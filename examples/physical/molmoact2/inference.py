#!/usr/bin/env python3
"""
MolmoAct2 Vision-Language-Action Inference

Loads a MolmoAct2 checkpoint and runs single-step or multi-step action
prediction from camera images and a natural-language task instruction.

Supports three action modes:
  - continuous  (flow-matching expert, default)
  - discrete    (FAST tokenizer)

And optionally depth-token reasoning (MolmoAct2-Think variants).

Usage:
    # Quick demo with sample images bundled in the checkpoint
    python inference.py

    # Custom images
    python inference.py --images cam0.png cam1.png \
        --task "pick up the red block" \
        --state 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

    # Think variant with depth reasoning
    python inference.py --model allenai/MolmoAct2-Think-LIBERO --enable-depth-reasoning
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str = "cuda"):
    """Load MolmoAct2 model and processor from HuggingFace."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model: {model_id}", flush=True)
    print("This may take a few minutes on first run...", flush=True)

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    print("Model loaded successfully!", flush=True)
    return model, processor


def load_action_tokenizer(tokenizer_id: str = "allenai/MolmoAct2-FAST-Tokenizer"):
    """Load the FAST action tokenizer for discrete action mode."""
    from transformers import AutoProcessor

    print(f"Loading action tokenizer: {tokenizer_id}", flush=True)
    tokenizer = AutoProcessor.from_pretrained(
        tokenizer_id, trust_remote_code=True
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

# Default LIBERO robot state (8-dim absolute joint-pose)
DEFAULT_LIBERO_STATE = np.array([
    -0.05338004603981972,
     0.007029631175100803,
     0.6783280968666077,
     3.1407692432403564,
     0.0017593271331861615,
    -0.08994418382644653,
     0.03878866136074066,
    -0.03878721222281456,
], dtype=np.float32)

DEFAULT_TASK = (
    "put the white mug on the left plate and "
    "put the yellow and white mug on the right plate"
)


def load_sample_images(model_id: str):
    """Download sample images bundled with the checkpoint."""
    from huggingface_hub import hf_hub_download

    names = ["sample_agentview_rgb.png", "sample_wrist_rgb.png"]
    images = []
    for name in names:
        try:
            path = hf_hub_download(model_id, f"assets/{name}")
            images.append(Image.open(path).convert("RGB"))
            print(f"  Loaded sample image: {name}", flush=True)
        except Exception:
            print(f"  Sample image not found in repo: {name}", flush=True)
    return images


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_action(
    model,
    processor,
    images: list,
    task: str,
    state: np.ndarray,
    norm_tag: str = "libero",
    action_mode: str = "continuous",
    enable_depth_reasoning: bool = False,
    enable_adaptive_depth: bool = False,
    depth_cache=None,
    action_tokenizer=None,
    num_steps: int = 10,
):
    """
    Run one step of MolmoAct2 action prediction.

    Returns a dict with keys: actions, depth_cache, depth_bins.
    """
    kwargs = dict(
        processor=processor,
        images=images,
        task=task,
        state=state,
        norm_tag=norm_tag,
        action_mode=action_mode,
        enable_depth_reasoning=enable_depth_reasoning,
        normalize_language=True,
        num_steps=num_steps,
        enable_cuda_graph=True,
    )

    if enable_depth_reasoning:
        kwargs["enable_adaptive_depth"] = enable_adaptive_depth
        kwargs["depth_cache"] = depth_cache

    if action_mode == "discrete" and action_tokenizer is not None:
        kwargs["action_tokenizer"] = action_tokenizer

    with torch.no_grad():
        out = model.predict_action(**kwargs)

    return {
        "actions": out.actions,
        "depth_cache": getattr(out, "depth_cache", None),
        "depth_bins": getattr(out, "depth_bins", None),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_actions(actions) -> str:
    """Pretty-print the predicted action chunk."""
    arr = actions.cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions)
    if arr.ndim == 1:
        arr = arr[None, :]
    lines = []
    for t, row in enumerate(arr):
        vals = "  ".join(f"{v:+.5f}" for v in row)
        lines.append(f"  t={t:3d}: [{vals}]")
    return "\n".join(lines)


def save_results(actions, output_dir: str):
    """Save predicted actions to artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    arr = actions.cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions)
    np.save(os.path.join(output_dir, "actions.npy"), arr)

    metrics = {
        "action_chunk_length": int(arr.shape[0]) if arr.ndim > 1 else 1,
        "action_dim": int(arr.shape[-1]),
        "action_mean": float(arr.mean()),
        "action_std": float(arr.std()),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved actions.npy and metrics.json to {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MolmoAct2 Vision-Language-Action Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Quick demo with sample images from the checkpoint
  python inference.py

  # Custom images and task
  python inference.py --images cam0.png cam1.png \\
      --task "pick up the red block" \\
      --state 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

  # Think variant with depth reasoning
  python inference.py --model allenai/MolmoAct2-Think-LIBERO \\
      --enable-depth-reasoning

  # Discrete action mode
  python inference.py --action-mode discrete
""",
    )

    parser.add_argument(
        "--model", type=str,
        default="allenai/MolmoAct2-LIBERO",
        help="HuggingFace model ID (default: allenai/MolmoAct2-LIBERO)",
    )
    parser.add_argument(
        "--images", nargs="+", type=str, default=None,
        help="Paths to camera images (order must match checkpoint camera order)",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Natural-language task instruction",
    )
    parser.add_argument(
        "--state", nargs="+", type=float, default=None,
        help="Robot state vector (space-separated floats)",
    )
    parser.add_argument(
        "--norm-tag", type=str, default="libero",
        help="Normalization tag matching the checkpoint (default: libero)",
    )
    parser.add_argument(
        "--action-mode", type=str, default="continuous",
        choices=["continuous", "discrete"],
        help="Action decoding mode (default: continuous)",
    )
    parser.add_argument(
        "--enable-depth-reasoning", action="store_true",
        help="Enable depth-token reasoning (Think variants)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=10,
        help="Flow solver iterations for continuous mode (default: 10)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (default: auto — cuda if available, else cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./artifacts",
        help="Directory to save action outputs (default: ./artifacts)",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cpu":
        print("WARNING: Running on CPU will be very slow.", flush=True)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({vram:.1f} GB)", flush=True)

    # Load model
    model, processor = load_model(args.model, device)

    # Load optional FAST tokenizer for discrete mode
    action_tokenizer = None
    if args.action_mode == "discrete":
        action_tokenizer = load_action_tokenizer()

    # Resolve images
    if args.images:
        images = [Image.open(p).convert("RGB") for p in args.images]
        print(f"Loaded {len(images)} image(s) from disk.", flush=True)
    else:
        print("No --images provided; loading sample images from checkpoint...", flush=True)
        images = load_sample_images(args.model)
        if not images:
            print("ERROR: Could not load sample images. Provide --images.", flush=True)
            sys.exit(1)

    # Resolve task
    task = args.task or DEFAULT_TASK
    print(f"Task: {task}", flush=True)

    # Resolve state
    if args.state is not None:
        state = np.array(args.state, dtype=np.float32)
    else:
        state = DEFAULT_LIBERO_STATE
    print(f"State ({len(state)} dims): {state.tolist()}", flush=True)

    # Run inference
    print("\nRunning inference...", flush=True)
    result = predict_action(
        model=model,
        processor=processor,
        images=images,
        task=task,
        state=state,
        norm_tag=args.norm_tag,
        action_mode=args.action_mode,
        enable_depth_reasoning=args.enable_depth_reasoning,
        enable_adaptive_depth=args.enable_depth_reasoning,
        action_tokenizer=action_tokenizer,
        num_steps=args.num_steps,
    )

    actions = result["actions"]
    print(f"\nPredicted actions:\n{format_actions(actions)}", flush=True)

    if result["depth_bins"] is not None:
        db = result["depth_bins"]
        arr = db.cpu().numpy() if hasattr(db, "cpu") else np.asarray(db)
        print(f"\nDepth bins (10x10):\n{arr}", flush=True)

    # Save outputs
    save_results(actions, args.output_dir)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
