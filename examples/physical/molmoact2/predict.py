#!/usr/bin/env python3
"""
MolmoAct2 Vision-Language-Action Inference

Loads a MolmoAct2 checkpoint and predicts robot actions from camera images
and a natural-language task instruction.

Three action modes:
  - continuous   (flow-matching expert, default)
  - discrete     (FAST tokenizer)

Optional depth-token reasoning (MolmoAct2-Think variants).

Usage:
    # Default: uses bundled sample images from zzsi/cvl
    python predict.py

    # Custom images
    python predict.py --images cam0.png cam1.png \\
        --task "pick up the red block" \\
        --state 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

    # Think variant with depth reasoning
    python predict.py --model allenai/MolmoAct2-Think-LIBERO \\
        --enable-depth-reasoning

    # Discrete action mode
    python predict.py --action-mode discrete
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

# CVL dual-mode path support
try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except ImportError:
    def resolve_input_path(path):
        return path

    def resolve_output_path(path=None, default_filename="result.txt"):
        if path is None:
            path = default_filename
        return str(Path("./outputs") / path) if not path.startswith("/") else path


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

HF_DATA_REPO = "zzsi/cvl"
HF_DATA_PREFIX = "molmoact2"
SAMPLE_FILES = ["sample_agentview_rgb.png", "sample_wrist_rgb.png"]


def ensure_sample_data():
    """Download sample images from zzsi/cvl if not already cached."""
    cache_root = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "cvl_data" / "molmoact2"

    marker = cache_root / ".downloaded"
    if marker.exists():
        return str(cache_root)

    cache_root.mkdir(parents=True, exist_ok=True)
    print("Downloading sample images from zzsi/cvl...", flush=True)

    for name in SAMPLE_FILES:
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=f"{HF_DATA_PREFIX}/{name}",
            repo_type="dataset",
        )
        local_target = cache_root / name
        shutil.copy2(downloaded, local_target)
        print(f"  Cached: {local_target}", flush=True)

    marker.touch()
    return str(cache_root)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str = "cuda"):
    """Load MolmoAct2 model and processor from HuggingFace."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model: {model_id}", flush=True)
    print("First run downloads ~20GB (cached afterward)...", flush=True)

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    print("Model loaded.", flush=True)
    return model, processor


def load_action_tokenizer(tokenizer_id: str = "allenai/MolmoAct2-FAST-Tokenizer"):
    """Load the FAST action tokenizer for discrete mode."""
    from transformers import AutoProcessor

    print(f"Loading action tokenizer: {tokenizer_id}", flush=True)
    return AutoProcessor.from_pretrained(tokenizer_id, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Default robot state (LIBERO 8-dim absolute joint-pose)
# ---------------------------------------------------------------------------

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

    Returns dict with keys: actions, depth_cache, depth_bins.
    """
    # The actual model uses `inference_action_mode` (not `action_mode`)
    kwargs = dict(
        processor=processor,
        images=images,
        task=task,
        state=state,
        norm_tag=norm_tag,
        inference_action_mode=action_mode,
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
# Output helpers
# ---------------------------------------------------------------------------

def format_actions(actions) -> str:
    """Pretty-print the predicted action chunk."""
    arr = actions.cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions)
    arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
    if arr.ndim == 1:
        arr = arr[None, :]
    lines = []
    for t, row in enumerate(arr):
        vals = "  ".join(f"{float(v):+.5f}" for v in row.flat)
        lines.append(f"  t={t:3d}: [{vals}]")
    return "\n".join(lines)


def save_results(actions, output_dir: str):
    """Save predicted actions to artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    arr = actions.cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions)
    # Normalize to 2D: [chunk_len, action_dim]
    arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
    if arr.ndim == 1:
        arr = arr[None, :]
    np.save(os.path.join(output_dir, "actions.npy"), arr)

    metrics = {
        "action_chunk_length": int(arr.shape[0]),
        "action_dim": int(arr.shape[1]),
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
  # Default demo with sample images
  python predict.py

  # Custom images and task
  python predict.py --images cam0.png cam1.png \\
      --task "pick up the red block" \\
      --state 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

  # Think variant with depth reasoning
  python predict.py --model allenai/MolmoAct2-Think-LIBERO \\
      --norm-tag libero --enable-depth-reasoning

  # Discrete action mode
  python predict.py --action-mode discrete
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
        "--output-dir", type=str, default="molmoact2_outputs",
        help="Directory to save action outputs (default: molmoact2_outputs)",
    )

    args = parser.parse_args()

    # Resolve output dir via CVL path helper
    output_dir = resolve_output_path(
        args.output_dir.rstrip("/") + "/",
        default_filename="molmoact2_outputs/",
    ).rstrip("/")

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

    # Optional FAST tokenizer
    action_tokenizer = None
    if args.action_mode == "discrete":
        action_tokenizer = load_action_tokenizer()

    # Resolve images
    if args.images:
        resolved = [resolve_input_path(p) for p in args.images]
        images = [Image.open(p).convert("RGB") for p in resolved]
        print(f"Loaded {len(images)} image(s) from disk.", flush=True)
    else:
        data_dir = ensure_sample_data()
        images = [
            Image.open(os.path.join(data_dir, name)).convert("RGB")
            for name in SAMPLE_FILES
        ]
        print(f"Loaded {len(images)} sample image(s).", flush=True)

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

    save_results(actions, output_dir)
    print(f"\nOutputs saved to: {output_dir}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
