#!/usr/bin/env python3
"""
V3: Continuous Space-Time Video Super-Resolution with 3D Fourier Fields (ICLR 2026)
Wraps prs-eth/v3 inference for dockerized execution.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import gdown

try:
    from cvlization.paths import (
        get_input_dir,
        get_output_dir,
        resolve_input_path,
        resolve_output_path,
    )
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_input_dir():
        return os.getcwd()

    def get_output_dir():
        out = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out, exist_ok=True)
        return out

    def resolve_input_path(path, base_dir):
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    def resolve_output_path(path, base_dir):
        return path if os.path.isabs(path) else os.path.join(base_dir, path)


CHECKPOINT_GDRIVE_ID = "15nw5NhEIf7VvetEtQI1cnrLNWPi_9FGj"
CHECKPOINT_FILENAME = "v3.pkl"
V3_REPO_DIR = "/opt/v3"

# Built-in sample clips (hosted on zzsi/cvl HuggingFace dataset)
SAMPLES = {
    "vid4_city": {
        "hf_prefix": "v3_vsr/vid4_city",
        "num_frames": 34,
        "description": "Vid4 'city' clip (34 frames, 704x576)",
    },
}


def ensure_sample_input(name: str, cache_dir: Path) -> tuple[Path, str]:
    """Download a built-in sample clip from HuggingFace."""
    from huggingface_hub import hf_hub_download

    sample = SAMPLES[name]
    eval_set = name
    # HFEvalVideoFolder expects: data_dir/eval_set/<video_subdir>/<frame>.png
    frames_dir = cache_dir / "samples" / eval_set / "000"

    if frames_dir.exists() and len(list(frames_dir.glob("*.png"))) >= sample["num_frames"]:
        print(f"Sample '{name}' found in cache: {frames_dir}")
        return cache_dir / "samples", eval_set

    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample '{name}' ({sample['description']})...")
    token = os.environ.get("HF_TOKEN")

    for i in range(sample["num_frames"]):
        fname = f"{i:08d}.png"
        downloaded = hf_hub_download(
            repo_id="zzsi/cvl",
            repo_type="dataset",
            filename=f"{sample['hf_prefix']}/{fname}",
            token=token,
        )
        # Copy to expected directory structure
        import shutil
        shutil.copy2(downloaded, frames_dir / fname)

    print(f"Sample saved to: {frames_dir}")
    return cache_dir / "samples", eval_set


def ensure_checkpoint(cache_dir: Path) -> Path:
    """Download V3 checkpoint from Google Drive if not cached."""
    ckpt_path = cache_dir / CHECKPOINT_FILENAME
    if ckpt_path.exists():
        print(f"Checkpoint found: {ckpt_path}")
        return ckpt_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading V3 checkpoint from Google Drive...")
    url = f"https://drive.google.com/uc?id={CHECKPOINT_GDRIVE_ID}"
    gdown.download(url, str(ckpt_path), quiet=False)

    if not ckpt_path.exists():
        raise RuntimeError(f"Failed to download checkpoint to {ckpt_path}")
    print(f"Checkpoint saved: {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(
        description="V3: Continuous Space-Time Video Super-Resolution",
    )
    parser.add_argument(
        "--input", default="vid4_city",
        help="Path to input video directory, or a built-in sample name: "
             f"{list(SAMPLES.keys())}. Default: vid4_city",
    )
    parser.add_argument(
        "--output", default="v3_result",
        help="Output directory for super-resolved frames (default: v3_result under output dir)",
    )
    parser.add_argument(
        "--space-scale", type=int, default=4,
        help="Spatial upsampling factor (default: 4)",
    )
    parser.add_argument(
        "--time-scale", type=int, default=2,
        help="Temporal upsampling factor (default: 2)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to V3 checkpoint. If omitted, downloads from Google Drive.",
    )
    parser.add_argument(
        "--cache-dir", default="/root/.cache/cvlization/v3",
        help="Cache directory for checkpoint and sample data.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    INP = get_input_dir()
    OUT = get_output_dir()
    cache_dir = Path(args.cache_dir)

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = ensure_checkpoint(cache_dir / "checkpoints")

    # Resolve input
    if args.input in SAMPLES:
        data_dir, eval_set = ensure_sample_input(args.input, cache_dir)
        data_dir_arg = str(data_dir)
    else:
        input_path = Path(resolve_input_path(args.input, INP))
        if not input_path.exists():
            print(f"Error: Input path not found: {input_path}")
            return 1
        # The upstream script expects --data-dir to contain a subfolder named by --eval-sets
        data_dir_arg = str(input_path.parent)
        eval_set = input_path.name

    # Resolve output
    output_path = Path(resolve_output_path(args.output, OUT))
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(V3_REPO_DIR, "run_inference.py"),
        "--data-dir", data_dir_arg,
        "--checkpoint-path", str(ckpt_path),
        "--eval-sets", eval_set,
        "--space-scale", str(args.space_scale),
        "--time-scale", str(args.time_scale),
        "--save-dir", str(output_path),
    ]

    print(f"Running V3 inference (space={args.space_scale}x, time={args.time_scale}x):")
    if args.verbose:
        print(" ".join(cmd))

    result = subprocess.run(cmd, cwd=V3_REPO_DIR)
    if result.returncode != 0:
        print(f"Error: V3 inference failed with exit code {result.returncode}")
        return result.returncode

    # Count output frames
    out_frames = list(output_path.rglob("*.png"))
    print(f"Done. {len(out_frames)} frames saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
