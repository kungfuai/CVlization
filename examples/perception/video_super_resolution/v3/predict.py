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


def ensure_sample_data(cache_dir: Path) -> tuple[Path, str]:
    """Create sample data for testing.

    Generates 29 frames at 256x256 so the model has enough source frames
    (the encoder expects up to SEQ_LEN_ENC=14 source frames; with time_scale=2
    we need at least (14-1)*2+1=27 target frames).

    Returns (data_dir, eval_set_name) where data_dir/eval_set_name/ contains the PNGs.
    This matches how HFEvalVideoFolder expects: each subdirectory is one video.
    """
    NUM_FRAMES = 29
    FRAME_SIZE = 256
    EVAL_SET = "sample_clip"
    data_dir = cache_dir / "sample_data"
    # HFEvalVideoFolder expects: root/<video_dir>/<frame>.png
    # So frames go in data_dir/EVAL_SET/000/
    frames_dir = data_dir / EVAL_SET / "000"
    if frames_dir.exists() and len(list(frames_dir.glob("*.png"))) >= NUM_FRAMES:
        return data_dir, EVAL_SET

    frames_dir.mkdir(parents=True, exist_ok=True)
    print("Generating sample frames for testing...")

    from PIL import Image
    import numpy as np

    rng = np.random.RandomState(42)
    for i in range(NUM_FRAMES):
        # Textured frames with slow motion to simulate a real video
        y, x = np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE].astype(np.float32)
        phase = 2 * np.pi * i / NUM_FRAMES
        r = np.clip(128 + 80 * np.sin(x / 20.0 + phase), 0, 255)
        g = np.clip(128 + 80 * np.sin(y / 25.0 + phase * 0.7), 0, 255)
        b = np.clip(128 + 80 * np.cos((x + y) / 30.0 + phase * 1.3), 0, 255)
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        # Add slight noise for texture
        noise = rng.randint(-10, 11, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(frames_dir / f"{i:04d}.png")

    print(f"Sample frames saved to: {frames_dir}")
    return data_dir, EVAL_SET


def main():
    parser = argparse.ArgumentParser(
        description="V3: Continuous Space-Time Video Super-Resolution",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to input video directory (folder of PNG frames or a video file). "
             "If omitted, generates sample test frames.",
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
    if args.input is None:
        data_dir, eval_set = ensure_sample_data(cache_dir)
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
