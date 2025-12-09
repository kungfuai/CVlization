#!/usr/bin/env python3
"""
HunyuanVideo-1.5 - Text-to-Video and Image-to-Video generation.

A lightweight video generation model (8.3B params) from Tencent that achieves
state-of-the-art quality with efficient inference on consumer GPUs.

Features:
- Text-to-Video (T2V) and Image-to-Video (I2V) generation
- 480p and 720p resolution support
- CPU offloading for low VRAM GPUs (min 14GB)
- CFG distillation for 2x speedup
- Step distillation for 75% speedup (480p I2V)
- Cache inference (deepcache, teacache, taylorcache)
- SageAttention support

This is a wrapper around the upstream generate.py from:
https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5

License: Tencent Hunyuan Community License
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# CVL dual-mode execution support
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
        return Path(".")

    def get_output_dir():
        return Path("outputs")

    def resolve_input_path(path):
        return Path(path)

    def resolve_output_path(path):
        return Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo-1.5 video generation wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-video (480p, basic)
  python predict.py --prompt "A cat walking in the garden"

  # Text-to-video with CFG distillation (2x faster)
  python predict.py --prompt "Ocean waves at sunset" --cfg_distilled

  # Image-to-video with step distillation (fastest, 480p only)
  python predict.py --prompt "The person starts dancing" --image ref.jpg --enable_step_distill

  # 720p generation
  python predict.py --prompt "A beautiful landscape" --resolution 720p
        """
    )

    # Core arguments
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Reference image path for I2V mode (optional)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path (default: outputs/output_<timestamp>.mp4)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["480p", "720p"],
        default="480p",
        help="Video resolution (default: 480p)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="tencent/HunyuanVideo-1.5",
        help="HuggingFace model ID or local path (default: tencent/HunyuanVideo-1.5)"
    )

    # Speed optimizations
    parser.add_argument(
        "--cfg_distilled",
        action="store_true",
        help="Enable CFG distillation for ~2x speedup"
    )
    parser.add_argument(
        "--enable_step_distill",
        action="store_true",
        help="Enable step distillation for ~75%% speedup (480p I2V only, uses 8-12 steps)"
    )
    parser.add_argument(
        "--enable_cache",
        action="store_true",
        help="Enable cache inference (deepcache) for speedup"
    )
    parser.add_argument(
        "--cache_type",
        type=str,
        default="deepcache",
        choices=["deepcache", "teacache", "taylorcache"],
        help="Cache type (default: deepcache)"
    )
    parser.add_argument(
        "--use_sageattn",
        action="store_true",
        help="Enable SageAttention for faster inference"
    )

    # Generation parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of inference steps (default: 50, or 8-12 with step_distill)"
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=121,
        help="Number of frames (default: 121, ~5s at 24fps)"
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="Aspect ratio (default: 16:9)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Memory options
    parser.add_argument(
        "--no_offload",
        action="store_true",
        help="Disable CPU offloading (requires more VRAM but faster)"
    )
    parser.add_argument(
        "--no_sr",
        action="store_true",
        help="Disable super resolution"
    )
    parser.add_argument(
        "--overlap_offload",
        action="store_true",
        help="Enable overlapped group offloading (faster but uses more CPU RAM, ~64GB+ recommended)"
    )

    # Multi-GPU
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel inference (default: 1)"
    )

    args = parser.parse_args()

    # Resolve output path
    if args.output:
        output_path = Path(resolve_output_path(args.output))
    else:
        output_path = Path(resolve_output_path("outputs/output.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve image path for I2V
    image_path = "none"
    if args.image:
        image_path = str(resolve_input_path(args.image))
        if not Path(image_path).exists():
            print(f"Error: Reference image not found: {image_path}")
            sys.exit(1)

    # Build command for upstream generate.py
    script_dir = Path(__file__).parent
    generate_script = script_dir / "generate.py"

    cmd = [
        sys.executable, str(generate_script),
        "--prompt", args.prompt,
        "--image_path", image_path,
        "--resolution", args.resolution,
        "--aspect_ratio", args.aspect_ratio,
        "--seed", str(args.seed),
        "--video_length", str(args.video_length),
        "--output_path", str(output_path),
        "--model_path", args.model_path,
        "--rewrite", "false",  # Disable prompt rewriting (requires external LLM)
        "--sr", "false" if args.no_sr else "true",
    ]

    # Offloading
    if args.no_offload:
        cmd.extend(["--offloading", "false"])

    # Overlap group offloading (disabled by default for systems with <64GB RAM)
    if not args.overlap_offload:
        cmd.extend(["--overlap_group_offloading", "false"])

    # Speed optimizations
    if args.cfg_distilled:
        cmd.extend(["--cfg_distilled", "true"])
    if args.enable_step_distill:
        cmd.extend(["--enable_step_distill", "true"])
    if args.enable_cache:
        cmd.extend(["--enable_cache", "true", "--cache_type", args.cache_type])
    if args.use_sageattn:
        cmd.extend(["--use_sageattn", "true"])

    # Inference steps
    if args.num_inference_steps:
        cmd.extend(["--num_inference_steps", str(args.num_inference_steps)])

    # Print info
    print("=" * 60)
    print("HunyuanVideo-1.5 Video Generation")
    print("=" * 60)
    print(f"Prompt: {args.prompt[:80]}...")
    print(f"Resolution: {args.resolution}")
    print(f"Frames: {args.video_length} (~{args.video_length/24:.1f}s at 24fps)")
    print(f"Output: {output_path}")
    if args.image:
        print(f"Mode: Image-to-Video (I2V)")
        print(f"Reference: {image_path}")
    else:
        print(f"Mode: Text-to-Video (T2V)")
    print(f"Optimizations: cfg_distilled={args.cfg_distilled}, step_distill={args.enable_step_distill}, cache={args.enable_cache}")
    print("=" * 60)

    # Run with torchrun for multi-GPU or python for single GPU
    if args.num_gpus > 1:
        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={args.num_gpus}",
        ] + cmd[1:]  # Skip sys.executable
        cmd = torchrun_cmd

    print(f"Running: {' '.join(cmd[:10])}...")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nVideo saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during generation: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGeneration cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
