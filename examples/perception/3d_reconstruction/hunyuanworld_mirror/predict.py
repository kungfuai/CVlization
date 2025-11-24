#!/usr/bin/env python3
"""
HunyuanWorld-Mirror inference for 3D reconstruction.

Simplified wrapper around the repository's infer.py script.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run HunyuanWorld-Mirror 3D reconstruction inference"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/images",
        help="Input directory of images or video file (default: data/images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=518,
        help="Target image size (default: 518)",
    )
    parser.add_argument(
        "--save-pointmap",
        action="store_true",
        default=True,
        help="Save point cloud PLY (default: True)",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        default=True,
        help="Save depth maps (default: True)",
    )
    parser.add_argument(
        "--save-normal",
        action="store_true",
        default=True,
        help="Save surface normals (default: True)",
    )
    parser.add_argument(
        "--save-gs",
        action="store_true",
        default=False,
        help="Save 3D Gaussians PLY (default: False, requires CUDA toolkit)",
    )
    parser.add_argument(
        "--save-colmap",
        action="store_true",
        default=False,
        help="Save COLMAP reconstruction (default: False)",
    )
    parser.add_argument(
        "--save-rendered",
        action="store_true",
        default=False,
        help="Save rendered interpolation video (default: False, requires CUDA toolkit)",
    )
    parser.add_argument(
        "--apply-sky-mask",
        action="store_true",
        default=False,
        help="Apply sky segmentation filtering (default: False)",
    )
    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Build command for repo's infer.py
    cmd = [
        "python",
        "/opt/hunyuanworld-mirror/infer.py",
        "--input_path",
        str(input_path),
        "--output_path",
        args.output,
        "--target_size",
        str(args.target_size),
    ]

    # Add boolean flags
    if args.save_pointmap:
        cmd.append("--save_pointmap")
    if args.save_depth:
        cmd.append("--save_depth")
    if args.save_normal:
        cmd.append("--save_normal")
    if args.save_gs:
        cmd.append("--save_gs")
    if args.save_colmap:
        cmd.append("--save_colmap")
    if args.save_rendered:
        cmd.append("--save_rendered")
    if args.apply_sky_mask:
        cmd.append("--apply_sky_mask")

    print(f"Running HunyuanWorld-Mirror inference...")
    print(f"  Input: {input_path}")
    print(f"  Output: {args.output}")
    print(f"  Target size: {args.target_size}px")
    print()

    # Run the inference script
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Inference completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference failed with error code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
