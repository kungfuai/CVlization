#!/usr/bin/env python3
"""
MapAnything inference script for 3D reconstruction from images.
"""
import argparse
import sys
from pathlib import Path

import torch

from cvlization.paths import resolve_input_path, resolve_output_path
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Run MapAnything inference for 3D reconstruction"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to directory containing input images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/map-anything",
        help="Model name or path (default: facebook/map-anything)",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Automatic mixed precision dtype (default: bf16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu)",
    )
    args = parser.parse_args()

    # Setup paths
    images_path = Path(resolve_input_path(args.images))
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not images_path.exists():
        print(f"Error: Images path does not exist: {images_path}")
        sys.exit(1)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Import MapAnything modules
    print("Loading MapAnything modules...")
    try:
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images
    except ImportError as e:
        print(f"Error importing MapAnything: {e}")
        print("Make sure MapAnything is installed: pip install -e /opt/map-anything")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = MapAnything.from_pretrained(args.model).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load images
    print(f"Loading images from: {images_path}")
    try:
        views = load_images(str(images_path))
        print(f"Loaded {len(views)} images")
    except Exception as e:
        print(f"Error loading images: {e}")
        sys.exit(1)

    # Run inference
    print("Running inference...")
    try:
        with torch.no_grad():
            predictions = model.infer(views, use_amp=True, amp_dtype=args.amp_dtype)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

    # Save results
    print(f"Saving results to: {output_path}")
    try:
        # Handle both single dict and list of dicts
        if isinstance(predictions, list):
            pred = predictions[0] if len(predictions) > 0 else {}
        else:
            pred = predictions

        # Save point cloud (XYZ coordinates)
        if "points" in pred:
            points = pred["points"].cpu().numpy()
            points_file = output_path / "point_cloud.npy"
            np.save(points_file, points)
            print(f"  Saved point cloud: {points_file} (shape: {points.shape})")

            # Also save as simple text format for easy viewing
            points_txt = output_path / "point_cloud.txt"
            np.savetxt(points_txt, points.reshape(-1, 3), fmt="%.6f")
            print(f"  Saved point cloud (txt): {points_txt}")

        # Save depth maps
        if "depth" in pred:
            depth = pred["depth"].cpu().numpy()
            depth_file = output_path / "depth_maps.npy"
            np.save(depth_file, depth)
            print(f"  Saved depth maps: {depth_file} (shape: {depth.shape})")

        # Save camera intrinsics
        if "intrinsics" in pred:
            intrinsics = pred["intrinsics"].cpu().numpy()
            intrinsics_file = output_path / "camera_intrinsics.npy"
            np.save(intrinsics_file, intrinsics)
            print(f"  Saved camera intrinsics: {intrinsics_file} (shape: {intrinsics.shape})")

        # Save camera poses
        if "poses" in pred:
            poses = pred["poses"].cpu().numpy()
            poses_file = output_path / "camera_poses.npy"
            np.save(poses_file, poses)
            print(f"  Saved camera poses: {poses_file} (shape: {poses.shape})")

        # Save confidence scores
        if "confidence" in pred:
            confidence = pred["confidence"].cpu().numpy()
            confidence_file = output_path / "confidence.npy"
            np.save(confidence_file, confidence)
            print(f"  Saved confidence scores: {confidence_file} (shape: {confidence.shape})")

        # Save validity masks
        if "valid" in pred:
            valid = pred["valid"].cpu().numpy()
            valid_file = output_path / "validity_mask.npy"
            np.save(valid_file, valid)
            print(f"  Saved validity mask: {valid_file} (shape: {valid.shape})")

        # Save summary
        summary_file = output_path / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("MapAnything Inference Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Images: {images_path}\n")
            f.write(f"Number of images: {len(views)}\n")
            f.write(f"Predictions type: {type(predictions)}\n\n")
            f.write("Output shapes:\n")
            if isinstance(pred, dict):
                for key, value in pred.items():
                    if hasattr(value, "shape"):
                        f.write(f"  {key}: {value.shape}\n")
        print(f"  Saved summary: {summary_file}")

    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
