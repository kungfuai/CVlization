#!/usr/bin/env python3
"""
DUSt3R: Geometric 3D Vision Made Easy

3D reconstruction from uncalibrated images using dense matching and optimization.
Outputs 3D scene (GLB), depth maps, and confidence maps.

License: CC BY-NC-SA 4.0 (non-commercial use only)
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from cvlization.paths import resolve_input_path, resolve_output_path

# DUSt3R imports
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import matplotlib.pyplot as pl


def main():
    parser = argparse.ArgumentParser(
        description="Run DUSt3R 3D reconstruction inference"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/images",
        help="Input directory of images (default: data/images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        help="HuggingFace model name or local checkpoint path",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        choices=[224, 512],
        help="Image size for inference (default: 512)",
    )
    parser.add_argument(
        "--scene-graph",
        type=str,
        default="complete",
        choices=["complete", "swin", "oneref"],
        help="Scene graph type (default: complete)",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=300,
        help="Number of global alignment iterations (default: 300)",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Learning rate schedule (default: cosine)",
    )
    parser.add_argument(
        "--min-conf-thr",
        type=float,
        default=3.0,
        help="Minimum confidence threshold for point cloud (default: 3.0)",
    )
    parser.add_argument(
        "--as-pointcloud",
        action="store_true",
        default=False,
        help="Export as point cloud instead of mesh",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(resolve_input_path(args.input))
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(resolve_output_path(args.output))
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"DUSt3R 3D Reconstruction")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_name).to(args.device)
    model.eval()
    print("Model loaded successfully")
    print()

    # Load images
    print("Loading images...")
    # Use os.listdir to avoid symlink resolution issues
    import os
    try:
        image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))]
        filelist = sorted([os.path.join(str(input_path), f) for f in image_files])
    except Exception as e:
        print(f"Error reading input directory: {e}")
        sys.exit(1)

    if len(filelist) == 0:
        print(f"Error: No images found in {input_path}")
        sys.exit(1)

    print(f"Found {len(filelist)} images")

    try:
        square_ok = model.square_ok
    except:
        square_ok = False

    imgs = load_images(filelist, size=args.image_size, verbose=True,
                      patch_size=model.patch_size, square_ok=square_ok)

    # Handle single image case
    if len(imgs) == 1:
        import copy
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        print("Single image provided, duplicating for reconstruction")

    print(f"Loaded {len(imgs)} images")
    print()

    # Create image pairs
    print(f"Creating image pairs (scene graph: {args.scene_graph})...")
    pairs = make_pairs(imgs, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} pairs")
    print()

    # Run inference
    print("Running inference...")
    output = inference(pairs, model, args.device, batch_size=1, verbose=True)
    print("Inference complete")
    print()

    # Global alignment
    print("Running global alignment...")
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=args.device, mode=mode, verbose=True)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        lr = 0.01
        loss = scene.compute_global_alignment(
            init='mst',
            niter=args.niter,
            schedule=args.schedule,
            lr=lr
        )
        print(f"Global alignment complete (final loss: {loss:.6f})")
    else:
        print("Pair viewer mode (no global alignment needed)")
    print()

    # Export 3D scene
    print("Exporting 3D scene...")
    from dust3r.demo import _convert_scene_output_to_glb

    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.min_conf_thr)))
    msk = to_numpy(scene.get_masks())

    scene_file = _convert_scene_output_to_glb(
        str(output_path), rgbimg, pts3d, msk, focals, cams2world,
        as_pointcloud=args.as_pointcloud,
        transparent_cams=False,
        cam_size=0.05,
        silent=False
    )
    print(f"✓ Saved 3D scene: {scene_file}")

    # Save depth and confidence maps
    print()
    print("Exporting depth and confidence maps...")
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])

    # Normalize depths
    depths_max = max([d.max() for d in depths])
    depths_norm = [d / depths_max for d in depths]

    # Apply colormap to confidence
    cmap = pl.get_cmap('jet')
    confs_max = max([d.max() for d in confs])
    confs_colored = [cmap(d / confs_max) for d in confs]

    # Save maps
    for i in range(len(depths)):
        # Save depth map
        depth_path = output_path / f"depth_{i:04d}.png"
        depth_img = Image.fromarray((rgb(depths_norm[i]) * 255).astype(np.uint8))
        depth_img.save(depth_path)

        # Save raw depth as numpy
        depth_raw_path = output_path / f"depth_{i:04d}.npy"
        np.save(depth_raw_path, depths[i])

        # Save confidence map
        conf_path = output_path / f"confidence_{i:04d}.png"
        conf_img = Image.fromarray((rgb(confs_colored[i]) * 255).astype(np.uint8))
        conf_img.save(conf_path)

        print(f"✓ Saved view {i}: depth, raw depth, confidence")

    print()
    print("✅ Reconstruction completed successfully!")
    print(f"📁 Results saved to: {output_path}")

    # Summary
    total_points = sum(m.sum() for m in msk)
    print()
    print("Summary:")
    print(f"  - Views: {len(imgs)}")
    print(f"  - Total 3D points: {total_points:,}")
    print(f"  - Scene file: scene.glb")
    print(f"  - Depth maps: {len(depths)} views")
    print(f"  - Confidence maps: {len(confs)} views")


if __name__ == "__main__":
    main()
