#!/usr/bin/env python3
"""
MonST3R: 3D Reconstruction from Dynamic Videos

Performs 4D reconstruction (3D + time) from video/image sequences.
Outputs time-varying point clouds, depth maps, camera poses, and dynamic masks.

License: CC BY-NC-SA 4.0 (non-commercial use only)
"""
import argparse
import sys
import os
from pathlib import Path
import torch
import subprocess

# Add monst3r to path
sys.path.insert(0, '/opt/monst3r')

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb


def download_checkpoint_if_needed(checkpoint_path, download_cmd, name):
    """
    Download checkpoint if it doesn't exist.

    Args:
        checkpoint_path: Path where checkpoint should be
        download_cmd: Command to run to download (list of strings)
        name: Human-readable name for logging
    """
    if os.path.exists(checkpoint_path):
        print(f"✓ {name} checkpoint found: {checkpoint_path}")
        return True

    print(f"Downloading {name} checkpoint...")
    try:
        subprocess.run(download_cmd, check=True, cwd=os.path.dirname(checkpoint_path) or '.')
        if os.path.exists(checkpoint_path):
            print(f"✓ {name} checkpoint downloaded")
            return True
        else:
            print(f"✗ {name} checkpoint download failed (file not found after download)")
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} checkpoint download failed: {e}")
        return False


def ensure_checkpoints():
    """
    Ensure all required checkpoints are downloaded.
    Downloads lazily on first run, cached for subsequent runs.
    """
    print("Checking checkpoints...")

    # SAM2 checkpoint (required for dynamic mask segmentation)
    sam2_path = "/opt/monst3r/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    os.makedirs(os.path.dirname(sam2_path), exist_ok=True)
    download_checkpoint_if_needed(
        sam2_path,
        ["wget", "-q", sam2_url, "-O", sam2_path],
        "SAM2"
    )

    # RAFT checkpoints (required for optical flow)
    raft_dir = "/opt/monst3r/third_party/RAFT"
    raft_models_dir = f"{raft_dir}/models"
    raft_checkpoint = f"{raft_models_dir}/raft-things.pth"

    if not os.path.exists(raft_checkpoint):
        print("Downloading RAFT checkpoints...")
        os.makedirs(raft_models_dir, exist_ok=True)
        try:
            # Download models.zip from Dropbox
            subprocess.run([
                "wget", "-q",
                "https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip",
                "-O", f"{raft_dir}/models.zip"
            ], check=True)

            # Unzip
            subprocess.run([
                "unzip", "-q", f"{raft_dir}/models.zip",
                "-d", raft_dir
            ], check=True)

            # Clean up
            os.remove(f"{raft_dir}/models.zip")
            print("✓ RAFT checkpoints downloaded")
        except Exception as e:
            print(f"✗ RAFT checkpoint download failed: {e}")
            return False
    else:
        print(f"✓ RAFT checkpoint found: {raft_checkpoint}")

    print("✓ All checkpoints ready")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run MonST3R 4D reconstruction inference"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory of images or video file",
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
        default="Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt",
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
        "--min-conf-thr",
        type=float,
        default=3.0,
        help="Minimum confidence threshold for point cloud (default: 3.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=200,
        help="Maximum number of frames for video processing (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--not-batchify",
        action="store_true",
        default=True,
        help="Disable batchify mode for global optimization (reduces memory usage, default: True)",
    )
    parser.add_argument(
        "--batchify",
        action="store_false",
        dest="not_batchify",
        help="Enable batchify mode for global optimization (requires more VRAM)",
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"MonST3R 4D Reconstruction")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {args.device}")
    print()

    # Ensure checkpoints are downloaded
    if not ensure_checkpoints():
        print("Error: Failed to download required checkpoints")
        sys.exit(1)

    print()
    print("Loading model...")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_name).to(args.device)
    model.eval()
    print("✓ Model loaded")
    print()

    # Load images
    print("Loading images...")
    if input_path.is_file():
        # Video file
        filelist = str(input_path)
        print(f"Processing video: {input_path.name}")
    else:
        # Image directory
        filelist = str(input_path)
        print(f"Processing image directory")

    imgs = load_images(
        filelist,
        size=args.image_size,
        verbose=True,
        num_frames=args.num_frames
    )

    if len(imgs) == 0:
        print(f"Error: No images found in {input_path}")
        sys.exit(1)

    print(f"✓ Loaded {len(imgs)} images/frames")
    print()

    # Create image pairs
    print(f"Creating image pairs (scene graph: {args.scene_graph})...")
    pairs = make_pairs(imgs, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    print(f"✓ Created {len(pairs)} pairs")
    print()

    # Run inference
    print("Running inference...")
    output = inference(pairs, model, args.device, batch_size=args.batch_size, verbose=True)
    print("✓ Inference complete")
    print()

    # Global alignment
    print("Running global alignment...")
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(
            output,
            device=args.device,
            mode=mode,
            verbose=True,
            num_total_iter=args.niter,
            empty_cache=len(imgs) > 72,
            batchify=not args.not_batchify
        )
        scene.compute_global_alignment(init='mst', niter=args.niter, schedule='cosine', lr=0.01)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=args.device, mode=mode, verbose=True)

    print("✓ Global alignment complete")
    print()

    # Export results
    print("Exporting results...")

    # 3D scene (GLB)
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = args.min_conf_thr
    msk = to_numpy(scene.get_masks())
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    glb_file = convert_scene_output_to_glb(
        str(output_path),
        scene.imgs,
        pts3d,
        msk,
        focals,
        cams2world,
        as_pointcloud=False,
        transparent_cams=False,
        cam_size=0.05,
        show_cam=True,
        silent=False
    )
    print(f"✓ Saved 3D scene: {glb_file}")

    # Camera poses
    scene.save_tum_poses(str(output_path / "pred_traj.txt"))
    print(f"✓ Saved camera trajectory")

    # Intrinsics
    scene.save_intrinsics(str(output_path / "pred_intrinsics.txt"))
    print(f"✓ Saved camera intrinsics")

    # Depth maps
    scene.save_depth_maps(str(output_path))
    print(f"✓ Saved depth maps for {len(imgs)} frames")

    # Confidence maps
    scene.save_conf_maps(str(output_path))
    print(f"✓ Saved confidence maps")

    # Dynamic masks (if available)
    scene.save_dynamic_masks(str(output_path))
    print(f"✓ Saved dynamic masks")

    # RGB images
    scene.save_rgb_imgs(str(output_path))
    print(f"✓ Saved RGB images")

    print()
    print("✅ Reconstruction completed successfully!")
    print(f"📁 Results saved to: {output_path}")
    print()
    print("Summary:")
    print(f"  - Frames: {len(imgs)}")
    print(f"  - Scene file: scene.glb")
    print(f"  - Camera trajectory: pred_traj.txt")
    print(f"  - Depth/confidence/RGB maps saved")


if __name__ == "__main__":
    main()
