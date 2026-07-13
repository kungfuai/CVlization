#!/usr/bin/env python3
"""
LingBot-Vision: Dense Spatial Perception Feature Extraction

Loads the LingBot-Vision ViT backbone (pretrained with masked boundary modeling)
and extracts dense patch-level features from input images. Visualises features
using PCA projection to RGB, revealing boundary-aware spatial structure.

Reference: "Vision Pretraining for Dense Spatial Perception" (arXiv 2607.05247)
License: Apache 2.0
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from cvlization.paths import resolve_input_path, resolve_output_path

# ---------------------------------------------------------------------------
# Sample data from HuggingFace (lazy-downloaded on first run)
# ---------------------------------------------------------------------------
DEFAULT_INPUT = "data/images"
HF_REPO_ID = "zzsi/cvl"
HF_SAMPLE_FILES = [
    # Reuse images already hosted in zzsi/cvl under other example prefixes.
    ("dust3r/desk_view1.jpg", "sample_indoor.jpg"),
    ("samples/pose_estimation/dwpose/human.png", "sample_scene.png"),
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def download_sample_inputs_if_needed(input_dir: Path) -> None:
    """Download sample inputs from HuggingFace if they don't exist locally."""
    existing = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ] if input_dir.exists() else []

    if len(existing) >= 2:
        return

    print("Downloading sample inputs from HuggingFace...", flush=True)
    from huggingface_hub import hf_hub_download

    input_dir.mkdir(parents=True, exist_ok=True)

    for hf_path, local_name in HF_SAMPLE_FILES:
        local_path = input_dir / local_name
        if not local_path.exists():
            print(f"  Downloading {local_name}...", flush=True)
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                repo_type="dataset",
            )
            import shutil
            shutil.copy2(downloaded, local_path)

    print("  Sample inputs ready.", flush=True)


# ---------------------------------------------------------------------------
# PCA visualisation
# ---------------------------------------------------------------------------

def pca_rgb(tokens: torch.Tensor, grid: tuple) -> np.ndarray:
    """Project patch tokens to RGB via PCA (top 3 principal components).

    Args:
        tokens: Patch tokens of shape [B, N, C] (uses first batch element).
        grid: Spatial grid (H_patches, W_patches).

    Returns:
        pca_map: RGB image array of shape (H_patches, W_patches, 3) in [0, 1].
    """
    feats = tokens[0].float().cpu().numpy()  # [N, C]
    mean = feats.mean(axis=0, keepdims=True)
    feats_c = feats - mean

    # SVD -> top-3 components
    _, _, Vt = np.linalg.svd(feats_c, full_matrices=False)
    pca3 = feats_c @ Vt[:3].T  # [N, 3]

    # Normalise to 1-99 percentile range for contrast
    lo = np.percentile(pca3, 1, axis=0, keepdims=True)
    hi = np.percentile(pca3, 99, axis=0, keepdims=True)
    pca3 = np.clip((pca3 - lo) / (hi - lo + 1e-8), 0, 1)

    H, W = grid
    return pca3.reshape(H, W, 3)


def make_panel(img_rgb: np.ndarray, pca_map: np.ndarray, title: str) -> np.ndarray:
    """Create a labelled side-by-side panel: input | PCA features."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(pca_map)
    axes[1].set_title("PCA Features (top-3)", fontsize=12)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout()

    # Render to array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LingBot-Vision dense feature extraction with PCA visualisation"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input image file or directory (default: bundled HF samples)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--variant", type=str, default="large",
        choices=["small", "base", "large", "giant"],
        help="Model variant (default: large)",
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Resize images to this size before inference (default: 512)",
    )
    parser.add_argument(
        "--resize-mode", type=str, default="square",
        choices=["square", "shortest"],
        help="Resize mode: square or shortest-side crop (default: square)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Inference precision (default: bf16)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: auto-detect)",
    )
    args = parser.parse_args()

    # Deferred import so --help is fast
    from lingbot_vision import (
        DTYPE_MAP,
        extract_patch_tokens,
        load_image,
        load_pretrained_backbone,
    )

    dtype = DTYPE_MAP.get(args.dtype, torch.float32)

    # ---- Resolve input path ------------------------------------------------
    if args.input is None:
        input_path = Path(DEFAULT_INPUT)
        print(f"No --input provided, using bundled samples: {DEFAULT_INPUT}", flush=True)
        download_sample_inputs_if_needed(input_path)
    else:
        input_path = Path(resolve_input_path(args.input))

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", flush=True)
        sys.exit(1)

    # Collect image files
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS
        )

    if not image_files:
        print(f"Error: No images found in {input_path}", flush=True)
        sys.exit(1)

    # ---- Resolve output path -----------------------------------------------
    output_path = Path(resolve_output_path(args.output))
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Print config ------------------------------------------------------
    print(f"LingBot-Vision Dense Feature Extraction", flush=True)
    print(f"  Variant : {args.variant}", flush=True)
    print(f"  Size    : {args.image_size}px ({args.resize_mode})", flush=True)
    print(f"  Dtype   : {args.dtype}", flush=True)
    print(f"  Device  : {args.device}", flush=True)
    print(f"  Input   : {input_path}  ({len(image_files)} image(s))", flush=True)
    print(f"  Output  : {output_path}", flush=True)
    print(flush=True)

    # ---- Load model --------------------------------------------------------
    print("Loading LingBot-Vision backbone...", flush=True)
    backbone, embed_dim = load_pretrained_backbone(
        variant=args.variant,
        device=args.device,
        dtype=dtype,
    )
    print(
        f"  Loaded ViT-{args.variant[0].upper()} | "
        f"patch_size={backbone.patch_size} | embed_dim={embed_dim}",
        flush=True,
    )
    print(flush=True)

    # ---- Process images ----------------------------------------------------
    metrics = {"images_processed": 0, "variant": args.variant, "embed_dim": embed_dim}

    for img_path in image_files:
        stem = img_path.stem
        print(f"Processing: {img_path.name}", flush=True)

        # Load and preprocess
        img_norm, img_rgb, (H, W) = load_image(
            str(img_path),
            size=args.image_size,
            patch_size=backbone.patch_size,
            mode=args.resize_mode,
        )

        # Extract patch tokens
        with torch.no_grad():
            patch_tokens, patch_grid = extract_patch_tokens(
                backbone, img_norm, args.device, dtype
            )

        ph, pw = patch_grid
        print(
            f"  Patches: {ph}x{pw} = {ph * pw} tokens, "
            f"each {embed_dim}-dim",
            flush=True,
        )

        # PCA visualisation
        pca_map = pca_rgb(patch_tokens, patch_grid)

        # Upscale PCA map to input resolution
        pca_img = Image.fromarray((pca_map * 255).astype(np.uint8))
        pca_full = pca_img.resize((W, H), Image.BILINEAR)

        # Save individual PCA image
        pca_path = output_path / f"{stem}_pca.png"
        pca_full.save(pca_path)
        print(f"  Saved: {pca_path.name}", flush=True)

        # Save labelled comparison panel
        panel = make_panel(img_rgb, np.array(pca_full), img_path.name)
        panel_path = output_path / f"{stem}_panel.png"
        Image.fromarray(panel).save(panel_path)
        print(f"  Saved: {panel_path.name}", flush=True)

        metrics["images_processed"] += 1

    # ---- Write metrics.json ------------------------------------------------
    import json

    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(flush=True)
    print(f"Metrics saved to: {metrics_path}", flush=True)

    # ---- Summary -----------------------------------------------------------
    print(flush=True)
    print(f"Done. Processed {metrics['images_processed']} image(s).", flush=True)
    print(f"Results in: {output_path}", flush=True)


if __name__ == "__main__":
    main()
