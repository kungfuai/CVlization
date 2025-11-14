"""
FastAvatar Inference Script for CVlization
==========================================
Instant 3D face reconstruction from single image using Gaussian Splatting.

This script:
1. Downloads pretrained model weights to centralized cache
2. Runs feedforward inference from a single image
3. Saves results (PLY file, W vector, DINO points)
"""

import argparse
import os
import sys
from pathlib import Path
import gdown
import zipfile

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.inference_feedforward_no_guidance import FeedforwardInferenceEngine


# Google Drive file ID for pretrained weights
PRETRAINED_WEIGHTS_URL = "https://drive.google.com/uc?id=1_XPTo_1rgzxvGQcRI7Toa3iGagytPTjK"

# Cache directory (CVlization standard: use data/container_cache)
CACHE_DIR = Path(os.getenv("CACHE_DIR", Path(__file__).parent / "data" / "container_cache" / "fastavatar"))


def download_pretrained_weights(cache_dir: Path) -> Path:
    """
    Download and extract pretrained model weights.

    Args:
        cache_dir: Directory to cache downloaded weights

    Returns:
        Path to extracted weights directory
    """
    weights_dir = cache_dir / "pretrained_weights"

    # Check if already downloaded
    encoder_path = weights_dir / "encoder_neutral_flame.pth"
    decoder_path = weights_dir / "decoder_neutral_flame.pth"
    dino_path = weights_dir / "dino_encoder.pth"
    avg_ply = weights_dir / "averaged_model.ply"

    if all(p.exists() for p in [encoder_path, decoder_path, dino_path, avg_ply]):
        print(f"Using cached weights from: {weights_dir}")
        return weights_dir

    print("Downloading pretrained weights...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download zip file
    zip_path = cache_dir / "pretrained_weights.zip"
    if not zip_path.exists():
        print(f"Downloading from Google Drive to {zip_path}")
        gdown.download(PRETRAINED_WEIGHTS_URL, str(zip_path), quiet=False)

    # Extract
    print(f"Extracting to {weights_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)

    # Verify extraction
    if not all(p.exists() for p in [encoder_path, decoder_path, dino_path, avg_ply]):
        raise FileNotFoundError(
            f"Failed to extract all required weights. Expected files:\n"
            f"  - {encoder_path}\n"
            f"  - {decoder_path}\n"
            f"  - {dino_path}\n"
            f"  - {avg_ply}"
        )

    print("Weights downloaded and extracted successfully!")
    return weights_dir


def main():
    parser = argparse.ArgumentParser(
        description='FastAvatar: Instant 3D face reconstruction from single image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to input face image'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results',
        help='Directory to save results'
    )

    # Model weights (optional, will auto-download if not provided)
    parser.add_argument(
        '--encoder_checkpoint', type=str, default=None,
        help='Path to encoder checkpoint (auto-downloads if not specified)'
    )
    parser.add_argument(
        '--decoder_checkpoint', type=str, default=None,
        help='Path to decoder checkpoint (auto-downloads if not specified)'
    )
    parser.add_argument(
        '--dino_checkpoint', type=str, default=None,
        help='Path to DINO encoder checkpoint (auto-downloads if not specified)'
    )

    # Cache
    parser.add_argument(
        '--cache_dir', type=str, default=str(CACHE_DIR),
        help='Directory to cache downloaded model weights'
    )

    # Options
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--no_save_ply', action='store_true',
        help='Skip saving PLY file'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Validation
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # Download weights if not provided
    cache_dir = Path(args.cache_dir)
    if not all([args.encoder_checkpoint, args.decoder_checkpoint, args.dino_checkpoint]):
        print("=" * 60)
        print("Model weights not provided, downloading from Google Drive...")
        print("=" * 60)
        weights_dir = download_pretrained_weights(cache_dir)

        args.encoder_checkpoint = args.encoder_checkpoint or str(weights_dir / "encoder_neutral_flame.pth")
        args.decoder_checkpoint = args.decoder_checkpoint or str(weights_dir / "decoder_neutral_flame.pth")
        args.dino_checkpoint = args.dino_checkpoint or str(weights_dir / "dino_encoder.pth")

    # Verify checkpoints exist
    for name, path in [
        ("Encoder", args.encoder_checkpoint),
        ("Decoder", args.decoder_checkpoint),
        ("DINO", args.dino_checkpoint)
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    print("\n" + "=" * 60)
    print("FASTAVATAR INFERENCE")
    print("=" * 60)
    print(f"Input image: {args.image}")
    print(f"Encoder: {args.encoder_checkpoint}")
    print(f"Decoder: {args.decoder_checkpoint}")
    print(f"DINO: {args.dino_checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Initialize engine
    print("\nInitializing inference engine...")
    engine = FeedforwardInferenceEngine(
        encoder_path=args.encoder_checkpoint,
        decoder_path=args.decoder_checkpoint,
        dino_path=args.dino_checkpoint,
        device=args.device
    )

    # Run inference
    print(f"\nRunning inference...")
    results = engine.predict_from_image(args.image)

    # Save results
    print(f"\nSaving results to: {args.output_dir}")
    engine.save_results(
        results,
        args.output_dir,
        save=not args.no_save_ply
    )

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"W vector shape: {results['w_vector'].shape}")
    print(f"DINO points: {results['dino_points'].shape}")
    print(f"Gaussians: {results['splats']['means'].shape[0]}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)

    if not args.no_save_ply:
        print(f"\nVisualize the 3D reconstruction at: https://superspl.at/editor")
        print(f"Upload file: {Path(args.output_dir) / 'splats.ply'}")


if __name__ == "__main__":
    main()
