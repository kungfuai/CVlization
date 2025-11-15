#!/usr/bin/env python3
"""
Download and cache BFM (Basel Face Model) 2009 from HuggingFace.

The BFM model is required for the original face tracking pipeline.
This script downloads from a HuggingFace mirror to avoid manual registration.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# Cache directory (can be mounted from host for persistence)
CACHE_DIR = Path(os.environ.get("EGSTALKER_CACHE", "~/.cache/egstalker")).expanduser()
BFM_CACHE_DIR = CACHE_DIR / "bfm"


def download_file_from_url(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """
    Download a file with progress bar using requests.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
    """
    import requests

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle HuggingFace blob URLs - convert to resolve endpoint
    if "huggingface.co" in url and "/blob/" in url:
        # Convert blob URL to direct download URL
        url = url.replace("/blob/", "/resolve/")

    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_bfm_model(
    force_download: bool = False,
    cache_dir: Optional[Path] = None,
    download_all: bool = True
) -> Path:
    """
    Download and cache the BFM 2009 model file and supporting data.

    Args:
        force_download: If True, re-download even if cached
        cache_dir: Custom cache directory (default: ~/.cache/egstalker/bfm)
        download_all: If True, also download supporting BFM files from SadTalker

    Returns:
        Path to the cached BFM directory

    Raises:
        RuntimeError: If download fails
    """
    # Set up cache directory
    if cache_dir is None:
        cache_dir = BFM_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / "01_MorphableModel.mat"

    # Check if already cached
    if model_path.exists() and not force_download:
        print(f"BFM model already cached at: {model_path}")
        if not download_all:
            return cache_dir

    print("Downloading BFM 2009 model from HuggingFace...")
    print("Source: SadTalker repository (wsj1995/sadTalker)")

    # Base URL for SadTalker repository on HuggingFace
    base_url = "https://huggingface.co/wsj1995/sadTalker/resolve/af80749f8c9af3702fbd0272df14ff086986a1de"

    # Core BFM model file
    if not model_path.exists() or force_download:
        try:
            download_file_from_url(
                f"{base_url}/01_MorphableModel.mat",
                model_path,
                desc="Downloading 01_MorphableModel.mat"
            )
        except Exception as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download BFM model: {e}")

    # Download additional supporting files from SadTalker (optional)
    if download_all:
        supporting_files = [
            "BFM09_model_info.mat",
            "BFM_exp_idx.mat",
            "BFM_front_idx.mat",
            "Exp_Pca.bin",
            "facemodel_info.mat",
            "similarity_Lm3D_all.mat",
            "std_exp.txt"
        ]

        for filename in supporting_files:
            file_path = cache_dir / filename
            if not file_path.exists() or force_download:
                try:
                    download_file_from_url(
                        f"{base_url}/{filename}",
                        file_path,
                        desc=f"Downloading {filename}"
                    )
                except Exception as e:
                    print(f"Warning: Could not download {filename}: {e}")
                    # Continue with other files

        # Download required .npy files from AD-NeRF repository
        print("\nDownloading required .npy files from AD-NeRF repository...")
        adnerf_base = "https://github.com/YudongGuo/AD-NeRF/raw/master/data_util/face_tracking/3DMM"
        npy_files = [
            "exp_info.npy",
            "keys_info.npy",
            "topology_info.npy"
        ]

        for filename in npy_files:
            file_path = cache_dir / filename
            if not file_path.exists() or force_download:
                try:
                    download_file_from_url(
                        f"{adnerf_base}/{filename}",
                        file_path,
                        desc=f"Downloading {filename}"
                    )
                except Exception as e:
                    print(f"Warning: Could not download {filename}: {e}")
                    # Continue with other files

    print(f"\nBFM model successfully cached at: {cache_dir}")
    print(f"Main model size: {model_path.stat().st_size / (1024**2):.2f} MB")

    # Verify all required files
    required_files = ["01_MorphableModel.mat", "exp_info.npy", "keys_info.npy", "topology_info.npy"]
    missing = [f for f in required_files if not (cache_dir / f).exists()]
    if missing:
        print(f"\nWarning: Missing required files: {', '.join(missing)}")
    else:
        print("✓ All required BFM files present")

    return cache_dir


def get_bfm_model_path(auto_download: bool = True) -> Path:
    """
    Get path to cached BFM model, optionally downloading if not present.

    Args:
        auto_download: If True, download model if not cached

    Returns:
        Path to the BFM model file

    Raises:
        FileNotFoundError: If model not cached and auto_download=False
    """
    model_path = BFM_CACHE_DIR / "01_MorphableModel.mat"

    if not model_path.exists():
        if auto_download:
            print("BFM model not found in cache. Downloading...")
            return download_bfm_model()
        else:
            raise FileNotFoundError(
                f"BFM model not found at: {model_path}\n"
                f"Run: python -m data_utils.bfm_model.download_bfm"
            )

    return model_path


def main():
    """Command-line interface for downloading BFM model."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and cache BFM 2009 model from HuggingFace"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=BFM_CACHE_DIR,
        help=f"Custom cache directory (default: {BFM_CACHE_DIR})"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if model is cached, don't download"
    )

    args = parser.parse_args()

    if args.check:
        model_path = BFM_CACHE_DIR / "01_MorphableModel.mat"
        if model_path.exists():
            print(f"✓ BFM model cached at: {model_path}")
            print(f"  File size: {model_path.stat().st_size / (1024**2):.2f} MB")
            sys.exit(0)
        else:
            print(f"✗ BFM model not cached")
            print(f"  Expected location: {model_path}")
            print(f"\nTo download: python -m data_utils.bfm_model.download_bfm")
            sys.exit(1)

    try:
        model_path = download_bfm_model(
            force_download=args.force,
            cache_dir=args.cache_dir
        )
        print(f"\nBFM model ready at: {model_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
