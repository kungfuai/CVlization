#!/usr/bin/env python3
"""
Download and cache FastAvatar multi-view datasets.

This script downloads multi-view datasets from the FastAvatar repository
and caches them locally to avoid repeated downloads.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# Cache directory (can be mounted from host for persistence)
CACHE_DIR = Path(os.environ.get("FASTAVATAR_CACHE", "~/.cache/fastavatar")).expanduser()
DATASETS_CACHE = CACHE_DIR / "datasets"


# Available datasets from FastAvatar repository
AVAILABLE_DATASETS = {
    "422": {
        "description": "Subject 422 - 16 multi-view images with FLAME parameters",
        "files": [
            "https://github.com/hliang2/FastAvatar/raw/main/data/422.zip",
        ],
        "type": "zip"
    },
    # Add more datasets as needed
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
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


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """
    Extract zip or tar archive.

    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.bz2', '.xz']:
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def download_dataset(
    dataset_id: str,
    force_download: bool = False,
    cache_dir: Optional[Path] = None
) -> Path:
    """
    Download and cache a FastAvatar multi-view dataset.

    Args:
        dataset_id: Dataset identifier (e.g., "422")
        force_download: If True, re-download even if cached
        cache_dir: Custom cache directory (default: ~/.cache/fastavatar/datasets)

    Returns:
        Path to the cached dataset directory

    Raises:
        ValueError: If dataset_id is not available
    """
    if dataset_id not in AVAILABLE_DATASETS:
        available = ", ".join(AVAILABLE_DATASETS.keys())
        raise ValueError(f"Dataset '{dataset_id}' not available. Choose from: {available}")

    # Set up cache directory
    if cache_dir is None:
        cache_dir = DATASETS_CACHE
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache = cache_dir / dataset_id

    # Check if already cached
    if dataset_cache.exists() and not force_download:
        print(f"Dataset '{dataset_id}' already cached at: {dataset_cache}")
        return dataset_cache

    print(f"Downloading dataset '{dataset_id}'...")
    dataset_info = AVAILABLE_DATASETS[dataset_id]

    # Download files
    for url in dataset_info["files"]:
        filename = url.split('/')[-1]
        download_path = cache_dir / f"{dataset_id}_{filename}"

        print(f"Downloading from: {url}")
        download_file(url, download_path, desc=f"Downloading {filename}")

        # Extract if it's an archive
        if dataset_info.get("type") in ["zip", "tar"]:
            print(f"Extracting {filename}...")
            extract_archive(download_path, cache_dir)

            # Clean up archive file
            download_path.unlink()

    if not dataset_cache.exists():
        raise RuntimeError(f"Download completed but dataset not found at: {dataset_cache}")

    print(f"\nDataset '{dataset_id}' successfully cached at: {dataset_cache}")
    return dataset_cache


def list_datasets() -> None:
    """List all available datasets."""
    print("\nAvailable FastAvatar Datasets:")
    print("=" * 60)
    for dataset_id, info in AVAILABLE_DATASETS.items():
        cached = "✓ CACHED" if (DATASETS_CACHE / dataset_id).exists() else "✗ Not cached"
        print(f"\n{dataset_id}: {info['description']}")
        print(f"  Status: {cached}")
        print(f"  Source: {info['files'][0]}")
    print("\n" + "=" * 60)


def main():
    """Command-line interface for downloading datasets."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and cache FastAvatar multi-view datasets"
    )
    parser.add_argument(
        "dataset_id",
        nargs="?",
        help="Dataset to download (e.g., '422'). Use --list to see available datasets."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DATASETS_CACHE,
        help=f"Custom cache directory (default: {DATASETS_CACHE})"
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset_id:
        parser.print_help()
        print("\nUse --list to see available datasets")
        sys.exit(1)

    try:
        dataset_path = download_dataset(
            args.dataset_id,
            force_download=args.force,
            cache_dir=args.cache_dir
        )
        print(f"\nDataset ready at: {dataset_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
