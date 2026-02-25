"""
Pretrained weight management for video artifact removal models.

Supports two sources:
- Official pretrained weights (auto-downloaded from GitHub releases)
- User-trained checkpoints (looked up in local cache/workspace)
"""

import os
from pathlib import Path
from typing import Optional

# Official pretrained weight URLs
PRETRAINED_URLS = {
    "lama": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    "elir": "https://github.com/eladc-git/ELIR/releases/download/v1.0/elir_bfr.pth",
}

# Expected filenames for official weights
PRETRAINED_FILENAMES = {
    "lama": "big-lama.pt",
    "elir": "elir_bfr.pth",
}

# User-trained checkpoint paths (relative to cache dir)
TRAINED_CHECKPOINTS = {
    "lama": "checkpoints/lama_best.pt",
    "elir": "checkpoints/elir_best.pt",
}


def get_cache_dir() -> Path:
    """Get the CVlization cache directory for models."""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "cvlization" / "models" / "video_enhancement"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, dest: Path, quiet: bool = False) -> bool:
    """Download a file from URL with progress bar.

    Returns:
        True if downloaded, False if already existed
    """
    if dest.exists() and dest.stat().st_size > 0:
        if not quiet:
            print(f"✓ Already cached: {dest}")
        return False

    if not quiet:
        print(f"Downloading {dest.name} from {url}...")

    try:
        import requests
        from tqdm import tqdm

        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest.parent / f".{dest.name}.tmp"

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(temp_dest, 'wb') as f:
            if quiet or total_size == 0:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        temp_dest.rename(dest)

        if not quiet:
            print(f"✓ Downloaded: {dest}")
        return True

    except ImportError:
        raise RuntimeError(
            "requests and tqdm are required for downloading. "
            "Install with: pip install requests tqdm"
        )
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        raise RuntimeError(f"Failed to download from {url}: {e}")


def get_pretrained_path(model_type: str, download: bool = True, quiet: bool = False) -> Optional[Path]:
    """Get path to pretrained weights.

    Lookup order:
    1. User-trained checkpoint in cache (lama_best.pt / elir_best.pt)
    2. User-trained checkpoint in local ./checkpoints/ directory
    3. Official pretrained weights in cache (downloaded from GitHub if needed)

    Args:
        model_type: Model type ("lama" or "elir")
        download: If True, download official weights when no local checkpoint found
        quiet: Suppress progress output

    Returns:
        Path to weights, or None if not available
    """
    if model_type not in PRETRAINED_URLS:
        return None

    cache_dir = get_cache_dir()

    # 1. Check for user-trained checkpoint in cache
    trained_rel = TRAINED_CHECKPOINTS.get(model_type)
    if trained_rel:
        trained_path = cache_dir.parent.parent / "pretrained" / trained_rel
        if trained_path.exists() and trained_path.stat().st_size > 0:
            if not quiet:
                print(f"Using user-trained checkpoint: {trained_path}")
            return trained_path

    # 2. Check for user-trained checkpoint in local workspace
    workspace_path = Path("checkpoints")
    for pt in workspace_path.glob(f"{model_type}*.pt") if workspace_path.exists() else []:
        if not quiet:
            print(f"Using workspace checkpoint: {pt}")
        return pt

    # 3. Official pretrained weights
    filename = PRETRAINED_FILENAMES[model_type]
    dest = cache_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        if not quiet:
            print(f"Using cached pretrained weights: {dest}")
        return dest

    if not download:
        return None

    url = PRETRAINED_URLS[model_type]
    download_file(url, dest, quiet=quiet)
    return dest


def ensure_pretrained(model_type: str) -> Path:
    """Ensure pretrained weights are available, downloading if needed.

    Raises:
        RuntimeError: If weights cannot be found or downloaded
    """
    path = get_pretrained_path(model_type, download=True)
    if path is None:
        raise RuntimeError(f"No pretrained weights available for model type: {model_type}")
    return path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pretrained.py <model_type>")
        print("  model_type: lama, elir")
        sys.exit(1)

    model_type = sys.argv[1]
    try:
        path = ensure_pretrained(model_type)
        print(f"\nPretrained weights ready at: {path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
