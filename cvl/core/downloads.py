"""Download management for CVL examples.

Handles downloading models, datasets, and other resources from various sources:
- Google Drive (via gdown)
- HuggingFace Hub
- HTTP/HTTPS URLs
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


class DownloadError(Exception):
    """Raised when a download fails."""
    pass


def parse_url(url: str) -> Tuple[str, Dict]:
    """Parse a download URL and extract source type and metadata.

    Supported formats:
    - gdrive://FILE_ID or https://drive.google.com/...
    - hf://repo/file or https://huggingface.co/...
    - http://... or https://...

    Args:
        url: Download URL

    Returns:
        Tuple of (source_type, metadata_dict)
        source_type is one of: "gdrive", "hf", "http"
    """
    if url.startswith("gdrive://"):
        file_id = url.replace("gdrive://", "")
        return ("gdrive", {"file_id": file_id})

    elif "drive.google.com" in url:
        # Extract file ID from Google Drive URL
        # Formats: https://drive.google.com/uc?id=FILE_ID
        #          https://drive.google.com/file/d/FILE_ID/view
        if "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
        elif "/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
        else:
            raise DownloadError(f"Cannot parse Google Drive URL: {url}")
        return ("gdrive", {"file_id": file_id, "original_url": url})

    elif url.startswith("hf://"):
        # Format: hf://org/repo/path/to/file or hf://repo/path/to/file
        # HuggingFace repos are either "username/repo" or "org/repo"
        # Split into max 3 parts to handle org/repo/file
        path = url.replace("hf://", "")
        parts = path.split("/", 2)
        if len(parts) < 3:
            raise DownloadError(f"Invalid HuggingFace URL format (expected hf://org/repo/file): {url}")
        # First two parts are repo_id, rest is filename
        return ("hf", {"repo_id": f"{parts[0]}/{parts[1]}", "filename": parts[2]})

    elif "huggingface.co" in url:
        # Parse HF URL: https://huggingface.co/repo/resolve/main/file.safetensors
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 4 and path_parts[2] == "resolve":
            repo_id = "/".join(path_parts[:2])
            filename = "/".join(path_parts[4:])
            return ("hf", {"repo_id": repo_id, "filename": filename, "original_url": url})
        return ("http", {"url": url})

    elif url.startswith("http://") or url.startswith("https://"):
        return ("http", {"url": url})

    else:
        raise DownloadError(f"Unsupported URL format: {url}")


def check_file_exists(dest: Path) -> bool:
    """Check if a file already exists.

    Args:
        dest: Destination file path

    Returns:
        True if file exists and has non-zero size
    """
    return dest.exists() and dest.stat().st_size > 0


def download_gdrive(file_id: str, dest: Path, quiet: bool = False) -> None:
    """Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID
        dest: Destination file path
        quiet: Suppress progress output

    Raises:
        DownloadError: If download fails
    """
    try:
        # Check if gdown is available
        result = subprocess.run(
            ["gdown", "--version"],
            capture_output=True,
            check=False
        )
        if result.returncode != 0:
            raise DownloadError(
                "gdown is not installed. Install with: pip install gdown"
            )

        # Create parent directory
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download to temp location first (atomic)
        temp_dest = dest.parent / f".{dest.name}.tmp"

        # Build gdown command
        cmd = ["gdown", "-O", str(temp_dest), f"https://drive.google.com/uc?id={file_id}"]
        if quiet:
            cmd.append("--quiet")

        # Run download
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            if temp_dest.exists():
                temp_dest.unlink()
            raise DownloadError(f"gdown failed with exit code {result.returncode}")

        # Move to final location (atomic)
        temp_dest.rename(dest)

    except subprocess.CalledProcessError as e:
        raise DownloadError(f"Failed to download from Google Drive: {e}")
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        raise DownloadError(f"Failed to download from Google Drive: {e}")


def download_hf(repo_id: str, filename: str, dest: Path, quiet: bool = False, token: Optional[str] = None) -> None:
    """Download a file from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "facebook/wav2vec2-base")
        filename: File path within repo (e.g., "pytorch_model.bin")
        dest: Destination file path
        quiet: Suppress progress output
        token: HuggingFace API token (optional, for gated models)

    Raises:
        DownloadError: If download fails
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError

        # Create parent directory
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download directly to destination
        # hf_hub_download handles temp files internally
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=dest.parent,
            local_dir_use_symlinks=False
        )

        # If downloaded to different location, move it
        if Path(downloaded_path) != dest:
            Path(downloaded_path).rename(dest)

    except ImportError:
        raise DownloadError(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        )
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            raise DownloadError(
                f"Authentication required for {repo_id}. "
                "This model requires a HuggingFace token. "
                "Set HF_TOKEN environment variable or login with: huggingface-cli login"
            )
        raise DownloadError(f"Failed to download from HuggingFace: {e}")
    except Exception as e:
        raise DownloadError(f"Failed to download from HuggingFace: {e}")


def download_http(url: str, dest: Path, quiet: bool = False) -> None:
    """Download a file from HTTP/HTTPS URL.

    Args:
        url: HTTP/HTTPS URL
        dest: Destination file path
        quiet: Suppress progress output

    Raises:
        DownloadError: If download fails
    """
    try:
        import requests
        from tqdm import tqdm

        # Create parent directory
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download to temp location first (atomic)
        temp_dest = dest.parent / f".{dest.name}.tmp"

        # Stream download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(temp_dest, 'wb') as f:
            if quiet or total_size == 0:
                # No progress bar
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            else:
                # Show progress bar
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Move to final location (atomic)
        temp_dest.rename(dest)

    except ImportError:
        raise DownloadError(
            "requests and tqdm are required for HTTP downloads. "
            "Install with: pip install requests tqdm"
        )
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        raise DownloadError(f"Failed to download from {url}: {e}")


def download_file(url: str, dest: Path, quiet: bool = False, token: Optional[str] = None) -> bool:
    """Download a file from any supported source.

    Automatically detects source type (Google Drive, HuggingFace, HTTP)
    and uses appropriate download method.

    Args:
        url: Download URL
        dest: Destination file path
        quiet: Suppress progress output
        token: API token (for HuggingFace gated models)

    Returns:
        True if file was downloaded, False if already existed

    Raises:
        DownloadError: If download fails
    """
    # Check if file already exists (idempotent)
    if check_file_exists(dest):
        if not quiet:
            print(f"✓ Already exists: {dest}")
        return False

    if not quiet:
        print(f"Downloading {dest.name}...")

    # Parse URL and download
    source_type, metadata = parse_url(url)

    if source_type == "gdrive":
        download_gdrive(metadata["file_id"], dest, quiet=quiet)
    elif source_type == "hf":
        download_hf(metadata["repo_id"], metadata["filename"], dest, quiet=quiet, token=token)
    elif source_type == "http":
        download_http(metadata["url"], dest, quiet=quiet)
    else:
        raise DownloadError(f"Unknown source type: {source_type}")

    if not quiet:
        print(f"✓ Downloaded: {dest}")

    return True


def get_cache_dir() -> Path:
    """Get the CVlization cache directory.

    Returns:
        Path to ~/.cache/cvlization/
    """
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(cache_home) / "cvlization"


def download_resources(downloads: List[Dict], base_path: Path, quiet: bool = False) -> Dict[str, bool]:
    """Download multiple resources from example.yaml downloads section.

    Args:
        downloads: List of download specifications from example.yaml
        base_path: Base path for relative destinations (example directory, used as fallback)
        quiet: Suppress progress output

    Returns:
        Dict mapping destination paths to download status (True if downloaded, False if existed)

    Raises:
        DownloadError: If any download fails
    """
    results = {}

    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")

    # Get cache directory
    cache_dir = get_cache_dir()

    for download_spec in downloads:
        url = download_spec.get("url")
        dest_rel = download_spec.get("dest")
        requires_auth = download_spec.get("requires_auth", False)

        if not url or not dest_rel:
            raise DownloadError(f"Invalid download spec: {download_spec}")

        # Resolve destination path
        # If dest starts with models/ or data/, use cache directory
        # Otherwise use example directory (for backward compatibility)
        if dest_rel.startswith("models/") or dest_rel.startswith("data/"):
            dest = cache_dir / dest_rel
        else:
            dest = base_path / dest_rel

        # Check for required authentication
        if requires_auth and not hf_token:
            raise DownloadError(
                f"Download requires HuggingFace authentication: {url}\n"
                "Set HF_TOKEN environment variable or run: huggingface-cli login"
            )

        # Download
        try:
            downloaded = download_file(url, dest, quiet=quiet, token=hf_token)
            results[str(dest)] = downloaded
        except DownloadError as e:
            # Re-raise with more context
            raise DownloadError(f"Failed to download {url} to {dest}: {e}")

    return results
