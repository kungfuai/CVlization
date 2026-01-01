# -*- coding: utf-8 -*-
"""Lazy model download for VACE annotators and inference models.

Uses huggingface_hub for centralized caching at ~/.cache/huggingface/hub.
Models are downloaded on-demand when first needed.
"""

import os
from pathlib import Path


# HuggingFace repos for VACE models
VACE_ANNOTATORS_REPO = "ali-vilab/VACE-Annotators"
VACE_WAN_REPO = "Wan-AI/Wan2.1-VACE-1.3B"
VACE_LTX_REPO = "ali-vilab/VACE-LTX-Video-0.9"


def get_cache_dir():
    """Get the HuggingFace cache directory."""
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def download_vace_annotators(cache_dir: str = None) -> str:
    """Download VACE-Annotators models if not present.

    Returns the local path to the downloaded models.
    """
    from huggingface_hub import snapshot_download

    if cache_dir is None:
        cache_dir = get_cache_dir()

    print(f"Ensuring {VACE_ANNOTATORS_REPO} is downloaded...")
    local_dir = snapshot_download(
        repo_id=VACE_ANNOTATORS_REPO,
        cache_dir=cache_dir,
    )
    print(f"VACE-Annotators ready: {local_dir}")
    return local_dir


def download_vace_wan(cache_dir: str = None) -> str:
    """Download VACE-Wan inference model if not present."""
    from huggingface_hub import snapshot_download

    if cache_dir is None:
        cache_dir = get_cache_dir()

    print(f"Ensuring {VACE_WAN_REPO} is downloaded...")
    local_dir = snapshot_download(
        repo_id=VACE_WAN_REPO,
        cache_dir=cache_dir,
    )
    print(f"VACE-Wan model ready: {local_dir}")
    return local_dir


def download_vace_ltx(cache_dir: str = None) -> str:
    """Download VACE-LTX inference model if not present."""
    from huggingface_hub import snapshot_download

    if cache_dir is None:
        cache_dir = get_cache_dir()

    print(f"Ensuring {VACE_LTX_REPO} is downloaded...")
    local_dir = snapshot_download(
        repo_id=VACE_LTX_REPO,
        cache_dir=cache_dir,
    )
    print(f"VACE-LTX model ready: {local_dir}")
    return local_dir


def ensure_annotator_models(task: str = None) -> str:
    """Ensure annotator models are downloaded and return the base path.

    Args:
        task: Optional task name to download only required models.
              If None, downloads all annotator models.

    Returns:
        Path to the VACE-Annotators directory.
    """
    return download_vace_annotators()


def ensure_inference_models(base: str = "wan") -> str:
    """Ensure inference models are downloaded and return the checkpoint path.

    Args:
        base: Either "wan" or "ltx" for the base model.

    Returns:
        Path to the model checkpoint directory.
    """
    if base == "wan":
        return download_vace_wan()
    elif base == "ltx":
        return download_vace_ltx()
    else:
        raise ValueError(f"Unknown base model: {base}. Use 'wan' or 'ltx'.")


def get_annotator_model_path(annotators_dir: str, subpath: str) -> str:
    """Get the full path to an annotator model file.

    Args:
        annotators_dir: Base path to VACE-Annotators
        subpath: Relative path within annotators (e.g., "salient/u2net.pt")

    Returns:
        Full path to the model file.
    """
    return os.path.join(annotators_dir, subpath)


def patch_config_paths(config: dict, annotators_dir: str) -> dict:
    """Patch a config dict to use the downloaded annotators path.

    Recursively replaces "models/VACE-Annotators/" prefixes with the
    actual downloaded path.

    Args:
        config: Config dictionary with model paths
        annotators_dir: Actual path to downloaded VACE-Annotators

    Returns:
        Patched config dictionary.
    """
    import copy
    config = copy.deepcopy(config)

    def _patch(obj):
        if isinstance(obj, dict):
            return {k: _patch(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_patch(v) for v in obj]
        elif isinstance(obj, str) and "models/VACE-Annotators" in obj:
            return obj.replace("models/VACE-Annotators", annotators_dir)
        else:
            return obj

    return _patch(config)


if __name__ == "__main__":
    # Test download
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", action="store_true", help="Download annotator models")
    parser.add_argument("--wan", action="store_true", help="Download Wan inference model")
    parser.add_argument("--ltx", action="store_true", help="Download LTX inference model")
    parser.add_argument("--all", action="store_true", help="Download all models")
    args = parser.parse_args()

    if args.all or args.annotators:
        download_vace_annotators()
    if args.all or args.wan:
        download_vace_wan()
    if args.all or args.ltx:
        download_vace_ltx()

    if not any([args.all, args.annotators, args.wan, args.ltx]):
        print("Usage: python model_download.py [--annotators] [--wan] [--ltx] [--all]")
