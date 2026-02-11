"""Pretrained model weight management."""
import os
from pathlib import Path
from typing import Optional


# Known pretrained model paths (relative to cache)
PRETRAINED_MODELS = {
    "lama": "checkpoints/lama_best.pt",
    "elir": "checkpoints/elir_best.pt",
}


def get_pretrained_path(model_name: str, download: bool = False) -> Optional[str]:
    """
    Get path to pretrained weights for a model.

    Args:
        model_name: Model identifier ("lama", "elir")
        download: If True, attempt to download if not found locally

    Returns:
        Path string if found, None otherwise
    """
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "cvlization" / "pretrained"

    rel_path = PRETRAINED_MODELS.get(model_name)
    if rel_path is None:
        print(f"Unknown pretrained model: {model_name}")
        return None

    local_path = cache_dir / rel_path
    if local_path.exists():
        return str(local_path)

    # Also check workspace checkpoints
    workspace_path = Path("checkpoints")
    for pt in workspace_path.glob(f"{model_name}*.pt"):
        return str(pt)

    if download:
        print(f"Pretrained weights for '{model_name}' not found locally.")
        print(f"  Looked in: {local_path}")
        print(f"  Train from scratch or provide --pretrained <path>")

    return None
