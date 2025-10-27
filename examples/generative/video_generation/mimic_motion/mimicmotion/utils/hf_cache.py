import os
from typing import Optional

from huggingface_hub import hf_hub_download

HF_URI_PREFIX = "hf://"


def _expand(path: str) -> str:
    return os.path.expanduser(path)


def resolve_asset_path(path: str, *, env_override: Optional[str] = None) -> str:
    """
    Resolve a local path or Hugging Face URI into a concrete file path.

    - Accepts paths prefixed with ``hf://{repo_id}/{filename}`` and downloads them
      into the user's Hugging Face cache (``$HF_HOME`` or ``~/.cache/huggingface``).
    - Optional ``env_override`` allows callers to specify an environment variable
      whose value takes precedence when present (useful for runtime overrides).
    """
    candidate = os.environ.get(env_override) if env_override else None
    if candidate:
        path = candidate

    path = _expand(path)
    if os.path.exists(path):
        return path

    if path.startswith(HF_URI_PREFIX):
        repo_and_file = path[len(HF_URI_PREFIX) :]
        parts = repo_and_file.split("/", 2)
        if len(parts) < 3:
            raise ValueError(
                f"Invalid Hugging Face URI '{path}'. Expected format hf://<namespace>/<repo>/<filename>."
            )
        repo_id = "/".join(parts[:2])
        filename = parts[2]
        return hf_hub_download(repo_id=repo_id, filename=filename)

    return path
