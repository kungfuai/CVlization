from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPECS = (
    {
        "name": "llm",
        "env": "LLAMACPP_LLM_PATH",
        "default": MODEL_DIR / "phi-2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf?download=1",
        "sha256": "324356668fa5ba9f4135de348447bb2bbe2467eaa1b8fcfb53719de62fbd2499",
    },
    {
        "name": "embed",
        "env": "LLAMACPP_EMBED_PATH",
        "default": MODEL_DIR / "nomic-embed-text-v1.5.f16.gguf",
        "url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf?download=1",
        "sha256": "f7af6f66802f4df86eda10fe9bbcfc75c39562bed48ef6ace719a251cf1c2fdb",
    },
)


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with dest.open("wb") as fp, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dest.name}",
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fp.write(chunk)
                    bar.update(len(chunk))


def ensure_models(specs: Iterable[dict] = SPECS) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for spec in specs:
        env_value = os.getenv(spec["env"])
        if env_value:
            path = Path(env_value).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Environment override {spec['env']} points to missing file: {path}")
            resolved[spec["name"]] = path
            continue

        dest = Path(spec["default"]).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            digest = _hash_file(dest)
            if digest == spec["sha256"]:
                resolved[spec["name"]] = dest
                continue
            dest.unlink()

        download_allowed = os.getenv("LLAMACPP_DOWNLOAD_MODELS", "1").lower() not in {"0", "false", "no"}
        if not download_allowed:
            raise FileNotFoundError(
                f"Model {spec['name']} missing at {dest}. Enable downloads by setting LLAMACPP_DOWNLOAD_MODELS=1"
            )

        _download(spec["url"], dest)
        digest = _hash_file(dest)
        if digest != spec["sha256"]:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: expected {spec['sha256']}, got {digest}"
            )

        resolved[spec["name"]] = dest
    return resolved


if __name__ == "__main__":
    paths = ensure_models()
    for name, path in paths.items():
        print(f"{name}: {path}")
