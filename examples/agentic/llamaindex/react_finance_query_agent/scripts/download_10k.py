from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES: Dict[str, str] = {
    "lyft_2021.pdf": "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf",
    "uber_2021.pdf": "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf",
}

# SHA256 digests taken from upstream repository to guard against corruption
SHA256: Dict[str, str] = {
    "lyft_2021.pdf": "f9962f78002a2758a66ce228137612f9e1d50209f5ffd05b401222c2369c0bbc",
    "uber_2021.pdf": "6727329fe30d2418ae3fa7ed644a5fcc14f691ad61b1ecc4cda6412cc4a175c2",
}


def download_file(name: str, url: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / name

    if target.exists() and verify_checksum(target, SHA256.get(name)):
        return target

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with target.open("wb") as fp, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {name}",
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fp.write(chunk)
                    bar.update(len(chunk))

    if not verify_checksum(target, SHA256.get(name)):
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {name}")

    return target


def verify_checksum(path: Path, expected: str | None) -> bool:
    if expected is None or not path.exists():
        return False
    sha = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest() == expected


def main() -> None:
    for name, url in FILES.items():
        path = download_file(name, url)
        print(f"Fetched {path.relative_to(Path.cwd()) if path.is_absolute() else path}")


if __name__ == "__main__":
    main()
