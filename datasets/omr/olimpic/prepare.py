#!/usr/bin/env python3
"""
Prepare OLiMPiC (OpenScore Lieder Linearized MusicXML Piano Corpus) for HuggingFace Hub.

Downloads synthetic and/or scanned variants from GitHub Releases, converts to
HuggingFace image-to-text format, and pushes to Hub.

Structure inside tarballs:
  samples/{score_id}/p{page}-s{system}.png    ← system-level crop
  samples/{score_id}/p{page}-s{system}.lmx    ← linearized MusicXML tokens
  samples/{score_id}/p{page}-s{system}.musicxml
  samples.{split}.txt                          ← manifest (one stem per line)

Usage:
    python prepare.py --inspect
    python prepare.py --variant synthetic --push-to-hub zzsi/olimpic
    python prepare.py --variant scanned   --push-to-hub zzsi/olimpic-scanned
    python prepare.py --variant both      --push-to-hub zzsi/olimpic
"""

import argparse
import tarfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://github.com/ufal/olimpic-icdar24/releases/download/datasets"
TARBALLS = {
    "synthetic": "olimpic-1.0-synthetic.2024-02-12.tar.gz",
    "scanned":   "olimpic-1.0-scanned.2024-02-12.tar.gz",
}
# splits available per variant
SPLITS = {
    "synthetic": ["train", "dev", "test"],
    "scanned":   ["dev", "test"],
}
CACHE_DIR = Path.home() / ".cache" / "olimpic"


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def extract(tar_path: Path, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"Already extracted: {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def inspect(extract_dir: Path, variant: str) -> None:
    root = extract_dir / f"olimpic-1.0-{variant}"
    print(f"\n=== {root} ===")
    for p in sorted(root.iterdir()):
        size = f"  ({p.stat().st_size:,} bytes)" if p.is_file() else ""
        print(f"  {p.name}{size}")

    # Show a sample LMX
    manifest = root / "samples.dev.txt"
    if not manifest.exists():
        manifest = root / "samples.train.txt"
    stem = manifest.read_text().strip().splitlines()[0]
    lmx_path = root / f"{stem}.lmx"
    if lmx_path.exists():
        print(f"\n=== Sample LMX ({stem}.lmx) ===")
        print(lmx_path.read_text()[:500])


# ---------------------------------------------------------------------------
# Build HuggingFace dataset
# ---------------------------------------------------------------------------

def build_split(root: Path, split: str, source: str) -> list:
    manifest = root / f"samples.{split}.txt"
    stems = manifest.read_text().strip().splitlines()

    rows = []
    missing = 0
    for stem in tqdm(stems, desc=f"  {source}/{split}"):
        png = root / f"{stem}.png"
        lmx = root / f"{stem}.lmx"
        mxml = root / f"{stem}.musicxml"
        if not png.exists() or not lmx.exists():
            missing += 1
            continue
        # stem is like "samples/6547240/p4-s4"
        parts = Path(stem).parts   # ('samples', '6547240', 'p4-s4')
        score_id = parts[1] if len(parts) >= 2 else ""
        page_system = parts[2] if len(parts) >= 3 else ""
        rows.append({
            "image":       str(png),
            "lmx":         lmx.read_text().strip(),
            "musicxml":    mxml.read_text().strip() if mxml.exists() else "",
            "score_id":    score_id,
            "page_system": page_system,   # e.g. "p4-s4"
            "source":      source,        # "synthetic" or "scanned"
            "split":       split,
        })
    if missing:
        print(f"  WARNING: {missing} missing png/lmx pairs")
    print(f"  {source}/{split}: {len(rows)} rows")
    return rows


def build_dataset(extract_dir: Path, variants: list[str]):
    from datasets import Dataset, DatasetDict, Image as HFImage

    split_rows: dict[str, list] = {}

    for variant in variants:
        root = extract_dir / variant / f"olimpic-1.0-{variant}"
        for split in SPLITS[variant]:
            rows = build_split(root, split, source=variant)
            if split not in split_rows:
                split_rows[split] = []
            split_rows[split].extend(rows)

    split_datasets = {}
    for split, rows in split_rows.items():
        ds = Dataset.from_list(rows)
        ds = ds.cast_column("image", HFImage())
        split_datasets[split] = ds
        print(f"  {split}: {len(ds)} rows")

    return DatasetDict(split_datasets)


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

DATASET_CARD = """\
---
license: cc-by-sa-4.0
task_categories:
  - image-to-text
language:
  - en
tags:
  - music
  - optical-music-recognition
  - omr
  - sheet-music
  - musicxml
  - piano
size_categories:
  - 10K<n<100K
---

# OLiMPiC — OpenScore Lieder Linearized MusicXML Piano Corpus

A HuggingFace-formatted mirror of the [OLiMPiC dataset](https://github.com/ufal/olimpic-icdar24)
for end-to-end optical music recognition of pianoform music.

## Dataset description

OLiMPiC provides system-level (one staff row) crops of piano scores paired with
ground-truth annotations in Linearized MusicXML (LMX) format. Each sample is one
system — the smallest unit that makes musical sense for training sequence models.

- **Synthetic variant**: 17,945 rendered systems (train/dev/test)
- **Scanned variant**: ~2,900 real IMSLP scans (dev/test only)
- **Source**: OpenScore Lieder corpus (1,356 manually verified scores)

## Format

```python
{
    "image":       PIL.Image,   # system-level crop (one row of grand staff)
    "lmx":         str,         # Linearized MusicXML token sequence
    "musicxml":    str,         # full MusicXML for this system
    "score_id":    str,         # OpenScore score identifier
    "page_system": str,         # e.g. "p2-s3" (page 2, system 3)
    "source":      str,         # "synthetic" or "scanned"
    "split":       str,         # "train", "dev", or "test"
}
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("zzsi/olimpic")
example = ds["train"][0]
print(example["lmx"][:200])
example["image"].show()
```

## License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Attribution

Please cite the original work:

```bibtex
@inproceedings{OLiMPiC,
  title     = {Practical End-to-End Optical Music Recognition for Pianoform Music},
  author    = {Fier, Jiří and Hajič, Jan},
  booktitle = {International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2024}
}
```

Original dataset: <https://github.com/ufal/olimpic-icdar24>
Original authors: Jiří Fier, Jan Hajič (UFAL, Charles University)
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare OLiMPiC dataset for HuggingFace Hub"
    )
    parser.add_argument("--variant", default="synthetic",
        choices=["synthetic", "scanned", "both"],
        help="Which variant to process (default: synthetic)")
    parser.add_argument("--inspect", action="store_true",
        help="Download + extract, then print structure and exit")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
        help="Push to HuggingFace Hub repo")
    parser.add_argument("--output", default=None,
        help="Save dataset locally to this directory")
    args = parser.parse_args()

    variants = ["synthetic", "scanned"] if args.variant == "both" else [args.variant]

    # Download and extract each variant
    for variant in variants:
        tar_name = TARBALLS[variant]
        tar_path = CACHE_DIR / tar_name
        extract_dir = CACHE_DIR / variant
        download(f"{BASE_URL}/{tar_name}", tar_path)
        extract(tar_path, extract_dir)

    if args.inspect:
        for variant in variants:
            inspect(CACHE_DIR / variant, variant)
        return

    print("\nBuilding HuggingFace dataset ...")
    dd = build_dataset(CACHE_DIR, variants)
    print(f"\nDataset:\n{dd}")

    if args.output:
        out = Path(args.output)
        print(f"\nSaving to {out} ...")
        dd.save_to_disk(str(out))

    if args.push_to_hub:
        repo_id = args.push_to_hub
        print(f"\nPushing to {repo_id} ...")
        dd.push_to_hub(repo_id, private=False)

        from huggingface_hub import HfApi
        api = HfApi()
        card_path = CACHE_DIR / "README.md"
        card_path.write_text(DATASET_CARD)
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Done — https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
