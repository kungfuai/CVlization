#!/usr/bin/env python3
"""
Prepare DeepScoresV2 dense subset for HuggingFace Hub.

Downloads ds2_dense.tar.gz from Zenodo, converts COCO annotations to the
standard HuggingFace object-detection format (image + objects dict), and
pushes to zzsi/deep-scores-v2-dense.

Usage:
    python prepare_deep_scores_v2_dense.py --inspect
    python prepare_deep_scores_v2_dense.py
    python prepare_deep_scores_v2_dense.py --push-to-hub zzsi/deep-scores-v2-dense
"""

import argparse
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/4012193/files/ds2_dense.tar.gz"
CACHE_DIR = Path.home() / ".cache" / "deep_scores_v2_dense"
TAR_PATH = CACHE_DIR / "ds2_dense.tar.gz"
EXTRACT_DIR = CACHE_DIR / "extracted"


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

def inspect(extract_dir: Path) -> None:
    """Print top-level structure and a sample annotation entry."""
    print(f"\n=== Directory structure under {extract_dir} ===")
    for p in sorted(extract_dir.rglob("*"))[:60]:
        indent = "  " * (len(p.relative_to(extract_dir).parts) - 1)
        size = f"  ({p.stat().st_size:,} bytes)" if p.is_file() else ""
        print(f"{indent}{p.name}{size}")

    jsons = list(extract_dir.rglob("*.json"))
    if jsons:
        print(f"\n=== Sample from {jsons[0].name} ===")
        with open(jsons[0]) as f:
            data = json.load(f)
        for key in data:
            val = data[key]
            sample = val[:2] if isinstance(val, list) else val
            print(f"  {key}: ({len(val)} items) {sample}")


# ---------------------------------------------------------------------------
# Build HuggingFace dataset
# ---------------------------------------------------------------------------

def find_images_dir(extract_dir: Path) -> Path:
    """Locate the images directory inside the extracted archive."""
    for p in extract_dir.rglob("*.png"):
        return p.parent
    raise FileNotFoundError(f"Could not locate images directory in {extract_dir}")


def find_annotation_jsons(extract_dir: Path) -> dict[str, Path]:
    """Return {split: path} for each COCO annotation JSON found."""
    splits = {}
    for json_path in sorted(extract_dir.rglob("*.json")):
        name = json_path.stem.lower()
        if "train" in name:
            splits["train"] = json_path
        elif "test" in name or "val" in name:
            splits["test"] = json_path
    return splits


def parse_categories(raw: dict | list) -> dict[int, dict]:
    """
    DeepScoresV2 uses a dict of string-keyed categories rather than a list.
    Returns {int_id: {"name": str, "annotation_set": str}} regardless of input format.
    """
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    return {cat["id"]: cat for cat in raw}


def font_from_filename(file_name: str) -> str:
    """Extract font name from DeepScoresV2 filename: lg-...-aug-{font}--page-N.png"""
    import re
    m = re.search(r"-aug-([^-]+)--", file_name)
    return m.group(1) if m else "unknown"


def build_split(coco_json: Path, images_dir: Path, split: str,
                annotation_set: str = "deepscores"):
    """
    Convert one DeepScoresV2 JSON into a list of row dicts.

    annotation_set: "deepscores" (default) or "muscima++"
    Each image gets one row; objects are filtered to the chosen annotation set.
    An extra `font` column is extracted from the filename.
    """
    print(f"  Loading {coco_json.name} ({coco_json.stat().st_size // 1_000_000} MB) ...")
    with open(coco_json) as f:
        coco = json.load(f)

    # Categories: dict with string keys in DeepScoresV2
    all_cats = parse_categories(coco["categories"])
    # Keep only the requested annotation set
    cats = {cid: info for cid, info in all_cats.items()
            if info.get("annotation_set", "deepscores") == annotation_set}
    cat_id_to_name = {cid: info["name"] for cid, info in cats.items()}

    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_id_to_name:
            anns_by_image[ann["image_id"]].append(ann)

    rows = []
    missing = 0
    for img_info in tqdm(coco["images"], desc=f"  {split}"):
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            img_path = images_dir / Path(img_info["file_name"]).name
        if not img_path.exists():
            missing += 1
            continue

        anns = anns_by_image[img_info["id"]]
        rows.append({
            "image_id":       img_info["id"],
            "file_name":      img_info["file_name"],
            "font":           font_from_filename(img_info["file_name"]),
            "image":          str(img_path),
            "width":          img_info["width"],
            "height":         img_info["height"],
            "annotation_set": annotation_set,
            "objects": {
                "id":          [a["id"] for a in anns],
                "bbox":        [a["bbox"] for a in anns],  # [x, y, w, h] COCO
                "category_id": [a["category_id"] for a in anns],
                "category":    [cat_id_to_name[a["category_id"]] for a in anns],
                "area":        [float(a.get("area", 0)) for a in anns],
                "iscrowd":     [int(a.get("iscrowd", 0)) for a in anns],
            },
        })

    if missing:
        print(f"  WARNING: {missing} images not found on disk")
    print(f"  {len(rows)} rows, {sum(len(r['objects']['id']) for r in rows):,} annotations")
    return rows


def build_dataset(extract_dir: Path, annotation_set: str = "deepscores"):
    from datasets import Dataset, DatasetDict, Image as HFImage

    images_dir = find_images_dir(extract_dir)
    split_jsons = find_annotation_jsons(extract_dir)
    print(f"Images dir      : {images_dir}")
    print(f"Splits          : {list(split_jsons.keys())}")
    print(f"Annotation set  : {annotation_set}")

    split_datasets = {}
    for split, json_path in split_jsons.items():
        print(f"\nBuilding split: {split}  ({json_path.name})")
        rows = build_split(json_path, images_dir, split, annotation_set)
        ds = Dataset.from_list(rows)
        ds = ds.cast_column("image", HFImage())
        split_datasets[split] = ds

    return DatasetDict(split_datasets)


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

DATASET_CARD = """\
---
license: cc-by-4.0
task_categories:
  - object-detection
language:
  - en
tags:
  - music
  - optical-music-recognition
  - omr
  - sheet-music
  - coco
  - symbol-detection
size_categories:
  - 1K<n<10K
---

# DeepScoresV2 — Dense Subset

A HuggingFace-formatted mirror of the **dense** subset of the
[DeepScoresV2](https://zenodo.org/records/4012193) dataset for music object detection.

## Dataset description

DeepScoresV2 is a large-scale dataset of synthetically rendered music score pages
annotated with bounding boxes for musical symbols. The **dense** subset contains
**1,714 images** selected by the authors as the most diverse and representative
sample from the full 803k-image dataset.

Each image is a full score page rendered from MuseScore. Annotations follow
COCO format: `bbox` is `[x, y, width, height]` in pixel coordinates.

## Format

```python
{
    "image_id":  int,
    "file_name": str,
    "image":     PIL.Image,     # full score page
    "width":     int,
    "height":    int,
    "objects": {
        "id":          List[int],
        "bbox":        List[List[float]],  # [x, y, w, h], COCO format
        "category_id": List[int],
        "category":    List[str],          # symbol class name
        "area":        List[float],
        "iscrowd":     List[int],
    },
}
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("zzsi/deep-scores-v2-dense")
example = ds["train"][0]
print(example["objects"]["category"][:5])
example["image"].show()
```

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## Attribution

This dataset is a reformatted mirror of DeepScoresV2. Please cite the original work:

```bibtex
@inproceedings{DeepScoresV2,
  title     = {DeepScoresV2: A Dataset for Music Object Detection with a Challenging Test Set},
  author    = {Tuggener, Lukas and Satyawan, Yvan Putra and Pacha, Alexander
               and Schmidhuber, J{\\"u}rgen and Stadelmann, Thilo},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2021}
}
```

Original dataset: <https://zenodo.org/records/4012193>
Original authors: Lukas Tuggener, Yvan Putra Satyawan, Alexander Pacha,
Jürgen Schmidhuber, Thilo Stadelmann (ZHAW / IDSIA)
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DeepScoresV2 dense for HuggingFace Hub"
    )
    parser.add_argument("--inspect", action="store_true",
        help="Download + extract, then print structure and exit")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
        help="Push to HuggingFace Hub repo (e.g. zzsi/deep-scores-v2-dense)")
    parser.add_argument("--output", default=None,
        help="Save dataset locally to this directory")
    parser.add_argument("--annotation-set", default="deepscores",
        choices=["deepscores", "muscima++"],
        help="Which annotation set to include (default: deepscores)")
    args = parser.parse_args()

    download(ZENODO_URL, TAR_PATH)
    extract(TAR_PATH, EXTRACT_DIR)

    if args.inspect:
        inspect(EXTRACT_DIR)
        return

    print("\nBuilding HuggingFace dataset ...")
    dd = build_dataset(EXTRACT_DIR, annotation_set=args.annotation_set)
    print(f"\nDataset:\n{dd}")

    if args.output:
        out = Path(args.output)
        print(f"\nSaving to {out} ...")
        dd.save_to_disk(str(out))

    if args.push_to_hub:
        repo_id = args.push_to_hub
        print(f"\nPushing to {repo_id} ...")
        dd.push_to_hub(repo_id, private=False)

        # Upload dataset card
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
