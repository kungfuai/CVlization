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
            if isinstance(val, list):
                sample = val[:2]
                print(f"  {key}: (list, {len(val)} items) {sample}")
            elif isinstance(val, dict):
                first_keys = list(val.keys())[:2]
                print(f"  {key}: (dict, {len(val)} items) first keys={first_keys}")
            else:
                print(f"  {key}: {val}")


# ---------------------------------------------------------------------------
# Build HuggingFace dataset
# ---------------------------------------------------------------------------

def find_images_dir(extract_dir: Path) -> Path:
    """Locate the 'images' directory inside the extracted archive."""
    candidate = extract_dir / "ds2_dense" / "images"
    if candidate.is_dir():
        return candidate
    # Fallback: first directory containing PNGs without '_seg' in name
    for p in sorted(extract_dir.rglob("*.png")):
        if "_seg" not in p.name:
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


def a_bbox_to_coco(a_bbox: list) -> list:
    """Convert DeepScoresV2 a_bbox [x1, y1, x2, y2] to COCO [x, y, w, h]."""
    x1, y1, x2, y2 = a_bbox
    return [x1, y1, x2 - x1, y2 - y1]


def build_split(coco_json: Path, images_dir: Path, split: str,
                annotation_set: str = "deepscores"):
    """
    Convert one DeepScoresV2 JSON into a list of row dicts.

    DeepScoresV2 format differences from standard COCO:
    - images use 'filename' (not 'file_name') and have 'ann_ids' list
    - annotations is a dict keyed by string ann_id
    - each annotation has 'a_bbox' [x1,y1,x2,y2] instead of 'bbox' [x,y,w,h]
    - 'cat_id' is a list of string IDs (one per annotation set)
    - 'img_id' is a string

    annotation_set: "deepscores" (default) or "muscima++"
    Each image gets one row; objects are filtered to the chosen annotation set.
    An extra `font` column is extracted from the filename.
    """
    print(f"  Loading {coco_json.name} ({coco_json.stat().st_size // 1_000_000} MB) ...")
    with open(coco_json) as f:
        coco = json.load(f)

    # Categories: dict with string keys in DeepScoresV2
    all_cats = parse_categories(coco["categories"])
    # Keep only the requested annotation set; build str_id → name map
    cat_id_to_name: dict[str, str] = {
        str(cid): info["name"]
        for cid, info in all_cats.items()
        if info.get("annotation_set", "deepscores") == annotation_set
    }

    # annotations is a dict: {str_ann_id: {...}}
    all_anns: dict = coco["annotations"]

    rows = []
    missing = 0
    for img_info in tqdm(coco["images"], desc=f"  {split}"):
        filename = img_info.get("filename") or img_info.get("file_name", "")
        img_path = images_dir / filename
        if not img_path.exists():
            img_path = images_dir / Path(filename).name
        if not img_path.exists():
            missing += 1
            continue

        # Collect annotations for this image
        ann_ids = img_info.get("ann_ids", [])
        objs = {"id": [], "bbox": [], "category_id": [], "category": [], "area": []}
        for ann_id in ann_ids:
            ann = all_anns.get(str(ann_id))
            if ann is None:
                continue
            # cat_id is a list; pick the one matching our annotation_set
            matched_cat_id = None
            for cid in ann.get("cat_id", []):
                if str(cid) in cat_id_to_name:
                    matched_cat_id = str(cid)
                    break
            if matched_cat_id is None:
                continue
            objs["id"].append(int(ann_id))
            objs["bbox"].append(a_bbox_to_coco(ann["a_bbox"]))  # [x, y, w, h]
            objs["category_id"].append(int(matched_cat_id))
            objs["category"].append(cat_id_to_name[matched_cat_id])
            objs["area"].append(float(ann.get("area", 0)))

        rows.append({
            "image_id":       img_info["id"],
            "file_name":      filename,
            "font":           font_from_filename(filename),
            "image":          str(img_path),
            "width":          img_info["width"],
            "height":         img_info["height"],
            "annotation_set": annotation_set,
            "objects":        objs,
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
