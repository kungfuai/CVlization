#!/usr/bin/env python3
"""
SROIE Dataset Builder for OCR Reading Benchmark

Downloads SROIE receipt dataset from HuggingFace and writes TAR shards
compatible with vllm_ocr_eval's TarShardDataset format.

Dataset used: arvindrajan92/sroie_document_understanding
  - 652 full receipt images with per-word OCR annotations
  - bbox format: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (4-point quad)
  - Only has 'train' split; we use it as our evaluation corpus

Usage:
    python dataset_builder.py --output-dir data/shards/
    python dataset_builder.py --output-dir data/shards/ --shard-size 100
"""

import argparse
import io
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Return the CVlization cache directory for SROIE data."""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(cache_home) / "cvlization" / "data" / "ocr_reading" / "sroie"


def quad_points_to_aabb(points) -> List[float]:
    """
    Convert a 4-point quadrilateral (either nested [[x0,y0],...] or flat [x0,y0,...])
    to an axis-aligned bounding box [min_x, min_y, max_x, max_y].
    """
    # Normalise to flat list of coordinates
    flat: List[float] = []
    for pt in points:
        if isinstance(pt, (list, tuple)):
            flat.extend(pt)
        else:
            flat.append(float(pt))
    xs = flat[0::2]
    ys = flat[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def build_annotation(example) -> dict:
    """
    Build the JSON annotation dict expected by vllm_ocr_eval.

    Supports arvindrajan92/sroie_document_understanding format:
        - image: PIL Image
        - ocr: list of {"box": [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], "label": str, "text": str}

    Returns:
        {
            "text": {
                "lines": [{"text": "...", "box": [x1,y1,x2,y2]}, ...]
            },
            "image": {"width": W, "height": H}
        }
    """
    image = example["image"]
    width, height = image.size

    ocr_items: List[dict] = example.get("ocr", [])

    lines = []
    for item in ocr_items:
        text = item.get("text", "").strip()
        bbox = item.get("box", [])
        if not text or not bbox:
            continue
        aabb = quad_points_to_aabb(bbox)
        lines.append({"text": text, "box": aabb})

    return {
        "text": {"lines": lines},
        "image": {"width": width, "height": height},
    }


def write_shards(
    examples,
    output_dir: Path,
    shard_size: int = 100,
) -> List[Path]:
    """
    Write examples to TAR shards.

    Each shard contains entries like:
        000000.jpg, 000000.json, 000001.jpg, 000001.json, ...
    Shards are named shard-000000.tar, shard-000001.tar, etc.

    Returns list of created shard paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: List[Path] = []
    shard_idx = 0
    entry_count = 0
    current_shard: Optional[tarfile.TarFile] = None
    current_shard_path: Optional[Path] = None

    def open_shard(idx: int) -> Tuple[tarfile.TarFile, Path]:
        path = output_dir / f"shard-{idx:06d}.tar"
        return tarfile.open(path, "w"), path

    for example in examples:
        if entry_count % shard_size == 0:
            if current_shard is not None:
                current_shard.close()
                logger.info(f"Wrote {current_shard_path}")
            current_shard, current_shard_path = open_shard(shard_idx)
            shard_paths.append(current_shard_path)
            shard_idx += 1

        # Filename stem (6-digit zero-padded global index)
        stem = f"{entry_count:06d}"

        # --- JPEG bytes ---
        image = example["image"]
        jpg_buf = io.BytesIO()
        image.save(jpg_buf, format="JPEG", quality=95)
        jpg_bytes = jpg_buf.getvalue()

        jpg_info = tarfile.TarInfo(name=f"{stem}.jpg")
        jpg_info.size = len(jpg_bytes)
        current_shard.addfile(jpg_info, io.BytesIO(jpg_bytes))

        # --- JSON annotation ---
        annotation = build_annotation(example)
        json_bytes = json.dumps(annotation, ensure_ascii=False).encode("utf-8")

        json_info = tarfile.TarInfo(name=f"{stem}.json")
        json_info.size = len(json_bytes)
        current_shard.addfile(json_info, io.BytesIO(json_bytes))

        entry_count += 1

    if current_shard is not None:
        current_shard.close()
        logger.info(f"Wrote {current_shard_path}")

    logger.info(f"Total: {entry_count} images in {len(shard_paths)} shards")
    return shard_paths


def download_and_build(
    split: str = "train",
    output_dir: Optional[Path] = None,
    shard_size: int = 100,
) -> List[Path]:
    """
    Download SROIE from HuggingFace and write TAR shards.

    Uses arvindrajan92/sroie_document_understanding (652 receipts, train split only).

    Args:
        split: Dataset split (only 'train' available in this dataset)
        output_dir: Where to write shards (defaults to CVlization cache)
        shard_size: Number of images per shard

    Returns:
        List of shard file paths
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )

    if output_dir is None:
        output_dir = get_cache_dir() / split

    output_dir = Path(output_dir)

    # Check if already built
    existing = list(output_dir.glob("shard-*.tar"))
    if existing:
        logger.info(
            f"Found {len(existing)} existing shards in {output_dir}. "
            "Delete them to re-build."
        )
        return sorted(existing)

    repo_id = "arvindrajan92/sroie_document_understanding"
    logger.info(f"Downloading SROIE from HuggingFace ({repo_id}, split={split})...")
    ds = load_dataset(repo_id, split=split)
    logger.info(f"Loaded {len(ds)} examples")

    shard_paths = write_shards(ds, output_dir, shard_size=shard_size)
    return shard_paths


def main():
    parser = argparse.ArgumentParser(
        description="Build SROIE TAR shards for vllm-ocr-eval"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train"],
        help="Dataset split (only 'train' available; default: train)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for TAR shards (default: ~/.cache/cvlization/data/ocr_reading/sroie/<split>/)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100,
        help="Number of images per shard (default: 100)",
    )
    args = parser.parse_args()

    shard_paths = download_and_build(
        split=args.split,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
    )

    print(f"\nShards written to: {shard_paths[0].parent if shard_paths else 'N/A'}")
    for p in shard_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
