#!/usr/bin/env python3
"""
Build a page-level SFT dataset from zzsi/openscore.

Filters for single-page scores (ground truth by construction — the full-score
MusicXML exactly matches the single rendered page), adds the MusicXML content
as a column, and pushes to Hub.

Multi-page support is planned via dual-render (PNG + SVG textedit) in a future
iteration; this script handles the single-page subset as an immediate first step.

Usage:
    python prepare_sft.py --push-to-hub zzsi/openscore-sft
    python prepare_sft.py --output /tmp/openscore-sft
    python prepare_sft.py --inspect
"""

import argparse
import zipfile
from pathlib import Path

from tqdm import tqdm

CACHE_DIR = Path.home() / ".cache" / "openscores"

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
  - sft
size_categories:
  - 100<n<1K
---

# OpenScore SFT — Single-Page Score Transcription

A page-level supervised fine-tuning dataset derived from [zzsi/openscore](https://huggingface.co/datasets/zzsi/openscore).

## Why single-page only?

For single-page scores the full-score MusicXML is exact ground truth for the
rendered page image — no measure-range extraction or page-break detection is
needed. This makes the (image, label) pairing reliable by construction.

Multi-page support (via LilyPond SVG textedit parsing) is planned for a future
version.

## Format

```python
{
    "image":       PIL.Image,   # rendered full-page PNG (LilyPond, 300 DPI)
    "score_id":    str,         # OpenScore score identifier
    "composer":    str,
    "opus":        str,
    "title":       str,
    "corpus":      str,         # "lieder"
    "instruments": list[str],
    "page":        int,         # always 1
    "n_pages":     int,         # always 1
    "musicxml":    str,         # full-score MusicXML (= page GT)
}
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("zzsi/openscore-sft")
example = ds["train"][0]
print(example["musicxml"][:500])
example["image"].show()
```

## Training task

```
Input:  [page image]
Prompt: "Transcribe this music score page to MusicXML."
Target: example["musicxml"]
```

For shorter sequences, convert MusicXML to LMX or bekern at training time.

## License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Attribution

Scores from [OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder).
Images rendered by the CVlization LilyPond pipeline.
"""


# ---------------------------------------------------------------------------
# MXL index
# ---------------------------------------------------------------------------

CORPUS_CACHE_DIRS = {
    "lieder":   CACHE_DIR / "raw" / "lieder"   / "Lieder-main",
    "quartets": CACHE_DIR / "raw" / "quartets" / "StringQuartets-main",
    "orchestra": CACHE_DIR / "raw" / "orchestra" / "Hauptstimme-main",
}


def build_mxl_index(corpus: str = "lieder") -> dict[str, Path]:
    """Return {score_id: mxl_path} for all .mxl files in the local cache."""
    root = CORPUS_CACHE_DIRS.get(corpus)
    if root is None or not root.exists():
        print(f"  WARNING: cache directory not found: {root}")
        return {}
    index = {}
    for p in root.rglob("*.mxl"):
        index[p.stem] = p
    return index


def build_mxl_index_all() -> dict[str, Path]:
    """Return {score_id: mxl_path} across all corpora."""
    index = {}
    for corpus in CORPUS_CACHE_DIRS:
        index.update(build_mxl_index(corpus))
    return index


def read_musicxml(mxl_path: Path) -> str:
    """Read MusicXML text from an .mxl (zip) file."""
    try:
        with zipfile.ZipFile(mxl_path) as z:
            xml_names = [n for n in z.namelist()
                         if n.endswith(".xml") and "META" not in n]
            if not xml_names:
                return ""
            return z.read(xml_names[0]).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  WARNING: could not read {mxl_path}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_sft_dataset(source_ds):
    """
    Filter source dataset for single-page scores and add musicxml column.
    Returns a DatasetDict with the same splits.
    """
    from datasets import Dataset, DatasetDict, Image as HFImage

    # Build MXL index (lieder only — quartets/orchestra have no single-page scores)
    print("Indexing MXL files ...")
    mxl_index = build_mxl_index("lieder")
    print(f"  {len(mxl_index)} lieder MXL files found in cache")

    split_datasets = {}
    for split_name, split_ds in source_ds.items():
        single = split_ds.filter(
            lambda r: r["n_pages"] == 1,
            desc=f"filter {split_name}",
        )
        print(f"\n{split_name}: {len(single)}/{len(split_ds)} single-page scores")

        rows = []
        missing = 0
        for row in tqdm(single, desc=f"  adding musicxml ({split_name})"):
            sid = row["score_id"]
            mxl_path = mxl_index.get(sid)
            if mxl_path is None:
                missing += 1
                continue
            musicxml = read_musicxml(mxl_path)
            if not musicxml:
                missing += 1
                continue
            rows.append({**row, "musicxml": musicxml})

        if missing:
            print(f"  WARNING: {missing} scores skipped (MXL not found or unreadable)")

        if not rows:
            print(f"  Skipping empty split: {split_name}")
            continue

        new_ds = Dataset.from_list(rows)
        new_ds = new_ds.cast_column("image", HFImage())
        split_datasets[split_name] = new_ds
        print(f"  {split_name}: {len(new_ds)} rows")

    return DatasetDict(split_datasets)


# ---------------------------------------------------------------------------
# Scores config (one row per score, no image, full MusicXML)
# ---------------------------------------------------------------------------

def build_scores_config(source_ds) -> "DatasetDict":
    """
    Build a score-level config for zzsi/openscore:
      one row per unique score_id, full MusicXML, no image.

    Splits are inherited from the pages config (each score belongs to exactly
    one split). This config is the join table for the multi-page SFT pipeline:
      load 'scores' → get full MusicXML
      slice by measure range (from SVG textedit mapping) → per-page GT
    """
    from datasets import Dataset, DatasetDict

    print("Building MXL index (all corpora) ...")
    mxl_index = build_mxl_index_all()
    print(f"  {len(mxl_index)} MXL files found")

    split_datasets = {}
    for split_name, split_ds in source_ds.items():
        # Deduplicate: take first row per score_id for metadata
        seen: dict[str, dict] = {}
        for row in split_ds:
            sid = row["score_id"]
            if sid not in seen:
                seen[sid] = {k: v for k, v in row.items() if k != "image"}

        print(f"\n{split_name}: {len(seen)} unique scores")

        rows = []
        missing = 0
        for sid, meta in tqdm(seen.items(), desc=f"  {split_name}"):
            mxl_path = mxl_index.get(sid)
            if mxl_path is None:
                missing += 1
                continue
            musicxml = read_musicxml(mxl_path)
            if not musicxml:
                missing += 1
                continue
            rows.append({**meta, "musicxml": musicxml})

        if missing:
            print(f"  WARNING: {missing} scores skipped (MXL not found)")
        print(f"  {split_name}: {len(rows)} rows")
        split_datasets[split_name] = Dataset.from_list(rows)

    return DatasetDict(split_datasets)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build page-level SFT dataset from zzsi/openscore"
    )
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
        help="Push to HuggingFace Hub (e.g. zzsi/openscore-sft)")
    parser.add_argument("--output", default=None,
        help="Save dataset locally to this directory")
    parser.add_argument("--scores-config", action="store_true",
        help="Push score-level config (one row/score, full MusicXML) to zzsi/openscore")
    parser.add_argument("--inspect", action="store_true",
        help="Print stats and a sample MusicXML snippet, then exit")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading zzsi/openscore (from local cache if available) ...")
    source = load_dataset("zzsi/openscore")
    print(f"  Loaded: {source}")

    if args.scores_config:
        print("\nBuilding scores config ...")
        scores_dd = build_scores_config(source)
        print(f"\nScores config:\n{scores_dd}")
        print("\nPushing scores config to zzsi/openscore ...")
        scores_dd.push_to_hub("zzsi/openscore", config_name="scores", private=False)
        print("Done — https://huggingface.co/datasets/zzsi/openscore")
        return

    if args.inspect:
        mxl_index = build_mxl_index("lieder")
        for split_name, split_ds in source.items():
            singles = [r for r in split_ds.select(range(min(500, len(split_ds))))
                       if r["n_pages"] == 1]
            print(f"\n{split_name}: {len(singles)} single-page (sampled)")
            if singles:
                r = singles[0]
                print(f"  Example: {r['score_id']} — {r['composer']}: {r['title']}")
                mxl_path = mxl_index.get(r["score_id"])
                if mxl_path:
                    xml = read_musicxml(mxl_path)
                    print(f"  MusicXML snippet:\n{xml[:400]}")
        return

    dd = build_sft_dataset(source)
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
        card_path = CACHE_DIR / "openscore_sft_README.md"
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
