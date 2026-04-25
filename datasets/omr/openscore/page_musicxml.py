#!/usr/bin/env python3
"""
Build per-page MusicXML ground truth for zzsi/openscore.

For each page in the openscore dataset, this script:
  1. Renders the source MXL to SVG (with all bar numbers visible) via Docker.
  2. Extracts the bar-number range for each SVG page.
  3. Slices the full MusicXML into per-page chunks using music21.
  4. Pushes the updated dataset with a 'musicxml' column to zzsi/openscore.

Usage:
    python page_musicxml.py --corpus lieder --inspect
    python page_musicxml.py --corpus lieder --push-to-hub zzsi/openscore
    python page_musicxml.py --corpus all    --push-to-hub zzsi/openscore
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Import shared pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import (
    render_scores_to_svg,
    render_single_score_svg,
    extract_bar_nums_from_svgs,
    extract_bar_nums_from_svg,
    compute_page_ranges,
    read_musicxml,
    slice_musicxml,
    total_measures_in_mxl,
    process_score,
    svg_dir_for_score,
    svg_page_index,
)

# ---------------------------------------------------------------------------
# Openscore-specific paths and metadata
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "openscores"

CORPUS_CACHE_DIRS = {
    "lieder":    CACHE_DIR / "raw" / "lieder"    / "Lieder-main",
    "quartets":  CACHE_DIR / "raw" / "quartets"  / "StringQuartets-main",
    "orchestra": CACHE_DIR / "raw" / "orchestra"  / "Hauptstimme-main",
}

CORPORA_META = {
    "lieder":    {"score_glob": "scores/**/*.mxl", "exclude": []},
    "quartets":  {"score_glob": "scores/**/*.mxl", "exclude": []},
    "orchestra": {"score_glob": "data/**/*.mxl",   "exclude": ["_melody.mxl"]},
}


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_pages_with_musicxml(corpus_name: str) -> "DatasetDict":
    """Build per-page dataset from SVG renderings + MusicXML slicing."""
    from datasets import load_dataset, Dataset, DatasetDict

    corpus_root = CORPUS_CACHE_DIRS[corpus_name]
    svg_base    = CACHE_DIR / "svg" / corpus_name
    meta        = CORPORA_META[corpus_name]
    score_glob  = meta["score_glob"]
    exclude     = meta["exclude"]

    # Load existing dataset for train/dev/test split assignment by score_id
    print(f"Loading zzsi/openscore for split assignments ...")
    for cfg_name in ["pages_transcribed", "pages"]:
        try:
            source = load_dataset("zzsi/openscore", cfg_name)
            print(f"  Loaded config '{cfg_name}': {source}")
            break
        except Exception as e:
            print(f"  Config '{cfg_name}' failed: {e}")
            continue
    else:
        raise RuntimeError("Cannot load zzsi/openscore for split assignments")

    # Determine which scores belong to which split
    score_to_split: dict[str, str] = {}
    for split_name, split_ds in source.items():
        for row in split_ds:
            if row["corpus"] == corpus_name:
                score_to_split.setdefault(row["score_id"], split_name)

    # Build MXL index
    all_mxl = sorted(corpus_root.glob(score_glob))
    mxl_index: dict[str, Path] = {}
    for f in all_mxl:
        if not any(pat in f.name for pat in exclude):
            mxl_index[f.stem] = f
    print(f"  {len(mxl_index)} MXL files indexed for {corpus_name}")

    # Render to SVG
    print(f"  Rendering SVGs ...")
    render_scores_to_svg(corpus_root, svg_base, score_glob, exclude)

    # Process all scores
    split_rows: dict[str, list] = {"train": [], "dev": [], "test": []}
    skipped = 0
    total_pages = 0

    for score_id, mxl_path in sorted(mxl_index.items()):
        split = score_to_split.get(score_id, "train")
        sdir = svg_dir_for_score(mxl_path, corpus_root, svg_base)
        svg_files = sorted(sdir.glob("*.svg"), key=svg_page_index)
        if not svg_files:
            skipped += 1
            continue

        page_data = process_score(mxl_path, sdir, score_id, len(svg_files))
        if not page_data:
            skipped += 1
            continue

        for pd in page_data:
            split_rows[split].append({
                "score_id":  score_id,
                "corpus":    corpus_name,
                "page":      pd["page"],
                "n_pages":   len(svg_files),
                "bar_start": pd["bar_start"],
                "bar_end":   pd["bar_end"],
                "musicxml":  pd["musicxml"],
            })
            total_pages += 1

    print(f"\n  Total: {total_pages} pages from {len(mxl_index) - skipped} scores "
          f"({skipped} scores skipped)")

    split_dicts = {}
    for split_name, rows in split_rows.items():
        if rows:
            split_dicts[split_name] = Dataset.from_list(rows)
            print(f"    {split_name}: {len(rows)} rows with musicxml")

    return DatasetDict(split_dicts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build per-page MusicXML GT for zzsi/openscore"
    )
    parser.add_argument("--corpus", default="lieder",
                        choices=["lieder", "quartets", "orchestra", "all"])
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    parser.add_argument("--output", default=None)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    corpora = (["lieder", "quartets", "orchestra"]
               if args.corpus == "all" else [args.corpus])

    if args.inspect:
        _run_inspect(corpora[0])
        return

    for corpus in corpora:
        print(f"\n=== Processing corpus: {corpus} ===")
        dd = build_pages_with_musicxml(corpus)
        print(f"\nDataset:\n{dd}")

        if args.output:
            out = Path(args.output) / corpus
            print(f"\nSaving to {out} ...")
            dd.save_to_disk(str(out))

        if args.push_to_hub:
            print(f"\nPushing to {args.push_to_hub} (config_name='pages-{corpus}') ...")
            dd.push_to_hub(args.push_to_hub, config_name=f"pages-{corpus}",
                           private=False)
            print(f"Done — https://huggingface.co/datasets/{args.push_to_hub}")


def _run_inspect(corpus_name: str) -> None:
    """Quick test on one score to validate the pipeline."""
    corpus_root = CORPUS_CACHE_DIRS[corpus_name]
    svg_base    = CACHE_DIR / "svg" / corpus_name
    meta        = CORPORA_META[corpus_name]
    exclude     = meta["exclude"]

    all_mxl = sorted(corpus_root.glob(meta["score_glob"]))
    test_mxl = None
    for f in all_mxl:
        if not any(pat in f.name for pat in exclude):
            if "Mahler" in str(f) and test_mxl is None:
                test_mxl = f
    if test_mxl is None:
        for f in all_mxl:
            if not any(pat in f.name for pat in exclude):
                test_mxl = f
                break

    if test_mxl is None:
        print("No MXL files found in cache.")
        return

    print(f"Testing with: {test_mxl}")
    sdir = svg_dir_for_score(test_mxl, corpus_root, svg_base)
    sdir.mkdir(parents=True, exist_ok=True)

    if not list(sdir.glob("*.svg")):
        print("Rendering SVG ...")
        render_single_score_svg(test_mxl, sdir)

    svg_files = sorted(sdir.glob("*.svg"), key=svg_page_index)
    print(f"SVG pages: {len(svg_files)}")

    for svg_f in svg_files:
        nums = extract_bar_nums_from_svg(svg_f)
        if nums:
            print(f"  {svg_f.name}: bars {nums[:5]}...{nums[-5:]} (min={nums[0]}, max={nums[-1]})")

    total = total_measures_in_mxl(test_mxl)
    bar_nums_per_page = extract_bar_nums_from_svgs(svg_files)
    page_ranges = compute_page_ranges(bar_nums_per_page, total)
    print(f"\nPage ranges (total measures={total}):")
    for page, (start, end) in page_ranges.items():
        print(f"  Page {page}: bars {start}-{end}")

    if 1 in page_ranges:
        start, end = page_ranges[1]
        xml = slice_musicxml(test_mxl, start, None if 1 == max(page_ranges) else end)
        print(f"\nPage 1 MusicXML: {len(xml)} chars")


if __name__ == "__main__":
    main()
