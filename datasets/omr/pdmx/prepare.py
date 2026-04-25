#!/usr/bin/env python3
"""
Build per-page MusicXML dataset from PDMX (Public Domain MusicXML).

Downloads MXL files from Zenodo, filters by instrumentation/lyrics,
renders to SVG with LilyPond, extracts page breaks, slices MusicXML,
and pushes to HuggingFace as zzsi/pdmx-omr.

Usage:
    python prepare.py --inspect                    # download + print metadata stats
    python prepare.py --filter voice+piano         # process vocal+piano scores
    python prepare.py --filter piano --max-scores 100  # test with 100 piano scores
    python prepare.py --filter voice+piano --push-to-hub zzsi/pdmx-omr
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Import shared pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import (
    render_scores_to_svg,
    process_score,
    svg_dir_for_score,
    svg_page_index,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZENODO_RECORD = "15571083"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
CACHE_DIR = Path.home() / ".cache" / "pdmx"
MXL_DIR = CACHE_DIR / "mxl"
CSV_PATH = CACHE_DIR / "PDMX.csv"
SVG_DIR = CACHE_DIR / "svg"

# MIDI program numbers for common instruments
PIANO = "0"
VOICE_PROGRAMS = {"52", "53", "54", "85", "91"}  # Choir Aahs, Voice Oohs, Synth Voice, etc.


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> None:
    """Download a file with wget (shows progress)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url} ...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)


def download_pdmx():
    """Download PDMX CSV metadata and MXL archive from Zenodo."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # CSV metadata (small, always download)
    csv_url = f"{ZENODO_BASE}/PDMX.csv"
    download_file(csv_url, CSV_PATH)

    # MXL archive (1.9 GB)
    mxl_archive = CACHE_DIR / "mxl.tar.gz"
    download_file(f"{ZENODO_BASE}/mxl.tar.gz", mxl_archive)

    # Extract if not done
    if not MXL_DIR.exists() or not list(MXL_DIR.glob("*.mxl")):
        print(f"  Extracting MXL archive ...")
        MXL_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(mxl_archive) as tar:
            tar.extractall(CACHE_DIR)
        print(f"  Extracted to {MXL_DIR}")


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

def load_metadata() -> list[dict]:
    """Load PDMX.csv and return list of row dicts."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"PDMX.csv not found at {CSV_PATH}. Run with --inspect first.")
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_instruments(row: dict) -> set[str]:
    """Extract set of MIDI program numbers from the tracks column."""
    tracks = row.get("tracks", "")
    if not tracks:
        return set()
    return set(tracks.split("-"))


def filter_scores(metadata: list[dict],
                  instrument_filter: str = "all",
                  has_lyrics: bool = False,
                  max_parts: int = 10,
                  license_ok: bool = True) -> list[dict]:
    """Filter PDMX metadata by instrumentation, lyrics, license."""
    filtered = []
    for row in metadata:
        # License filter
        if license_ok and row.get("subset:no_license_conflict") == "False":
            continue

        # Deduplication filter
        if row.get("subset:deduplicated") == "False":
            continue

        instruments = get_instruments(row)

        # Instrument filter
        if instrument_filter == "voice+piano":
            has_piano = PIANO in instruments
            has_voice = bool(instruments & VOICE_PROGRAMS)
            if not (has_piano and has_voice):
                continue
        elif instrument_filter == "piano":
            if PIANO not in instruments:
                continue
            # Allow piano-only or piano+voice
            non_piano = instruments - {PIANO} - VOICE_PROGRAMS
            if non_piano:
                continue
        elif instrument_filter == "all":
            pass
        else:
            continue

        # Parts filter
        n_tracks = len(instruments)
        if n_tracks > max_parts:
            continue

        # Lyrics filter
        if has_lyrics:
            n_lyrics = int(row.get("n_lyrics", 0) or 0)
            if n_lyrics < 10:  # filter out title-only lyrics
                continue

        filtered.append(row)

    return filtered


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_pages(score_rows: list[dict], max_scores: int | None = None,
                seed: int = 42) -> list[dict]:
    """Process filtered scores through the render→slice pipeline.

    Returns list of page dicts: {score_id, page, n_pages, bar_start, bar_end, musicxml}.
    """
    all_pages = []
    skipped = 0

    scores = score_rows[:max_scores] if max_scores else score_rows

    for i, row in enumerate(scores):
        score_id = row.get("id", "")
        mxl_filename = row.get("path", "")

        # Find the MXL file
        mxl_path = MXL_DIR / mxl_filename
        if not mxl_path.exists():
            # Try alternate locations
            mxl_path = CACHE_DIR / mxl_filename
        if not mxl_path.exists():
            skipped += 1
            continue

        sdir = svg_dir_for_score(mxl_path, MXL_DIR, SVG_DIR)
        svg_files = sorted(sdir.glob("*.svg"), key=svg_page_index)

        if not svg_files:
            # Need to render — but batch rendering is more efficient.
            # For now skip and let the caller run batch render first.
            skipped += 1
            continue

        page_data = process_score(mxl_path, sdir, score_id, len(svg_files))
        if not page_data:
            skipped += 1
            continue

        for pd in page_data:
            all_pages.append({
                "score_id":  score_id,
                "page":      pd["page"],
                "n_pages":   len(svg_files),
                "bar_start": pd["bar_start"],
                "bar_end":   pd["bar_end"],
                "musicxml":  pd["musicxml"],
            })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(scores)} scores, "
                  f"{len(all_pages)} pages ...", flush=True)

    print(f"\n  Total: {len(all_pages)} pages from {len(scores) - skipped} scores "
          f"({skipped} skipped)")
    return all_pages


def split_pages(pages: list[dict], seed: int = 42,
                train_frac: float = 0.8, dev_frac: float = 0.1) -> dict:
    """Split pages into train/dev/test by score_id (no score spans splits)."""
    from datasets import Dataset, DatasetDict

    # Group by score_id
    score_ids = sorted(set(p["score_id"] for p in pages))
    rng = random.Random(seed)
    rng.shuffle(score_ids)

    n = len(score_ids)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    train_ids = set(score_ids[:n_train])
    dev_ids = set(score_ids[n_train:n_train + n_dev])

    splits = {"train": [], "dev": [], "test": []}
    for p in pages:
        if p["score_id"] in train_ids:
            splits["train"].append(p)
        elif p["score_id"] in dev_ids:
            splits["dev"].append(p)
        else:
            splits["test"].append(p)

    dd = {}
    for name, rows in splits.items():
        if rows:
            dd[name] = Dataset.from_list(rows)
            print(f"    {name}: {len(rows)} pages")

    return DatasetDict(dd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", default="voice+piano",
                        choices=["voice+piano", "piano", "all"],
                        help="Instrument filter (default: voice+piano)")
    parser.add_argument("--has-lyrics", action="store_true",
                        help="Require lyrics (n_lyrics >= 10)")
    parser.add_argument("--max-parts", type=int, default=5,
                        help="Max number of instrument tracks")
    parser.add_argument("--max-scores", type=int, default=None,
                        help="Limit number of scores (for testing)")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    parser.add_argument("--output", default=None,
                        help="Save dataset locally")
    parser.add_argument("--inspect", action="store_true",
                        help="Download + print metadata stats")
    parser.add_argument("--render-only", action="store_true",
                        help="Only render SVGs (skip slicing + push)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Step 1: Download (CSV only for --inspect, full archive otherwise)
    print("=== Downloading PDMX ===")
    if args.inspect:
        download_file(f"{ZENODO_BASE}/PDMX.csv", CSV_PATH)
    else:
        download_pdmx()

    # Step 2: Load and filter metadata
    print("\n=== Loading metadata ===")
    metadata = load_metadata()
    print(f"  Total scores in PDMX: {len(metadata)}")

    filtered = filter_scores(
        metadata,
        instrument_filter=args.filter,
        has_lyrics=args.has_lyrics,
        max_parts=args.max_parts,
    )
    print(f"  After filter ({args.filter}, lyrics={args.has_lyrics}, "
          f"max_parts={args.max_parts}): {len(filtered)} scores")

    if args.inspect:
        # Print stats and exit
        from collections import Counter
        instruments = Counter()
        for row in filtered:
            instruments[row.get("tracks", "")] += 1
        print(f"\n  Top instrumentations:")
        for inst, count in instruments.most_common(10):
            print(f"    {inst}: {count}")

        genres = Counter(row.get("genres", "") for row in filtered)
        print(f"\n  Top genres:")
        for g, count in genres.most_common(10):
            print(f"    {g or '(none)'}: {count}")

        n_lyrics = [int(row.get("n_lyrics", 0) or 0) for row in filtered]
        print(f"\n  Lyrics: {sum(1 for n in n_lyrics if n >= 10)} with ≥10 lyric tokens")
        print(f"  Median n_lyrics: {sorted(n_lyrics)[len(n_lyrics)//2]}")
        return

    scores_to_process = filtered[:args.max_scores] if args.max_scores else filtered

    # Step 3: Render SVGs
    print(f"\n=== Rendering {len(scores_to_process)} scores to SVG ===")
    # Create a temporary directory with symlinks to the filtered MXL files,
    # so we can use render_scores_to_svg with a single corpus_root
    render_dir = CACHE_DIR / "render_subset"
    render_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for row in scores_to_process:
        mxl_path = MXL_DIR / row.get("path", "")
        if not mxl_path.exists():
            mxl_path = CACHE_DIR / row.get("path", "")
        if mxl_path.exists():
            link = render_dir / mxl_path.name
            if not link.exists():
                link.symlink_to(mxl_path)
            linked += 1
    print(f"  Linked {linked} MXL files to {render_dir}")

    render_scores_to_svg(render_dir, SVG_DIR, "*.mxl")

    if args.render_only:
        print("  Render-only mode — stopping here.")
        return

    # Step 4: Slice MusicXML per page
    print(f"\n=== Slicing MusicXML ===")
    pages = build_pages(scores_to_process, max_scores=args.max_scores)

    if not pages:
        print("No pages produced. Check rendering output.")
        return

    # Step 5: Split and push
    print(f"\n=== Building dataset ===")
    dd = split_pages(pages, seed=args.seed)
    print(f"\n{dd}")

    if args.output:
        out = Path(args.output)
        print(f"\nSaving to {out} ...")
        dd.save_to_disk(str(out))

    if args.push_to_hub:
        config_name = f"pages-{args.filter}"
        if args.has_lyrics:
            config_name += "-lyrics"
        print(f"\nPushing to {args.push_to_hub} (config_name='{config_name}') ...")
        dd.push_to_hub(args.push_to_hub, config_name=config_name, private=False)
        print(f"Done — https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
