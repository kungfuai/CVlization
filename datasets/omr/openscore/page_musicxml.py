#!/usr/bin/env python3
"""
Build per-page MusicXML ground truth for zzsi/openscore (pages config).

For each page in the openscore dataset, this script:
  1. Renders the source MXL to SVG (with all bar numbers visible) via Docker.
  2. Extracts the bar-number range for each SVG page.
  3. Slices the full MusicXML into per-page chunks using music21.
  4. Pushes the updated dataset with a 'musicxml' column to zzsi/openscore
     under config_name='pages'.

Single-page scores: ground truth is the full MusicXML (no slicing needed).
Multi-page scores: ground truth is per-page slice from bar range.

Usage:
    python page_musicxml.py --corpus lieder --inspect
    python page_musicxml.py --corpus lieder --push-to-hub zzsi/openscore
    python page_musicxml.py --corpus all    --push-to-hub zzsi/openscore
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (mirrors prepare.py)
# ---------------------------------------------------------------------------

CACHE_DIR    = Path.home() / ".cache" / "openscores"
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent.parent
LILYPOND_DIR = (REPO_ROOT / "examples" / "perception" /
                "optical_music_recognition" / "lilypond")

CORPUS_CACHE_DIRS = {
    "lieder":   CACHE_DIR / "raw" / "lieder"   / "Lieder-main",
    "quartets": CACHE_DIR / "raw" / "quartets" / "StringQuartets-main",
    "orchestra": CACHE_DIR / "raw" / "orchestra" / "Hauptstimme-main",
}

CORPORA_META = {
    "lieder":    {"score_glob": "scores/**/*.mxl", "exclude": []},
    "quartets":  {"score_glob": "scores/**/*.mxl", "exclude": []},
    "orchestra": {"score_glob": "data/**/*.mxl",   "exclude": ["_melody.mxl"]},
}

# ---------------------------------------------------------------------------
# SVG rendering: MXL → per-page SVG with all bar numbers visible
# ---------------------------------------------------------------------------

# Inline Python to run inside Docker — renders MXL → SVG with all-bar-numbers-visible
_DOCKER_RENDER_PY = r"""
import sys, os, re, shutil, tempfile
from pathlib import Path
sys.path.insert(0, '/workspace')
from predict import musicxml_to_ly, patch_ly, render_ly

raw_dir    = Path('/raw')
svg_dir    = Path('/svg')
score_glob = sys.argv[1]
exclude    = sys.argv[2:]

exclude = [p for p in exclude if p]
all_files = sorted(raw_dir.glob(score_glob))
score_files = [f for f in all_files if not any(pat in f.name for pat in exclude)]
print(f"  Rendering {len(score_files)} scores to SVG ...", flush=True)

for sf in score_files:
    rel       = sf.relative_to(raw_dir)
    out_dir   = svg_dir / rel.parent / rel.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already rendered
    if list(out_dir.glob("*.svg")):
        continue

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ly = musicxml_to_ly(sf, tmp)
            patch_ly(ly)

            # Inject all-bar-numbers-visible into \layout block
            text = ly.read_text()
            bar_ctx = (
                '\n  \\context {\n'
                '    \\Score\n'
                '    \\override BarNumber.break-visibility = ##(#t #t #t)\n'
                '    barNumberVisibility = #all-bar-numbers-visible\n'
                '  }'
            )
            if r'\layout' in text:
                text = re.sub(r'\\layout\s*\{', lambda m: m.group(0) + bar_ctx, text, count=1)
            else:
                text += '\n\\layout {\n' + bar_ctx + '\n}\n'
            ly.write_text(text)

            svg_files = render_ly(ly, 'svg', tmp)
            for f in svg_files:
                shutil.copy(f, out_dir / f.name)
    except Exception as e:
        print(f"  WARN {rel}: {e}", flush=True)

print("  SVG batch done", flush=True)
"""


def render_corpus_to_svg(corpus_name: str, corpus_root: Path, svg_dir: Path) -> None:
    """Run Docker to batch-render all MXLs in corpus_root to SVG pages."""
    svg_dir.mkdir(parents=True, exist_ok=True)
    meta = CORPORA_META[corpus_name]
    score_glob = meta["score_glob"]
    exclude = meta["exclude"]

    # Write the inline script to a temp file so Docker can run it
    script_path = Path("/tmp/_page_musicxml_render.py")
    script_path.write_text(_DOCKER_RENDER_PY)

    cmd = [
        "docker", "run", "--rm",
        "--mount", f"type=bind,src={corpus_root},dst=/raw,readonly",
        "--mount", f"type=bind,src={svg_dir},dst=/svg",
        "--mount", f"type=bind,src={LILYPOND_DIR},dst=/workspace,readonly",
        "--mount", f"type=bind,src={script_path},dst=/render_script.py,readonly",
        "cvlization/lilypond:latest",
        "python3", "/render_script.py", score_glob,
    ] + exclude

    print(f"  Running Docker SVG batch-render for {corpus_name} ...")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"  WARNING: Docker exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Bar number extraction from SVG
# ---------------------------------------------------------------------------

def _walk_text(elem, parent_transform=""):
    """Walk SVG tree, yielding (text_content, font_size, y_pos) for each <text> element."""
    transform = elem.attrib.get("transform", parent_transform)
    tag = elem.tag.split("}")[-1]

    if tag == "text":
        all_text = (elem.text or "")
        for child in elem:
            all_text += (child.text or "") + (child.tail or "")
        all_text = all_text.strip()
        fs = float(elem.attrib.get("font-size", "999"))
        m = re.search(r"translate\([^,]+,\s*([\d.]+)\)", transform)
        y = float(m.group(1)) if m else None
        yield (all_text, fs, y)

    for child in elem:
        yield from _walk_text(child, transform)


def _svg_page_index(svg_path: Path) -> int:
    """Extract page index from 'score-N.svg' filename for numeric sorting."""
    m = re.search(r"-(\d+)\.svg$", svg_path.name)
    return int(m.group(1)) if m else 0


def _collect_numeric_items(svg_path: Path) -> list[tuple[float, int]]:
    """Return (font_size, value) for every numeric <text> element in the SVG."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    items: list[tuple[float, int]] = []
    for text, fs, y in _walk_text(root):
        if re.match(r"^\d+$", text):
            items.append((fs, int(text)))
    return items


def _largest_consecutive_cluster(nums: list[int]) -> list[int]:
    """
    Return the largest run of consecutive integers in a sorted list.
    Gaps of ≤ 1 are treated as consecutive (allows a single missing bar number).
    When two clusters tie in length, the LATER (larger) one is preferred,
    since bar numbers grow through the score and contaminating values
    (page numbers, multimeasure-rest counts) are typically small.
    """
    best, current = [nums[0]], [nums[0]]
    for v in nums[1:]:
        if v - current[-1] <= 1:
            current.append(v)
        else:
            if len(current) >= len(best):  # >= to prefer later cluster on tie
                best = current
            current = [v]
    if len(current) >= len(best):
        best = current
    return best


def extract_bar_nums_from_svgs(svg_files: list[Path]) -> dict[int, list[int]]:
    """
    Extract bar numbers from all SVG pages of ONE score, returning
    {page_index: [sorted bar numbers]} with [] for pages that have no
    extractable bar numbers.

    Uses a cross-page canonical font size to robustly separate bar numbers
    (small font, appear many times across all pages) from page numbers and
    other numerals (larger font, appear only once per page).

    Strategy:
    1. Parse every page's SVG and collect (font_size, value) pairs.
    2. Determine the canonical bar-number font size by counting occurrences
       ACROSS ALL PAGES: bar numbers dominate by count (one per system per page
       × many pages), while page numbers appear exactly once per page.
    3. For each page, extract candidates at the canonical font size, then
       remove outliers via largest-consecutive-cluster filtering.
    4. Pages with no elements at the canonical font size get [].
    """
    # ── pass 1: collect all numeric items per page ────────────────────────
    page_items: dict[int, list[tuple[float, int]]] = {}
    for svg_f in svg_files:
        idx = _svg_page_index(svg_f)
        page_items[idx] = _collect_numeric_items(svg_f)

    all_items = [item for items in page_items.values() for item in items]
    if not all_items:
        return {idx: [] for idx in page_items}

    # ── determine canonical bar-number font size across all pages ─────────
    # Bar numbers appear 1×/system × many_systems × many_pages ≫ n_pages
    # Page numbers appear exactly once per page (= n_pages times total).
    # So the most-common font size across all pages is the bar-number size.
    size_counts = Counter(fs for fs, _ in all_items)
    max_count = max(size_counts.values())
    tied = [fs for fs, cnt in size_counts.items() if cnt == max_count]
    if len(tied) == 1:
        bar_fs = tied[0]
    else:
        # Rare tie: prefer smaller font (bar nums are smaller than page nums)
        bar_fs = min(tied)

    # ── pass 2: extract per-page bar numbers using canonical bar_fs ───────
    result: dict[int, list[int]] = {}
    for idx, items in page_items.items():
        candidates = sorted({v for fs, v in items if abs(fs - bar_fs) < 0.1})
        if not candidates:
            result[idx] = []
        elif len(candidates) == 1:
            result[idx] = candidates
        else:
            cluster = _largest_consecutive_cluster(candidates)
            # Safety: if cluster is tiny vs all candidates, skip filtering
            if len(cluster) < len(candidates) * 0.3 and len(candidates) > 5:
                result[idx] = candidates
            else:
                result[idx] = cluster

    return result


def extract_bar_nums_from_svg(svg_path: Path) -> list[int]:
    """Single-page wrapper around extract_bar_nums_from_svgs (for inspect/testing)."""
    return extract_bar_nums_from_svgs([svg_path]).get(_svg_page_index(svg_path), [])


def compute_page_ranges(bar_nums_per_page: dict[int, list[int]],
                        total_measures: int) -> dict[int, tuple[int, int]]:
    """
    Given {page: [sorted bar numbers]} ([] = no bar nums on that page),
    return {page: (bar_start, bar_end)} for pages WITH bar numbers only.

    bar_start = min(bar nums on page N)
    bar_end   = min(bar nums on next page WITH bar nums) - 1
              (or total_measures for the last such page)

    Pages with [] bar numbers are omitted from the output — their bars will
    be absorbed into the adjacent pages' ranges.
    """
    # Only work with pages that have extractable bar numbers
    active = {p: nums for p, nums in bar_nums_per_page.items() if nums}
    pages = sorted(active)
    ranges = {}
    for i, page in enumerate(pages):
        bar_start = min(active[page])
        # Skip pages whose bar_start exceeds the score length — LilyPond
        # sometimes emits a "phantom" bar number after the final measure.
        if bar_start > total_measures:
            break
        if i + 1 < len(pages):
            next_page = pages[i + 1]
            bar_end = min(active[next_page]) - 1
        else:
            bar_end = total_measures
        ranges[page] = (bar_start, bar_end)
    return ranges


# ---------------------------------------------------------------------------
# MusicXML slicing via music21
# ---------------------------------------------------------------------------

def read_musicxml(mxl_path: Path) -> str:
    """Read raw MusicXML text from an .mxl (zip) file."""
    with zipfile.ZipFile(mxl_path) as z:
        xml_names = [n for n in z.namelist()
                     if n.endswith(".xml") and "META" not in n]
        if not xml_names:
            return ""
        return z.read(xml_names[0]).decode("utf-8", errors="replace")


def _load_score(mxl_path: Path):
    """Parse an .mxl file and return (music21 Score, pickup_offset).

    pickup_offset is 1 when measure 0 exists (LilyPond bar N = music21 measure N-1),
    0 otherwise.
    """
    from music21 import converter

    with zipfile.ZipFile(mxl_path) as z:
        xml_names = [n for n in z.namelist()
                     if n.endswith(".xml") and "META" not in n]
        xml_bytes = z.read(xml_names[0])

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        f.write(xml_bytes)
        tmp_xml = Path(f.name)
    try:
        score = converter.parse(str(tmp_xml), format="musicxml")
    finally:
        tmp_xml.unlink(missing_ok=True)

    measures = list(score.parts[0].getElementsByClass("Measure"))
    first_num = measures[0].number if measures else 1
    pickup_offset = 1 if first_num == 0 else 0
    return score, pickup_offset


def slice_musicxml(mxl_path: Path, bar_start: int, bar_end: int | None) -> str:
    """
    Return MusicXML string for LilyPond bars bar_start..bar_end (inclusive).
    If bar_end is None, extracts to end of score.

    Handles pickup bars: when the first measure is numbered 0, LilyPond bar N
    maps to music21 measure N-1, so bar_start/bar_end are adjusted accordingly.
    """
    score, offset = _load_score(mxl_path)
    actual_start = bar_start - offset
    actual_end   = None if bar_end is None else bar_end - offset

    sliced = score.measures(actual_start, actual_end)
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        out_path = Path(f.name)
    try:
        sliced.write("musicxml", fp=str(out_path))
    except (TypeError, AttributeError):
        # music21 bug: arpeggio spanners that cross page boundaries cause
        # len() to be called on a Note instead of a collection.  Retry
        # without cross-boundary spanners — loses some expression marks at
        # page edges but is otherwise correct.
        sliced = score.measures(actual_start, actual_end, gatherSpanners=False)
        sliced.write("musicxml", fp=str(out_path))
    return out_path.read_text(encoding="utf-8")


def total_measures_in_mxl(mxl_path: Path) -> int:
    """Return the total number of measures in the first part of an .mxl file."""
    score, _ = _load_score(mxl_path)
    return len(list(score.parts[0].getElementsByClass("Measure")))


# ---------------------------------------------------------------------------
# Per-score pipeline: MXL + SVG pages → list of {page, bar_start, bar_end, musicxml}
# ---------------------------------------------------------------------------

def process_score(mxl_path: Path, svg_dir: Path, score_id: str,
                  n_pages_expected: int) -> list[dict]:
    """
    Given the MXL path and the SVG directory for this score, return
    a list of per-page dicts: {page, bar_start, bar_end, musicxml}.

    Returns [] on any error.
    """
    svg_files = sorted(svg_dir.glob("*.svg"), key=_svg_page_index)
    if not svg_files:
        print(f"    WARN {score_id}: no SVG files in {svg_dir}", flush=True)
        return []

    if len(svg_files) != n_pages_expected:
        print(f"    WARN {score_id}: expected {n_pages_expected} SVG pages,"
              f" found {len(svg_files)}", flush=True)
        # Proceed anyway — may have partial coverage

    # Special case: single-page score (SVG bar extraction not needed)
    if len(svg_files) == 1:
        musicxml = read_musicxml(mxl_path)
        if not musicxml:
            return []
        return [{"page": 1, "bar_start": 1, "bar_end": None, "musicxml": musicxml}]

    # Multi-page: extract bar numbers using cross-page canonical font size
    bar_nums_per_page = extract_bar_nums_from_svgs(svg_files)
    empty_pages = [p for p, nums in bar_nums_per_page.items() if not nums]
    if empty_pages:
        print(f"    INFO {score_id}: {len(empty_pages)} pages have no bar nums "
              f"(page-number only) — will be skipped: {empty_pages}", flush=True)

    # Get total measures
    try:
        total = total_measures_in_mxl(mxl_path)
    except Exception as e:
        print(f"    WARN {score_id}: cannot count measures: {e}", flush=True)
        return []

    page_ranges = compute_page_ranges(bar_nums_per_page, total)

    results = []
    for page, (bar_start, bar_end) in page_ranges.items():
        try:
            is_last = (page == max(page_ranges))
            musicxml = slice_musicxml(
                mxl_path,
                bar_start,
                None if is_last else bar_end,
            )
            results.append({
                "page":      page,
                "bar_start": bar_start,
                "bar_end":   bar_end,
                "musicxml":  musicxml,
            })
        except Exception as e:
            print(f"    WARN {score_id} page {page}: slice failed: {e}", flush=True)
            return []

    return results


# ---------------------------------------------------------------------------
# Build updated dataset with musicxml column
# ---------------------------------------------------------------------------

def build_pages_with_musicxml(corpus_name: str) -> "DatasetDict":
    """
    Build per-page dataset from SVG renderings + MusicXML slicing.

    Instead of matching against original page metadata, derives page breaks
    directly from the SVG renderings. This means the page count and measure
    ranges reflect the current LilyPond settings (staff size, max systems).
    """
    from datasets import load_dataset, Dataset, DatasetDict, Image as HFImage

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

    # Build MXL index for this corpus
    all_mxl = sorted(corpus_root.glob(score_glob))
    mxl_index: dict[str, Path] = {}
    for f in all_mxl:
        if not any(pat in f.name for pat in exclude):
            mxl_index[f.stem] = f
    print(f"  {len(mxl_index)} MXL files indexed for {corpus_name}")

    # Render to SVG (skips already-rendered)
    print(f"  Rendering SVGs ...")
    render_corpus_to_svg(corpus_name, corpus_root, svg_base)

    # Process all scores and build pages from SVG page breaks
    split_rows: dict[str, list] = {"train": [], "dev": [], "test": []}
    skipped = 0
    total_pages = 0

    for score_id, mxl_path in sorted(mxl_index.items()):
        split = score_to_split.get(score_id, "train")
        svg_dir = _svg_dir_for_score(mxl_path, corpus_root, svg_base)
        svg_files = sorted(svg_dir.glob("*.svg"), key=_svg_page_index)
        if not svg_files:
            skipped += 1
            continue

        n_svg_pages = len(svg_files)
        page_data = process_score(mxl_path, svg_dir, score_id, n_svg_pages)
        if not page_data:
            skipped += 1
            continue

        for pd in page_data:
            split_rows[split].append({
                "score_id":  score_id,
                "corpus":    corpus_name,
                "page":      pd["page"],
                "n_pages":   n_svg_pages,
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


def _svg_dir_for_score(mxl_path: Path, corpus_root: Path, svg_base: Path) -> Path:
    """Return the directory where SVG pages for this score should be."""
    rel = mxl_path.relative_to(corpus_root)
    return svg_base / rel.parent / rel.stem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build per-page MusicXML GT for zzsi/openscore"
    )
    parser.add_argument("--corpus", default="lieder",
                        choices=["lieder", "quartets", "orchestra", "all"],
                        help="Which corpus to process")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                        help="Push updated pages config to Hub (e.g. zzsi/openscore)")
    parser.add_argument("--output", default=None,
                        help="Save dataset locally to this directory")
    parser.add_argument("--inspect", action="store_true",
                        help="Process one score and print result, then exit")
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
            dd.push_to_hub(args.push_to_hub,
                           config_name=f"pages-{corpus}",
                           private=False)
            print(f"Done — https://huggingface.co/datasets/{args.push_to_hub}")


def _run_inspect(corpus_name: str) -> None:
    """Quick test on a few multi-page scores to validate the pipeline."""
    corpus_root = CORPUS_CACHE_DIRS[corpus_name]
    svg_base    = CACHE_DIR / "svg" / corpus_name
    meta        = CORPORA_META[corpus_name]
    exclude     = meta["exclude"]

    # Find a multi-page score in the cache (prefer Mahler for lieder)
    all_mxl = sorted(corpus_root.glob(meta["score_glob"]))
    test_mxl = None
    for f in all_mxl:
        if not any(pat in f.name for pat in exclude):
            if "Mahler" in str(f) and test_mxl is None:
                test_mxl = f  # prefer Mahler (known multi-page, validated)
    if test_mxl is None:
        for f in all_mxl:
            if not any(pat in f.name for pat in exclude):
                test_mxl = f
                break

    if test_mxl is None:
        print("No MXL files found in cache.")
        return

    score_id = test_mxl.stem
    print(f"Testing with: {test_mxl}")

    svg_dir = _svg_dir_for_score(test_mxl, corpus_root, svg_base)
    svg_dir.mkdir(parents=True, exist_ok=True)

    # Render if needed
    if not list(svg_dir.glob("*.svg")):
        print("Rendering SVG ...")
        _render_single_score_svg(test_mxl, svg_dir)

    svg_files = sorted(svg_dir.glob("*.svg"), key=_svg_page_index)
    print(f"SVG pages: {len(svg_files)}")

    for svg_f in svg_files:
        nums = extract_bar_nums_from_svg(svg_f)
        print(f"  {svg_f.name}: bars {nums[:5]}...{nums[-5:]} (min={nums[0]}, max={nums[-1]})")

    total = total_measures_in_mxl(test_mxl)
    bar_nums_per_page = extract_bar_nums_from_svgs(svg_files)
    page_ranges = compute_page_ranges(bar_nums_per_page, total)
    print(f"\nPage ranges (total measures={total}):")
    for page, (start, end) in page_ranges.items():
        print(f"  Page {page}: bars {start}-{end}")

    # Slice one page
    print("\nSlicing page 1 ...")
    start, end = page_ranges[1]
    xml = slice_musicxml(test_mxl, start, None if 1 == max(page_ranges) else end)
    print(f"  Page 1 MusicXML: {len(xml)} chars")
    print(f"  Snippet: {xml[:200]}")


def _render_single_score_svg(mxl_path: Path, out_dir: Path) -> None:
    """Render a single MXL file to SVG inside Docker.

    Copies the MXL to a clean /tmp path to avoid Docker mount issues with
    special characters (commas, apostrophes) in corpus directory paths.
    """
    import shutil as _shutil
    script = Path("/tmp/_page_musicxml_render_single.py")
    script.write_text(r"""
import sys, os, re, shutil, tempfile
from pathlib import Path
sys.path.insert(0, '/workspace')
from predict import musicxml_to_ly, patch_ly, render_ly

sf   = Path('/input/score.mxl')
out  = Path('/output')
out.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    ly = musicxml_to_ly(sf, tmp)
    patch_ly(ly)
    text = ly.read_text()
    bar_ctx = (
        '\n  \\context {\n'
        '    \\Score\n'
        '    \\override BarNumber.break-visibility = ##(#t #t #t)\n'
        '    barNumberVisibility = #all-bar-numbers-visible\n'
        '  }'
    )
    if r'\layout' in text:
        text = re.sub(r'\\layout\s*\{', lambda m: m.group(0) + bar_ctx, text, count=1)
    else:
        text += '\n\\layout {\n' + bar_ctx + '\n}\n'
    ly.write_text(text)
    svg_files = render_ly(ly, 'svg', tmp)
    for f in svg_files:
        shutil.copy(f, out / f.name)
print(f"Done: {len(list(Path('/output').glob('*.svg')))} SVG files", flush=True)
""")
    # Copy MXL to a clean /tmp path (avoids Docker --mount issues with
    # commas/apostrophes in corpus directory paths)
    clean_input_dir = Path("/tmp/_pmx_input")
    clean_input_dir.mkdir(exist_ok=True)
    clean_mxl = clean_input_dir / "score.mxl"
    _shutil.copy2(mxl_path, clean_mxl)

    # Use a clean output dir in /tmp, then copy results back
    clean_out_dir = Path("/tmp/_pmx_output")
    _shutil.rmtree(clean_out_dir, ignore_errors=True)
    clean_out_dir.mkdir(exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--mount", f"type=bind,src={clean_input_dir},dst=/input,readonly",
        "--mount", f"type=bind,src={clean_out_dir},dst=/output",
        "--mount", f"type=bind,src={LILYPOND_DIR},dst=/workspace,readonly",
        "--mount", f"type=bind,src={script},dst=/render_single.py,readonly",
        "cvlization/lilypond:latest",
        "python3", "/render_single.py",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"  Docker error:\n{result.stderr}", flush=True)
        return

    # Copy SVGs from clean temp dir to actual output dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for svg_f in sorted(clean_out_dir.glob("*.svg")):
        _shutil.copy2(svg_f, out_dir / svg_f.name)
    print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
