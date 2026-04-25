"""Shared MXL → per-page MusicXML pipeline for OMR dataset creation.

Provides the generic rendering, bar-number extraction, and MusicXML slicing
logic used by corpus-specific scripts (openscore, PDMX, etc.).

Public API:
    render_scores_to_svg(corpus_root, svg_dir, score_glob, exclude)
    render_single_score_svg(mxl_path, out_dir)
    extract_bar_nums_from_svgs(svg_files)
    compute_page_ranges(bar_nums_per_page, total_measures)
    process_score(mxl_path, svg_dir, score_id, n_pages_expected)
    read_musicxml(mxl_path)
    slice_musicxml(mxl_path, bar_start, bar_end)
    total_measures_in_mxl(mxl_path)
    svg_dir_for_score(mxl_path, corpus_root, svg_base)
"""

import re
import subprocess
import tempfile
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
LILYPOND_DIR = (REPO_ROOT / "examples" / "perception" /
                "optical_music_recognition" / "lilypond")

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


def render_scores_to_svg(corpus_root: Path, svg_dir: Path,
                         score_glob: str, exclude: list[str] | None = None) -> None:
    """Batch-render all MXL files in corpus_root to SVG pages via Docker."""
    svg_dir.mkdir(parents=True, exist_ok=True)
    exclude = exclude or []

    script_path = Path("/tmp/_pipeline_render.py")
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

    print(f"  Running Docker SVG batch-render ...")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"  WARNING: Docker exited with code {result.returncode}")


def render_single_score_svg(mxl_path: Path, out_dir: Path) -> None:
    """Render a single MXL to SVG pages (for inspect/debug)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mxl_path = mxl_path.resolve()

    script = f"""
import sys, shutil, tempfile
from pathlib import Path
sys.path.insert(0, '/workspace')
from predict import musicxml_to_ly, patch_ly, render_ly

import re
mxl = Path('{mxl_path.name}')
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    ly = musicxml_to_ly(Path('/data') / mxl.name, tmp)
    patch_ly(ly)
    text = ly.read_text()
    bar_ctx = (
        '\\n  \\\\context {{\\n'
        '    \\\\Score\\n'
        '    \\\\override BarNumber.break-visibility = ##(#t #t #t)\\n'
        '    barNumberVisibility = #all-bar-numbers-visible\\n'
        '  }}'
    )
    if r'\\layout' in text:
        text = re.sub(r'\\\\layout\\s*\\{{', lambda m: m.group(0) + bar_ctx, text, count=1)
    else:
        text += '\\n\\\\layout {{\\n' + bar_ctx + '\\n}}\\n'
    ly.write_text(text)
    for f in render_ly(ly, 'svg', tmp):
        shutil.copy(f, Path('/out') / f.name)
"""
    script_path = Path("/tmp/_pipeline_single_render.py")
    script_path.write_text(script)

    cmd = [
        "docker", "run", "--rm",
        "--mount", f"type=bind,src={mxl_path.parent},dst=/data,readonly",
        "--mount", f"type=bind,src={out_dir},dst=/out",
        "--mount", f"type=bind,src={LILYPOND_DIR},dst=/workspace,readonly",
        "--mount", f"type=bind,src={script_path},dst=/render_script.py,readonly",
        "cvlization/lilypond:latest",
        "python3", "/render_script.py",
    ]
    subprocess.run(cmd, capture_output=True)


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


def svg_page_index(svg_path: Path) -> int:
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
    """Return the largest run of consecutive integers (gap ≤ 1 allowed)."""
    best, current = [nums[0]], [nums[0]]
    for v in nums[1:]:
        if v - current[-1] <= 1:
            current.append(v)
        else:
            if len(current) >= len(best):
                best = current
            current = [v]
    if len(current) >= len(best):
        best = current
    return best


def extract_bar_nums_from_svgs(svg_files: list[Path]) -> dict[int, list[int]]:
    """Extract bar numbers from all SVG pages of one score.

    Returns {page_index: [sorted bar numbers]}. Uses cross-page font-size
    analysis to distinguish bar numbers from page numbers.
    """
    page_items: dict[int, list[tuple[float, int]]] = {}
    for svg_f in svg_files:
        idx = svg_page_index(svg_f)
        page_items[idx] = _collect_numeric_items(svg_f)

    all_items = [item for items in page_items.values() for item in items]
    if not all_items:
        return {idx: [] for idx in page_items}

    size_counts = Counter(fs for fs, _ in all_items)
    max_count = max(size_counts.values())
    tied = [fs for fs, cnt in size_counts.items() if cnt == max_count]
    bar_fs = min(tied) if len(tied) > 1 else tied[0]

    result: dict[int, list[int]] = {}
    for idx, items in page_items.items():
        candidates = sorted({v for fs, v in items if abs(fs - bar_fs) < 0.1})
        if not candidates:
            result[idx] = []
        elif len(candidates) == 1:
            result[idx] = candidates
        else:
            cluster = _largest_consecutive_cluster(candidates)
            if len(cluster) < len(candidates) * 0.3 and len(candidates) > 5:
                result[idx] = candidates
            else:
                result[idx] = cluster
    return result


def extract_bar_nums_from_svg(svg_path: Path) -> list[int]:
    """Single-page wrapper for testing."""
    return extract_bar_nums_from_svgs([svg_path]).get(svg_page_index(svg_path), [])


def compute_page_ranges(bar_nums_per_page: dict[int, list[int]],
                        total_measures: int) -> dict[int, tuple[int, int]]:
    """Compute {page: (bar_start, bar_end)} from bar numbers per page."""
    active = {p: nums for p, nums in bar_nums_per_page.items() if nums}
    pages = sorted(active)
    ranges = {}
    for i, page in enumerate(pages):
        bar_start = min(active[page])
        if bar_start > total_measures:
            break
        if i + 1 < len(pages):
            bar_end = min(active[pages[i + 1]]) - 1
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
    """Parse .mxl and return (music21 Score, pickup_offset)."""
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
    """Return MusicXML string for bars bar_start..bar_end (inclusive)."""
    score, offset = _load_score(mxl_path)
    actual_start = bar_start - offset
    actual_end = None if bar_end is None else bar_end - offset

    sliced = score.measures(actual_start, actual_end)
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        out_path = Path(f.name)
    try:
        sliced.write("musicxml", fp=str(out_path))
    except (TypeError, AttributeError):
        sliced = score.measures(actual_start, actual_end, gatherSpanners=False)
        sliced.write("musicxml", fp=str(out_path))
    return out_path.read_text(encoding="utf-8")


def total_measures_in_mxl(mxl_path: Path) -> int:
    """Return total number of measures in the first part."""
    score, _ = _load_score(mxl_path)
    return len(list(score.parts[0].getElementsByClass("Measure")))


# ---------------------------------------------------------------------------
# Per-score orchestrator
# ---------------------------------------------------------------------------

def process_score(mxl_path: Path, svg_dir: Path, score_id: str,
                  n_pages_expected: int) -> list[dict]:
    """Full pipeline for one score: SVGs → bar numbers → MusicXML slices.

    Returns list of {page, bar_start, bar_end, musicxml} dicts, or [] on error.
    """
    svg_files = sorted(svg_dir.glob("*.svg"), key=svg_page_index)
    if not svg_files:
        print(f"    WARN {score_id}: no SVG files in {svg_dir}", flush=True)
        return []

    if len(svg_files) != n_pages_expected:
        print(f"    WARN {score_id}: expected {n_pages_expected} SVG pages,"
              f" found {len(svg_files)}", flush=True)

    if len(svg_files) == 1:
        musicxml = read_musicxml(mxl_path)
        if not musicxml:
            return []
        return [{"page": 1, "bar_start": 1, "bar_end": None, "musicxml": musicxml}]

    bar_nums_per_page = extract_bar_nums_from_svgs(svg_files)
    empty_pages = [p for p, nums in bar_nums_per_page.items() if not nums]
    if empty_pages:
        print(f"    INFO {score_id}: {len(empty_pages)} pages have no bar nums — "
              f"will be skipped: {empty_pages}", flush=True)

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
                mxl_path, bar_start, None if is_last else bar_end,
            )
            results.append({
                "page": page,
                "bar_start": bar_start,
                "bar_end": bar_end,
                "musicxml": musicxml,
            })
        except Exception as e:
            print(f"    WARN {score_id} page {page}: slice failed: {e}", flush=True)
            return []
    return results


def svg_dir_for_score(mxl_path: Path, corpus_root: Path, svg_base: Path) -> Path:
    """Compute the SVG output directory for a given score."""
    rel = mxl_path.relative_to(corpus_root)
    return svg_base / rel.parent / rel.stem
