"""Build a local detection dataset (JSONL + PNGs) for OMR detection training.

Pipeline per source row:
  1. Load (image, musicxml, score_id) from a HF zzsi/* dataset.
  2. Render the MXL via Docker LilyPond to SVG (vector, DPI-independent).
  3. Rasterize the SVG to PNG via cairosvg at the target DPI — this
     guarantees the PNG's geometry matches the SVG's, so SVG-derived bboxes
     land exactly on the rendered ink (no SVG-vs-PNG layout drift).
  4. Run extract_bboxes on each SVG; bboxes come in mm and are scaled by
     PX_PER_MM to pixel coords.
  5. Use MusicXML's score-part / staves-per-part count as a hint to
     extract_bboxes' system-grouping (more reliable than geometric gap
     heuristics on multi-part scores like SATB+piano).
  6. Use keysig_extractor (MusicXML <fifths> declarations + SVG measure-
     label positions) for full keysig coverage including mid-piece changes
     and line-start restatements.

Output layout:
    <output>/
        images/<score_id>_p<page>.png
        labels_<split>.jsonl

JSONL schema (one line per page):
    {
        "score_id":  "synthetic_l7a_00042",
        "page":      1,
        "n_pages":   1,
        "image":     "images/synthetic_l7a_00042_p1.png",
        "width":     int,       # PNG width  (px)
        "height":    int,       # PNG height (px)
        "bboxes": {
            "systems":  [[x, y, w, h], ...],
            "staves":   [[sys_i, staff_i, x, y, w, h], ...],
            "barlines": [[sys_i, x, y, w, h, heavy, kind_id], ...],
                        # kind_id: 0=single 1=heavy 2=double
            "key_sigs": [[sys_i, x, y, w, h, fifths, kind_id], ...],
                        # kind_id: 0=change 1=line_start
            "measures": [[sys_i, m_idx, x, y, w, h], ...],
            "bar_numbers": [int, ...],
        }
    }

All coordinates are in pixels matching the saved PNG.

Usage:
    python make_dataset.py --output /tmp/detection_l7a --limit 50
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[5]
_OMR_PKG_DIR = _REPO_ROOT / "datasets" / "omr"
sys.path.insert(0, str(_OMR_PKG_DIR))
sys.path.insert(0, str(_THIS.parent))  # for extract_bboxes

from pipeline import LILYPOND_DIR  # noqa: E402
from extract_bboxes import extract_layout, _NS  # noqa: E402
from keysig_extractor import extract_keysigs  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
import zipfile  # noqa: E402

DPI = 150
PX_PER_MM = DPI / 25.4  # ~= 5.9055

BARLINE_KIND_ID = {"single": 0, "heavy": 1, "double": 2}
KEYSIG_KIND_ID = {"change": 0, "line_start": 1}


def _staves_per_system(mxl_text: str) -> int | None:
    """Sum of <staves> across all parts in the first measure (or 1 per part).

    Returns None on parse error; caller falls back to geometric grouping.
    """
    try:
        root = ET.fromstring(mxl_text)
    except ET.ParseError:
        return None
    total = 0
    for part in root.findall(".//part"):
        m1 = part.find("measure")
        if m1 is None:
            continue
        s_el = m1.find(".//staves")
        total += int(s_el.text) if s_el is not None and s_el.text else 1
    return total or None


def _count_measures(mxl_text: str) -> int:
    """True measure count from MusicXML (= measures of one part).

    Used as detector ground truth. Note SVG bar-number text is unreliable:
    LilyPond prints a trailing number after the final barline that does
    not correspond to a real measure.
    """
    n_parts = max(len(re.findall(r"<part\s+id", mxl_text)), 1)
    n_measure_tags = len(re.findall(r"<measure[\s>]", mxl_text))
    return n_measure_tags // n_parts


def _all_fifths(mxl_text: str) -> list[int]:
    """Every <fifths>N</fifths> value in the musicxml, in order seen.

    For openscore the row's musicxml is a per-page slice, so all key
    changes visible on that page appear here. For L7a/L9 it's the
    whole synthetic score (always single-key)."""
    return [int(m) for m in re.findall(r"<fifths>(-?\d+)</fifths>", mxl_text or "")]


# Renders both SVG and PNG of a single MXL through Docker in one shot.
# We need PNG and SVG generated from the same .ly so their coordinate
# spaces match (mm in SVG; mm * PX_PER_MM in PNG).
#
# We deliberately MATCH the HF zzsi/synthetic-scores rendering settings:
#   - No `all-bar-numbers-visible` injection (HF doesn't have it).
#   - PNG at -dresolution=150 (HF synthetic_scores/generate.py uses 150).
# That makes our detection-training PNGs visually identical to the
# images safckylj was trained on, so the YOLO trained here can be
# applied directly to HF images at inference -- no cross-rendering
# coordinate scaling.
#
# `MATCH_HF_RENDER` toggles this. When False (legacy), bar numbers are
# injected and DPI is LilyPond's default (~101). When True (default),
# we match HF exactly.
_RENDER_PY = r"""
import sys, shutil, subprocess, tempfile, re
from pathlib import Path
sys.path.insert(0, '/workspace')
from predict import musicxml_to_ly, patch_ly

mxl = Path('/data/score.musicxml')
out = Path('/out')
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    ly = musicxml_to_ly(mxl, tmp)
    patch_ly(ly)
    # Inject all-bar-numbers-visible so extract_bar_nums_from_svg can
    # anchor mid-piece keysig / measure labels by their visible number.
    text = ly.read_text()
    bar_ctx = ('\n  \\context {\n'
               '    \\Score\n'
               '    \\override BarNumber.break-visibility = ##(#t #t #t)\n'
               '    barNumberVisibility = #all-bar-numbers-visible\n'
               '  }')
    if r'\layout' in text:
        text = re.sub(r'\\layout\s*\{', lambda m: m.group(0) + bar_ctx,
                      text, count=1)
    else:
        text += '\n\\layout {\n' + bar_ctx + '\n}\n'
    ly.write_text(text)

    # SVG only -- we rasterize to PNG via cairosvg back on the host so the
    # PNG geometry exactly matches the SVG that bboxes were extracted from.
    subprocess.run(['lilypond', '--svg', str(ly)], cwd=str(tmp),
                   capture_output=True, check=True)
    for f in sorted(tmp.glob('*.svg')):
        shutil.copy(f, out / f.name)
"""


def _render_one(mxl_text: str, work_dir: Path) -> list[Path]:
    """Render an MXL text to per-page SVG via Docker LilyPond.

    Returns SVG paths sorted by page index. PNG rasterization happens
    on the host via cairosvg.
    """
    data_dir = work_dir / "data"
    out_dir = work_dir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "score.musicxml").write_text(mxl_text)
    script = work_dir / "_render.py"
    script.write_text(_RENDER_PY)

    cmd = [
        "docker", "run", "--rm",
        "--mount", f"type=bind,src={data_dir},dst=/data,readonly",
        "--mount", f"type=bind,src={out_dir},dst=/out",
        "--mount", f"type=bind,src={LILYPOND_DIR},dst=/workspace,readonly",
        "--mount", f"type=bind,src={script},dst=/render_script.py,readonly",
        "cvlization/lilypond:latest",
        "python3", "/render_script.py",
    ]
    subprocess.run(cmd, capture_output=True, check=False)

    def _page_idx(p: Path) -> int:
        stem = p.stem
        if stem == "score":
            return 1
        if stem.startswith("score-"):
            try:
                return int(stem.split("-")[-1])
            except ValueError:
                return 0
        return 0

    return sorted(out_dir.glob("*.svg"), key=_page_idx)


def _rasterize_svg(svg_path: Path, png_path: Path, dpi: int = DPI) -> tuple[int, int]:
    """Rasterize SVG to PNG at target DPI. Returns (width_px, height_px)."""
    import cairosvg  # local import
    root = ET.parse(svg_path).getroot()
    vb = root.attrib.get("viewBox", "0 0 0 0").split()
    vw_mm = float(vb[2])
    out_w = int(round(vw_mm * (dpi / 25.4)))
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                     output_width=out_w, background_color="white")
    from PIL import Image
    with Image.open(png_path) as im:
        return im.size  # (w, h)


def _svg_viewbox(svg_path: Path) -> tuple[float, float, float, float]:
    """Parse the viewBox to know the SVG canvas in mm."""
    root = ET.parse(svg_path).getroot()
    vb = root.attrib.get("viewBox", "0 0 0 0").split()
    return tuple(float(v) for v in vb)  # (min_x, min_y, w, h)


def _scale_bboxes(layout: dict, scale: float,
                  key_sigs: list[dict] | None = None) -> dict:
    """Convert all SVG-space bboxes to pixel space, with kind tags as ints."""
    def s(box):
        return [round(b * scale, 2) for b in box]

    systems = [s(item["bbox"]) for item in layout["systems"]]
    staves = [
        [item["system"], item["idx"], *s(item["bbox"])]
        for item in layout["staves"]
    ]
    barlines = [
        [item["system"], *s(item["bbox"]), int(item["heavy"]),
         BARLINE_KIND_ID.get(item.get("kind", "single"), 0)]
        for item in layout["barlines"]
    ]
    key_sigs_out = [
        [k["system"], *s(k["bbox"]), int(k.get("fifths", 0)),
         KEYSIG_KIND_ID.get(k.get("kind", "line_start"), 1)]
        for k in (key_sigs or [])
    ]
    measures = _derive_measures_px(layout, scale)
    return {
        "systems": systems,
        "staves": staves,
        "barlines": barlines,
        "key_sigs": key_sigs_out,
        "measures": measures,
        "bar_numbers": layout["bar_numbers"],
    }


def _derive_measures_px(layout: dict, scale: float) -> list[list]:
    """One bbox per (system, measure_index), spanning all staves of system.

    Built from barline x-positions per system: between consecutive barlines
    is one measure, plus the area from system left edge to the first bar.
    Returns [[sys_i, m_idx, x, y, w, h], ...] in pixel coords.
    """
    def s(v):
        return round(v * scale, 2)

    out: list[list] = []
    by_sys: dict[int, list[float]] = {}
    for b in layout["barlines"]:
        by_sys.setdefault(b["system"], []).append(b["bbox"][0])
    for sysd in layout["systems"]:
        sys_i = sysd["idx"]
        sx, sy, sw, sh = sysd["bbox"]
        xs = sorted(set(by_sys.get(sys_i, [])))
        if not xs:
            continue
        boundaries = [sx] + xs
        for m_idx, (x0, x1) in enumerate(zip(boundaries, boundaries[1:] + [sx + sw])):
            if x1 - x0 < 1.0:  # skip degenerate (e.g. doubled bars)
                continue
            out.append([sys_i, m_idx, s(x0), s(sy), s(x1 - x0), s(sh)])
    return out


def build(output_dir: Path, limit: int | None, split: str,
          repo: str = "zzsi/synthetic-scores",
          config: str = "level7a",
          streaming: bool = False,
          source_tag: str | None = None,
          dedup_by_score_id: bool = True) -> int:
    """Iterate HF rows and emit detection records.

    Returns number of pages written.
    `source_tag` defaults to the config name and is recorded on each row
    so that downstream training can stratify by source.
    """
    # Lazy import so the stub-time `--help` doesn't pull in heavy deps.
    from datasets import load_dataset
    from PIL import Image

    tag = source_tag or config
    print(f"Loading {repo} config={config} split={split} "
          f"(streaming={streaming}) ...", flush=True)
    ds = load_dataset(repo, config, split=split, streaming=streaming)

    def _iter(ds, limit):
        if streaming:
            it = iter(ds)
            for _ in range(limit if limit else 10**9):
                try:
                    yield next(it)
                except StopIteration:
                    return
        else:
            n = len(ds) if limit is None else min(limit, len(ds))
            for i in range(n):
                yield ds[i]

    rows_iter = _iter(ds, limit)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / f"labels_{split}.jsonl"

    pages_written = 0
    failed_render = 0
    seen_score_ids: set[str] = set()
    with labels_path.open("w") as f_out:
        for i, row in enumerate(rows_iter):
            score_id = row.get("score_id") or f"{tag}_{split}_{i:06d}"
            if dedup_by_score_id and score_id in seen_score_ids:
                continue
            seen_score_ids.add(score_id)
            mxl = row.get("musicxml")
            if not mxl:
                continue
            try:
                n_measures = _count_measures(mxl)
            except Exception:
                n_measures = None
            fifths_seq = _all_fifths(mxl)
            key_first = fifths_seq[0] if fifths_seq else None
            key_set = sorted(set(fifths_seq)) if fifths_seq else []
            staves_per_sys = _staves_per_system(mxl)
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                try:
                    svgs = _render_one(mxl, tmp)
                except Exception as e:
                    failed_render += 1
                    if failed_render <= 5:
                        print(f"  WARN {score_id}: render failed ({e!r})",
                              flush=True)
                    continue
                if not svgs:
                    failed_render += 1
                    continue
                # Need the MXL on disk for keysig_extractor.parse_mxl_*
                mxl_path = tmp / "score.musicxml"
                mxl_path.write_text(mxl)
                n_pages = len(svgs)
                for page_idx, svg_path in enumerate(svgs, start=1):
                    try:
                        layout = extract_layout(
                            svg_path, staves_per_system=staves_per_sys)
                    except Exception as e:
                        if failed_render <= 5:
                            print(f"  WARN {score_id} p{page_idx}: "
                                  f"extract failed ({e!r})", flush=True)
                        continue

                    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", score_id)[:80]
                    out_png = images_dir / f"{safe}_p{page_idx}.png"
                    try:
                        pw, ph = _rasterize_svg(svg_path, out_png)
                    except Exception as e:
                        if failed_render <= 5:
                            print(f"  WARN {score_id} p{page_idx}: "
                                  f"rasterize failed ({e!r})", flush=True)
                        continue
                    _, _, vw_mm, vh_mm = _svg_viewbox(svg_path)
                    scale = pw / vw_mm if vw_mm else PX_PER_MM

                    try:
                        key_sigs = extract_keysigs(svg_path, mxl_path, layout)
                    except Exception:
                        key_sigs = []
                    bboxes_px = _scale_bboxes(layout, scale, key_sigs=key_sigs)

                    rec = {
                        "score_id": score_id,
                        "source": tag,
                        "page": page_idx,
                        "n_pages": n_pages,
                        "n_measures": n_measures,
                        "key_first": key_first,    # first <fifths> in slice
                        "key_set": key_set,         # all distinct fifths on page
                        "image": f"images/{out_png.name}",
                        "width": pw,
                        "height": ph,
                        "bboxes": bboxes_px,
                    }
                    f_out.write(json.dumps(rec) + "\n")
                    pages_written += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}] pages={pages_written} failed={failed_render}",
                      flush=True)

    print(f"Wrote {pages_written} pages ({failed_render} failed) -> "
          f"{labels_path}", flush=True)
    return pages_written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", default="train",
                   choices=["train", "dev", "test"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap source rows for quick iteration")
    p.add_argument("--repo", default="zzsi/synthetic-scores")
    p.add_argument("--config", default="level7a")
    p.add_argument("--streaming", action="store_true",
                   help="use HF streaming (required for openscore "
                        "pages_transcribed, which is large)")
    p.add_argument("--source-tag", default=None,
                   help="defaults to --config; written into each "
                        "record as `source`")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build(args.output, args.limit, args.split,
          repo=args.repo, config=args.config, streaming=args.streaming,
          source_tag=args.source_tag)


if __name__ == "__main__":
    main()
