"""Build a local detection dataset (JSONL + PNGs) from L7a synthetic scores.

Pipeline per source row:
  1. Load (image, musicxml, score_id) from HF zzsi/synthetic-scores level7a.
  2. Write the MXL temporarily to disk.
  3. Re-render via Docker -> per-page SVG and per-page PNG (same .ly, same DPI).
  4. Run extract_bboxes on each SVG.
  5. Convert bboxes from SVG user units (~mm) to pixels using the DPI used
     for PNG rendering. LilyPond default rendering at -dresolution=150
     gives 150/25.4 ~= 5.91 px per mm.
  6. Save the PNG to <output>/images/<score_id>_p<page>.png and append a
     JSONL line with paths + bboxes (in pixel coords).

Output layout:
    <output>/
        images/<score_id>_p<page>.png
        labels.jsonl

JSONL schema (one line per page):
    {
        "score_id": "synthetic_l7a_00042",
        "page":     1,
        "n_pages":  1,
        "image":    "images/synthetic_l7a_00042_p1.png",
        "width":    int,       # PNG width  (px)
        "height":   int,       # PNG height (px)
        "bboxes": {
            "systems":  [[x, y, w, h], ...],
            "staves":   [[sys_i, staff_i, x, y, w, h], ...],
            "barlines": [[sys_i, x, y, w, h, heavy], ...],
            "bar_numbers": [int, ...],   # measure numbers on this page
        }
    }

Coordinates are in **pixels** matching the saved PNG.

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
import xml.etree.ElementTree as ET  # noqa: E402

DPI = 150
PX_PER_MM = DPI / 25.4  # ~= 5.9055


def _count_measures(mxl_text: str) -> int:
    """True measure count from MusicXML (= measures of one part).

    Used as detector ground truth. Note SVG bar-number text is unreliable:
    LilyPond prints a trailing number after the final barline that does
    not correspond to a real measure.
    """
    n_parts = max(len(re.findall(r"<part\s+id", mxl_text)), 1)
    n_measure_tags = len(re.findall(r"<measure[\s>]", mxl_text))
    return n_measure_tags // n_parts


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

MATCH_HF_RENDER = __MATCH_HF_RENDER__

mxl = Path('/data/score.musicxml')
out = Path('/out')
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    ly = musicxml_to_ly(mxl, tmp)
    patch_ly(ly)
    if not MATCH_HF_RENDER:
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

    # SVG render -- vector, DPI-independent
    subprocess.run(['lilypond', '--svg', str(ly)], cwd=str(tmp),
                   capture_output=True, check=True)
    # PNG render -- match HF DPI when requested
    png_cmd = ['lilypond', '--png']
    if MATCH_HF_RENDER:
        png_cmd.append('-dresolution=150')
    png_cmd.append(str(ly))
    subprocess.run(png_cmd, cwd=str(tmp), capture_output=True, check=True)

    for f in sorted(tmp.glob('*.svg')) + sorted(tmp.glob('*.png')):
        shutil.copy(f, out / f.name)
"""


def _render_one(mxl_text: str, work_dir: Path,
                match_hf: bool = True) -> tuple[list[Path], list[Path]]:
    """Render an MXL text to per-page SVG + PNG via Docker.

    Returns (svg_files, png_files) sorted by page index.
    `match_hf=True` skips the bar-number injection and renders PNG at
    150 DPI so the output matches the HF synthetic-scores PNGs that
    safckylj was trained on.
    """
    data_dir = work_dir / "data"
    out_dir = work_dir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "score.musicxml").write_text(mxl_text)
    script = work_dir / "_render.py"
    script.write_text(_RENDER_PY.replace("__MATCH_HF_RENDER__",
                                          "True" if match_hf else "False"))

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
        # LilyPond names: score-1.svg, score-2.svg, or 'score.png' (single page)
        stem = p.stem
        if stem == "score":
            return 1
        if stem.startswith("score-"):
            try:
                return int(stem.split("-")[-1])
            except ValueError:
                return 0
        return 0

    svgs = sorted(out_dir.glob("*.svg"), key=_page_idx)
    pngs = sorted(out_dir.glob("*.png"), key=_page_idx)
    return svgs, pngs


def _svg_viewbox(svg_path: Path) -> tuple[float, float, float, float]:
    """Parse the viewBox to know the SVG canvas in mm."""
    root = ET.parse(svg_path).getroot()
    vb = root.attrib.get("viewBox", "0 0 0 0").split()
    return tuple(float(v) for v in vb)  # (min_x, min_y, w, h)


def _scale_bboxes(layout: dict, scale: float) -> dict:
    """Convert all SVG-space bboxes to pixel space."""
    def s(box):
        return [round(b * scale, 2) for b in box]
    systems = [s(item["bbox"]) for item in layout["systems"]]
    staves = [
        [item["system"], item["idx"], *s(item["bbox"])]
        for item in layout["staves"]
    ]
    barlines = [
        [item["system"], *s(item["bbox"]), int(item["heavy"])]
        for item in layout["barlines"]
    ]
    key_sigs = [
        [item["system"], *s(item["bbox"])]
        for item in layout.get("key_sigs", [])
    ]
    return {
        "systems": systems,
        "staves": staves,
        "barlines": barlines,
        "key_sigs": key_sigs,
        "bar_numbers": layout["bar_numbers"],
    }


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
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                try:
                    svgs, pngs = _render_one(mxl, tmp)
                except Exception as e:
                    failed_render += 1
                    if failed_render <= 5:
                        print(f"  WARN {score_id}: render failed ({e!r})",
                              flush=True)
                    continue
                if not svgs or not pngs:
                    failed_render += 1
                    continue
                n_pages = len(svgs)
                for page_idx, svg_path in enumerate(svgs, start=1):
                    png_path = pngs[page_idx - 1] if page_idx - 1 < len(pngs) else pngs[0]

                    try:
                        layout = extract_layout(svg_path)
                    except Exception as e:
                        if failed_render <= 5:
                            print(f"  WARN {score_id} p{page_idx}: "
                                  f"extract failed ({e!r})", flush=True)
                        continue
                    _, _, vw_mm, vh_mm = _svg_viewbox(svg_path)
                    with Image.open(png_path) as im:
                        pw, ph = im.size
                    scale = pw / vw_mm if vw_mm else PX_PER_MM
                    bboxes_px = _scale_bboxes(layout, scale)

                    # File-safe stem (openscore score_ids may have spaces / punctuation)
                    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", score_id)[:80]
                    out_png = images_dir / f"{safe}_p{page_idx}.png"
                    Image.open(png_path).save(out_png)

                    rec = {
                        "score_id": score_id,
                        "source": tag,
                        "page": page_idx,
                        "n_pages": n_pages,
                        "n_measures": n_measures,
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
