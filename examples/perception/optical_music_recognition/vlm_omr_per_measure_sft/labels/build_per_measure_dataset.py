#!/usr/bin/env python3
"""Build a per-measure training dataset from rendered detection data.

For each rendered page in `--src-dir` (a `make_dataset`-style folder
with labels_train.jsonl + labels_dev.jsonl + images/):

  1. Look up the score's MusicXML from the corresponding HF source.
  2. Convert it to MXC2 (whole-page).
  3. Group MXC2 measures by m_num (across all parts) -> per-measure
     MXC2 fragment (multi-part, just that measure).
  4. From the page's GT bboxes, derive measure boxes via
     cells.derive_measures (one box per measure, spanning all staves).
  5. Pair boxes (left-to-right, top-to-bottom) with measure numbers
     in order; for openscore use the row's `bar_start` as the offset.
  6. Crop image per measure, emit (crop, measure_mxc2) records.

Output:
  <output>/images/<score_id>_m<absolute_meas_num>.png
  <output>/labels_<split>.jsonl

Each JSONL line:
  {
    "source": "l7a" | "l9" | "openscore",
    "score_id": ...,
    "page": int,
    "measure": int,           # 1-indexed absolute measure number
    "system": int,
    "image": "images/...png",
    "width": int, "height": int,
    "key_first": int | null,
    "n_parts": int,
    "mxc2": "<multi-part measure fragment>"
  }
"""

import argparse
import json
import re
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_DET = _THIS.parents[2] / "omr_detection"
_VLM = _THIS.parents[2] / "vlm_omr_sft"
sys.path.insert(0, str(_DET))
sys.path.insert(0, str(_VLM))

from cells import Box, derive_measures  # noqa: E402
from mxc2 import xml_to_mxc2  # noqa: E402
from mxc2_slice import iter_measures, n_parts  # noqa: E402

# Source repos
SOURCE_REPO = {
    "l7a": ("zzsi/synthetic-scores", "level7a", False),
    "l9":  ("zzsi/synthetic-scores", "level9",  False),
    "openscore": ("zzsi/openscore",  "pages_transcribed", True),
}


def _strip_xml_header(xml: str) -> str:
    """Subset of vlm_omr_sft.train.strip_musicxml_header for portability."""
    xml = re.sub(r"<\?xml[^?]*\?>\s*", "", xml)
    xml = re.sub(r"<!DOCTYPE[^>]*>\s*", "", xml)
    return xml


def _build_hf_index(source_tag: str, score_ids: set[str]) -> dict:
    """Map {score_id: (musicxml, bar_start, n_pages)} for the needed scores."""
    import os
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    from datasets import load_dataset
    repo, cfg, stream = SOURCE_REPO[source_tag]
    out: dict[str, tuple] = {}
    print(f"  indexing {repo}/{cfg} for {len(score_ids)} score_ids ...",
          flush=True)
    needed = set(score_ids)
    for split in ("train", "dev", "test"):
        if not needed:
            break
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=stream)
        except Exception as e:
            print(f"    {split}: load err {e!r}", flush=True)
            continue
        if stream:
            for row in ds:
                sid = row.get("score_id")
                if sid in needed and sid not in out:
                    out[sid] = (row.get("musicxml", ""),
                                int(row.get("bar_start", 1) or 1),
                                int(row.get("n_pages", 1) or 1))
                    needed.discard(sid)
                if not needed:
                    break
        else:
            for i in range(len(ds)):
                row = ds[i]
                sid = row.get("score_id")
                if sid in needed and sid not in out:
                    out[sid] = (row.get("musicxml", ""), 1, 1)
                    needed.discard(sid)
                if not needed:
                    break
    print(f"    indexed {len(out)} / {len(score_ids)}", flush=True)
    return out


def _measure_boxes_from_record(rec: dict) -> list[tuple[int, Box]]:
    """Derive per-measure boxes from a JSONL record's GT bboxes.

    Returns [(system_idx, bbox), ...] in reading order (top-to-bottom,
    left-to-right). The "measure" position within each system is the
    box's x-order.
    """
    systems = [tuple(b) for b in rec["bboxes"]["systems"]]
    staves = [tuple(b[2:6]) for b in rec["bboxes"]["staves"]]
    barlines = [tuple(b[1:5]) for b in rec["bboxes"]["barlines"]]
    measures = derive_measures(systems, staves, barlines)
    return [(m.system, m.bbox) for m in measures]


def _build_per_measure_mxc2(mxc2_full: str, measure_num: int) -> str | None:
    """For one measure number, concat slices from all parts into a
    multi-part MXC2 fragment for that measure only.

    Returns None if the measure number doesn't exist in mxc2_full.
    """
    pieces_by_part: dict[int, str] = {}
    for p_idx, m_num, slice_text in iter_measures(mxc2_full):
        try:
            mn = int(m_num)
        except (TypeError, ValueError):
            continue
        if mn != measure_num:
            continue
        pieces_by_part.setdefault(p_idx, slice_text)
    if not pieces_by_part:
        return None

    # Pull the header (P1 ... P_n + ---) from the full mxc2
    header_lines = []
    for line in mxc2_full.splitlines():
        if line.startswith("---"):
            header_lines.append("---")
            break
        header_lines.append(line)
    body = []
    # iter_measures yields p_idx as 1-indexed (1, 2, 3 ...) so no +1 here
    for p_idx in sorted(pieces_by_part):
        body.append(f"P{p_idx}")
        body.append(pieces_by_part[p_idx].rstrip())
    return "\n".join(header_lines + body) + "\n"


def _process_split(src_dir: Path, split: str, out_dir: Path) -> tuple[int, int]:
    """Returns (records_written, rows_skipped)."""
    from PIL import Image
    src_jsonl = src_dir / f"labels_{split}.jsonl"
    if not src_jsonl.exists():
        return 0, 0
    rows = [json.loads(l) for l in src_jsonl.open() if l.strip()]
    # Group by source for batched HF lookup
    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r.get("source", "l7a"), []).append(r)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"labels_{split}.jsonl"

    n_written = n_skipped = 0
    with out_jsonl.open("w") as f_out:
        for source_tag, src_rows in by_source.items():
            score_ids = {r["score_id"] for r in src_rows}
            hf_index = _build_hf_index(source_tag, score_ids)

            for i, rec in enumerate(src_rows):
                sid = rec["score_id"]
                if sid not in hf_index:
                    n_skipped += 1
                    continue
                mxl, bar_start, n_pages = hf_index[sid]
                if not mxl:
                    n_skipped += 1
                    continue
                try:
                    mxc2_full = xml_to_mxc2(_strip_xml_header(mxl),
                                             drop_beams=True)
                except Exception as e:
                    if n_skipped < 5:
                        print(f"    {sid}: xml_to_mxc2 err {e!r}", flush=True)
                    n_skipped += 1
                    continue
                try:
                    n_p = n_parts(mxc2_full)
                except Exception:
                    n_p = 1

                # measure boxes in reading order
                boxes = _measure_boxes_from_record(rec)
                if not boxes:
                    n_skipped += 1
                    continue

                page_img = Image.open(src_dir / rec["image"]).convert("RGB")
                # Map each measure box to an absolute measure number.
                # Reading order across systems = increasing measure number
                # in MXC2. For openscore the first measure on this page
                # is bar_start.
                for k, (sys_i, box) in enumerate(boxes):
                    abs_m = bar_start + k  # 1-indexed
                    measure_mxc2 = _build_per_measure_mxc2(mxc2_full, abs_m)
                    if measure_mxc2 is None:
                        # Out-of-range measure (page contained fewer real
                        # measures than detector found cells, or vice versa)
                        continue
                    x, y, w, h = box
                    if w <= 0 or h <= 0:
                        continue  # degenerate bbox from cell derivation
                    pad_x = max(4, int(0.01 * page_img.width))
                    pad_y = max(4, int(0.01 * page_img.height))
                    left = max(0, int(x - pad_x))
                    top = max(0, int(y - pad_y))
                    right = min(page_img.width, int(x + w + pad_x))
                    bottom = min(page_img.height, int(y + h + pad_y))
                    if right <= left or bottom <= top:
                        continue
                    cw, ch = right - left, bottom - top
                    # Skip degenerate slivers (unsloth refuses aspect > 200,
                    # and aspect > 20 is musically implausible for a measure).
                    if max(cw / max(ch, 1), ch / max(cw, 1)) > 20:
                        continue
                    crop = page_img.crop((left, top, right, bottom))
                    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid)[:60]
                    name = f"{source_tag}_{safe}_p{rec.get('page',1)}_m{abs_m:03d}.png"
                    crop.save(images_dir / name)
                    out_rec = {
                        "source": source_tag,
                        "score_id": sid,
                        "page": rec.get("page", 1),
                        "measure": abs_m,
                        "system": sys_i,
                        "image": f"images/{name}",
                        "width": crop.width, "height": crop.height,
                        "key_first": rec.get("key_first"),
                        "n_parts": n_p,
                        "mxc2": measure_mxc2,
                    }
                    f_out.write(json.dumps(out_rec) + "\n")
                    n_written += 1
                if (i + 1) % 50 == 0:
                    print(f"    [{source_tag} {i+1}/{len(src_rows)}] "
                          f"written={n_written} skipped={n_skipped}",
                          flush=True)
    print(f"  {split}: wrote {n_written} measure records "
          f"({n_skipped} rows skipped)", flush=True)
    return n_written, n_skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", required=True, type=Path,
                   help="A make_dataset output dir or a mixed dir with "
                        "labels_train.jsonl + labels_dev.jsonl + images/")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        _process_split(args.src_dir, split, args.output)


if __name__ == "__main__":
    main()
