"""SVG -> layout bboxes for LilyPond-rendered pages.

Public API: extract_layout(svg_path) -> dict with keys
  systems   : [bbox, ...]                  top-to-bottom on the page
  staves    : [{"system": i, "idx": j, "bbox": ...}, ...]
  barlines  : [{"system": i, "bbox": ...}, ...]
  bar_numbers : [int, ...]                 (delegated to pipeline.py)

Coordinates are in SVG user units (LilyPond emits mm-equivalents into a
viewBox; we keep these and let the rasterizer scale to pixels).

Heuristics (verified against L7a synthetic SVGs):
  - Staff line: <line> with y1==y2, length > MIN_STAFF_LEN, stroke-width
    ~ 0.1. Staves come in groups of 5 parallel lines.
  - Barline:   <rect> with width in [0.15, 0.30] (single), height >= 3.5.
               Wider rects (>= 0.4) are heavy/final barlines.
               Filtered out: stems (width ~0.13), beams (width >> 1).
  - System:    horizontal slab covering all staves whose top edges share
               an x-range. Detected by clustering staff groups by their
               vertical proximity (gap < SYSTEM_GAP).

Key signatures and clefs are NOT detected here -- they live in the
Emmentaler font glyphs and would require glyph fingerprinting. They are
left to the caller to attach from MusicXML, or to a separate detector.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Import the local datasets/omr/pipeline.py directly. We avoid `from
# datasets.omr.pipeline import ...` because the HuggingFace `datasets`
# package shadows our local datasets/ folder on sys.path.
_OMR_PKG_DIR = Path(__file__).resolve().parents[5] / "datasets" / "omr"
sys.path.insert(0, str(_OMR_PKG_DIR))

from pipeline import extract_bar_nums_from_svg  # noqa: E402

_NS = "{http://www.w3.org/2000/svg}"

# Geometric thresholds (SVG user units, ~mm).
MIN_STAFF_LEN = 50.0      # staff lines on L7a pages are 99-108 mm
LINES_PER_STAFF = 5
STAFF_LINE_GAP = 1.2      # spacing between adjacent staff lines is ~1.0
INTER_STAFF_GAP = 7.0     # max gap between two staves in the same system
SYSTEM_GAP = 12.0         # vertical gap that splits one system from next
BARLINE_W_MIN = 0.15
BARLINE_W_MAX = 0.30      # single barline width
BARLINE_HEAVY_MIN = 0.40  # heavy / final barline
BARLINE_H_MIN = 3.5


def _parse_translate(transform: str) -> tuple[float, float]:
    if not transform:
        return 0.0, 0.0
    m = re.search(r"translate\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)", transform)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def _walk(elem, tx=0.0, ty=0.0):
    """Yield (tag, abs_tx, abs_ty, attrib) for every descendant."""
    dx, dy = _parse_translate(elem.attrib.get("transform", ""))
    ntx, nty = tx + dx, ty + dy
    yield elem.tag.replace(_NS, ""), ntx, nty, elem.attrib
    for child in elem:
        yield from _walk(child, ntx, nty)


def _collect_primitives(svg_path: Path) -> tuple[list, list]:
    """Return (horizontal_lines, vertical_rects) with absolute coords."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    h_lines: list[tuple[float, float, float]] = []  # (x1, x2, y)
    v_rects: list[tuple[float, float, float, float]] = []  # (x, y, w, h)
    for tag, tx, ty, attrs in _walk(root):
        if tag == "line":
            try:
                x1 = float(attrs.get("x1", 0)) + tx
                x2 = float(attrs.get("x2", 0)) + tx
                y1 = float(attrs.get("y1", 0)) + ty
                y2 = float(attrs.get("y2", 0)) + ty
            except ValueError:
                continue
            if abs(y1 - y2) < 0.01 and (x2 - x1) > MIN_STAFF_LEN:
                h_lines.append((x1, x2, y1))
        elif tag == "rect":
            try:
                x = float(attrs.get("x", 0)) + tx
                y = float(attrs.get("y", 0)) + ty
                w = float(attrs.get("width", 0))
                h = float(attrs.get("height", 0))
            except ValueError:
                continue
            v_rects.append((x, y, w, h))
    return h_lines, v_rects


def _group_staves(h_lines: list[tuple[float, float, float]]) -> list[dict]:
    """Cluster horizontal lines into staves (5 consecutive lines, gap ~1)."""
    if not h_lines:
        return []
    # Sort top-to-bottom, then left-to-right
    lines = sorted(h_lines, key=lambda l: (round(l[2], 1), l[0]))
    staves: list[dict] = []
    cur: list[tuple[float, float, float]] = []
    for ln in lines:
        if not cur:
            cur.append(ln)
            continue
        same_y = abs(ln[2] - cur[-1][2]) < 0.05 and ln[0] != cur[-1][0]
        # "same row" continuation (LilyPond may emit overlapping segments)
        if same_y:
            cur.append(ln)
            continue
        gap = ln[2] - cur[-1][2]
        if 0 < gap <= STAFF_LINE_GAP:
            cur.append(ln)
        else:
            staves.append(_finalize_staff(cur))
            cur = [ln]
    if cur:
        staves.append(_finalize_staff(cur))
    # Keep only 5-line staves
    return [s for s in staves if s["n_lines"] == LINES_PER_STAFF]


def _finalize_staff(lines: list[tuple[float, float, float]]) -> dict:
    # Each staff row may have multiple line segments (different x ranges).
    # Distinct y values = number of staff lines.
    ys = sorted({round(l[2], 2) for l in lines})
    x_min = min(l[0] for l in lines)
    x_max = max(l[1] for l in lines)
    y_min = min(ys)
    y_max = max(ys)
    return {
        "n_lines": len(ys),
        "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
        "y_top": y_min,
        "y_bottom": y_max,
        "x_left": x_min,
        "x_right": x_max,
    }


def _bimodal_gap_threshold(gaps: list[float],
                           min_jump: float = 2.0) -> float | None:
    """Find a threshold separating 'within-system' from 'between-system'
    staff gaps. Returns None if the distribution is unimodal (no clear
    break) and the caller should fall back to a fixed threshold.

    Strategy: sort gaps, find the largest jump between consecutive values.
    Only use it if that jump is significantly larger than the typical
    inter-gap step — otherwise the distribution is one cluster.
    """
    if len(gaps) < 2:
        return None
    s = sorted(gaps)
    jumps = [(s[i] - s[i-1], i) for i in range(1, len(s))]
    biggest_jump, idx = max(jumps)
    if biggest_jump < min_jump:
        return None
    typical = sum(j for j, _ in jumps) / len(jumps)
    if biggest_jump < 3 * typical:
        return None  # not enough contrast to call it bimodal
    return (s[idx] + s[idx-1]) / 2


def _group_systems(staves: list[dict],
                   staves_per_system: int | None = None) -> list[dict]:
    """Group staves into systems by vertical proximity.

    If `staves_per_system` is provided (typically from MusicXML's
    score-part / staves-per-part count), groups are formed by chunking —
    the most reliable method. Otherwise falls back to bimodal gap analysis,
    which works for typical solo/piano/quartet layouts but can mis-split
    scores where within-system gap exceeds between-system gap (e.g. SATB
    with piano, where the piano brace adds inter-staff space).
    """
    if not staves:
        return []
    if staves_per_system and staves_per_system > 0:
        groups = [staves[i:i + staves_per_system]
                  for i in range(0, len(staves), staves_per_system)]
        # Drop trailing partial group if it's smaller than expected
        # (probably misdetected stave fragments below the music).
        groups = [g for g in groups if len(g) == staves_per_system]
        split_at = None  # not used; we already have groups
    elif len(staves) >= 3:
        gaps = [staves[i+1]["y_top"] - staves[i]["y_bottom"]
                for i in range(len(staves) - 1)]
        bimodal = _bimodal_gap_threshold(gaps)
        split_at = bimodal if bimodal is not None else SYSTEM_GAP
    else:
        split_at = SYSTEM_GAP

    if staves_per_system and staves_per_system > 0:
        systems = groups
    else:
        systems: list[list[dict]] = []
        cur: list[dict] = [staves[0]]
        for s in staves[1:]:
            gap = s["y_top"] - cur[-1]["y_bottom"]
            if gap < split_at:
                cur.append(s)
            else:
                systems.append(cur)
                cur = [s]
        systems.append(cur)
    out: list[dict] = []
    for sys_idx, group in enumerate(systems):
        x_left = min(s["x_left"] for s in group)
        x_right = max(s["x_right"] for s in group)
        y_top = min(s["y_top"] for s in group)
        y_bottom = max(s["y_bottom"] for s in group)
        out.append({
            "idx": sys_idx,
            "bbox": (x_left, y_top, x_right - x_left, y_bottom - y_top),
            "staves": group,
        })
    return out


def _collect_barlines(
    v_rects: list[tuple[float, float, float, float]],
    systems: list[dict],
) -> list[dict]:
    """Identify barlines vs stems.

    Width alone doesn't separate them: in some scores (e.g. Ravel quartets)
    stems and single barlines are both ~0.15 mm wide. The reliable signal is
    that a barline's y-range covers a whole staff (top staff line through
    bottom staff line), whereas a stem extends out from a notehead and
    doesn't reach both extremes.
    """
    # Tolerance for staff-line snap: real barlines slightly overshoot the
    # outermost staff lines; stems within the staff fall short.
    STAFF_SNAP_TOL = 0.3  # mm
    # x-quantization: rects at the same musical x are sometimes emitted twice
    # (e.g. LilyPond doubles the end-of-line bar when a cautionary measure
    # number is shown after the line break). 1 mm absorbs that jitter while
    # still being well below typical measure widths (5-10 mm).
    X_BIN = 1.0  # mm

    staves_flat: list[tuple[int, int, dict]] = []
    sys_n_staves: dict[int, int] = {}
    for sysd in systems:
        sys_n_staves[sysd["idx"]] = len(sysd["staves"])
        for st_i, st in enumerate(sysd["staves"]):
            staves_flat.append((sysd["idx"], st_i, st))

    # Pass 1: keep rects that pass width + spans-staff checks. Tag each with
    # the staff it covers so we can do cross-staff confirmation in pass 2.
    candidates: list[tuple[int, int, float, float, float, float, bool]] = []
    for x, y, w, h in v_rects:
        if h < BARLINE_H_MIN:
            continue
        is_single = BARLINE_W_MIN <= w <= BARLINE_W_MAX
        is_heavy = w >= BARLINE_HEAVY_MIN and w < 1.0
        if not (is_single or is_heavy):
            continue
        bar_top, bar_bot = y, y + h
        for sys_idx, st_i, st in staves_flat:
            spans = (bar_top <= st["y_top"] + STAFF_SNAP_TOL
                     and bar_bot >= st["y_bottom"] - STAFF_SNAP_TOL)
            in_x = st["x_left"] - 1 <= x <= st["x_right"] + 1
            if spans and in_x:
                candidates.append((sys_idx, st_i, x, y, w, h, is_heavy))
                break

    # Pass 2: per (system, staff), cluster bars whose x are within X_BIN.
    # Each cluster collapses to one bar (the leftmost = canonical bar; the
    # cautionary echo, when present, is dropped). Clustering by distance
    # avoids the round() boundary problem that grid-binning has.
    from collections import defaultdict
    by_staff: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for cand in candidates:
        sys_idx, st_i, *_ = cand
        by_staff[(sys_idx, st_i)].append(cand)

    # Per-staff: walk sorted bars; flag pairs within X_BIN as a thin double
    # bar (||). A single isolated bar keeps its width-based kind. We extend
    # the candidate tuple with a `kind` field for downstream class output.
    tagged: list[tuple] = []  # (sys, st_i, x, y, w, h, is_heavy, kind)
    for cands in by_staff.values():
        cands.sort(key=lambda c: c[2])
        i = 0
        while i < len(cands):
            c = cands[i]
            nxt = cands[i + 1] if i + 1 < len(cands) else None
            if nxt and nxt[2] - c[2] < X_BIN:
                tagged.append((*c, "double"))
                tagged.append((*nxt, "double"))
                i += 2
            else:
                kind = "heavy" if c[6] else "single"
                tagged.append((*c, kind))
                i += 1

    # Pass 3: cross-staff confirmation. For multi-staff systems, a real
    # barline appears at similar x in >=2 staves; an isolated stem appears
    # at only one. We bucket by 1 mm x-bin for this check, with two passes
    # (offset by 0.5 mm) to avoid the boundary problem.
    def buckets(cands, offset):
        out: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        for c in cands:
            sys_idx, _, x, *_ = c
            out[(sys_idx, int((x + offset) / X_BIN))].append(c)
        return out

    confirmed: set = set()  # set of id(cand)
    for offset in (0.0, X_BIN / 2):
        for (sys_idx, _), cands in buckets(tagged, offset).items():
            staves_hit = {c[1] for c in cands}
            if sys_n_staves[sys_idx] == 1 or len(staves_hit) >= 2:
                for c in cands:
                    confirmed.add(id(c))

    barlines: list[dict] = []
    for c in tagged:
        if id(c) not in confirmed:
            continue
        _, _, x, y, w, h, is_heavy, kind = c
        barlines.append({
            "system": c[0],
            "bbox": (x, y, w, h),
            "heavy": is_heavy,
            "kind": kind,
        })
    return barlines


def extract_layout(svg_path: Path,
                   staves_per_system: int | None = None) -> dict:
    h_lines, v_rects = _collect_primitives(svg_path)
    staves = _group_staves(h_lines)
    systems = _group_systems(staves, staves_per_system=staves_per_system)
    barlines = _collect_barlines(v_rects, systems)
    bar_numbers = extract_bar_nums_from_svg(svg_path)

    # Per-system staff index (top-to-bottom within the system)
    staves_out = []
    for sysd in systems:
        for j, s in enumerate(sysd["staves"]):
            staves_out.append({"system": sysd["idx"], "idx": j, "bbox": s["bbox"]})

    # Key-sig area per system: top staff's first measure (clef + keysig slab).
    # Derived geometrically from staves + barlines so it stays in lockstep
    # with cell derivation. We reuse the same logic at make_dataset/eval
    # time via cells.derive_keysig_areas.
    sys_boxes = [s["bbox"] for s in systems]
    staff_boxes = [s["bbox"] for s in staves_out]
    bar_boxes = [b["bbox"] for b in barlines]
    try:
        # cells.py lives one dir up; importable at make_dataset call time.
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
        from cells import derive_keysig_areas  # noqa: E402
        key_sig_boxes = derive_keysig_areas(sys_boxes, staff_boxes, bar_boxes)
    except Exception:
        key_sig_boxes = []

    key_sigs = [{"system": i, "bbox": b} for i, b in enumerate(key_sig_boxes)]

    return {
        "systems": [{"idx": s["idx"], "bbox": s["bbox"],
                     "staves": s["staves"]} for s in systems],
        "staves": staves_out,
        "barlines": barlines,
        "key_sigs": key_sigs,
        "clefs": [],
        "bar_numbers": bar_numbers,
    }


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("svg")
    args = p.parse_args()
    out = extract_layout(Path(args.svg))
    # Compact print: counts + first few of each
    print(json.dumps({
        "n_systems": len(out["systems"]),
        "n_staves":  len(out["staves"]),
        "n_barlines": len(out["barlines"]),
        "bar_numbers": out["bar_numbers"],
        "systems":  out["systems"],
        "barlines_per_system": {
            s["idx"]: sum(1 for b in out["barlines"] if b["system"] == s["idx"])
            for s in out["systems"]
        },
    }, indent=2, default=str))
