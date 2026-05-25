"""Clef + time-signature extractors.

Same MusicXML-driven pattern as keysig_extractor: parse declarations from
MusicXML, look up positions on the rendered page, emit bboxes.

  Clefs:    one per staff per system (G2/F4/C3 etc.); mid-piece changes rare.
  Time sig: one per system (typically on top staff); mid-piece changes rare.
"""
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_clefs_by_part_staff(mxl_path: Path) -> dict[tuple[str, int], str]:
    """{(part_id, staff_idx_1based): 'G2'|'F4'|...} for the FIRST clef of
    each (part, staff). Mid-piece clef changes can be added similarly later."""
    with zipfile.ZipFile(mxl_path) as z:
        for n in z.namelist():
            if n.endswith(".xml") and not n.startswith("META-INF"):
                root = ET.fromstring(z.read(n).decode("utf-8", errors="ignore"))
                break
        else:
            return {}
    out: dict[tuple[str, int], str] = {}
    for part in root.findall(".//part"):
        pid = part.attrib.get("id", "")
        for measure in part.findall("measure"):
            for clef in measure.findall(".//clef"):
                staff = int(clef.attrib.get("number", "1"))
                sign = clef.findtext("sign", "G") or "G"
                line = clef.findtext("line", "2") or "2"
                key = (pid, staff)
                if key not in out:
                    out[key] = f"{sign}{line}"
            if out:  # only first measure with clefs
                break
    return out


def parse_time_signatures_by_measure(mxl_path: Path) -> dict[int, str]:
    """{measure_num: 'beats/beat_type'} for measures that declare or change
    the time signature."""
    with zipfile.ZipFile(mxl_path) as z:
        for n in z.namelist():
            if n.endswith(".xml") and not n.startswith("META-INF"):
                root = ET.fromstring(z.read(n).decode("utf-8", errors="ignore"))
                break
        else:
            return {}
    out: dict[int, str] = {}
    part = root.find(".//part")
    if part is None:
        return {}
    last = None
    for m in part.findall("measure"):
        try:
            num = int(m.attrib.get("number", "0"))
        except ValueError:
            continue
        beats = m.findtext(".//time/beats")
        beat_type = m.findtext(".//time/beat-type")
        if beats and beat_type:
            ts = f"{beats}/{beat_type}"
            if ts != last:
                out[num] = ts
                last = ts
    return out


def extract_clefs(svg_path: Path, mxl_path: Path, layout: dict,
                  clef_width_mm: float = 5.0) -> list[dict]:
    """One bbox per staff per system at line-start. Clef label = e.g. 'G2'.

    Uses MusicXML to know which clef each staff in each system carries;
    uses extract_bboxes' systems/staves geometry to place the boxes.
    """
    clefs = parse_clefs_by_part_staff(mxl_path)
    # Flatten: ordered list of clef strings, one per staff per system.
    # Parts appear in document order; staves within a part in number order.
    ordered: list[str] = []
    for (pid, staff), label in sorted(clefs.items()):
        ordered.append(label)

    out: list[dict] = []
    for s in layout["systems"]:
        staves = s["staves"]
        if len(staves) != len(ordered):
            # Mismatch — fall back to "treble for all" rather than skip
            ordered_eff = ["G2"] * len(staves)
        else:
            ordered_eff = ordered
        for st_i, (st, label) in enumerate(zip(staves, ordered_eff)):
            out.append({
                "bbox": (st["x_left"], st["y_top"],
                         clef_width_mm, st["y_bottom"] - st["y_top"]),
                "clef": label,
                "system": s["idx"],
                "staff": st_i,
            })
    return out


def extract_time_signatures(svg_path: Path, mxl_path: Path, layout: dict,
                            timesig_width_mm: float = 5.0,
                            offset_after_keysig_mm: float = 12.0) -> list[dict]:
    """One bbox per system, spanning full system height, placed after where
    clef+keysig end. Plus mid-piece changes if any exist.

    Currently emits a line-start time sig for every system (modern engraving
    typically only shows it at the FIRST system of the piece + at changes,
    but for the OMR detector we err on the side of including it everywhere
    — the visual marker is the same).
    """
    # Imports kept local to avoid hard dep at module-import time.
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from keysig_extractor import (parse_measure_label_positions,
                                   parse_mxl_key_changes)
    ts_changes = parse_time_signatures_by_measure(mxl_path)
    label_pos = parse_measure_label_positions(svg_path)

    out: list[dict] = []
    # Mid-piece ts changes (skip the initial one for now — it's covered by
    # the per-system pass below).
    initial = min(ts_changes) if ts_changes else None
    for m_num, ts in ts_changes.items():
        if m_num == initial:
            continue
        if m_num not in label_pos:
            continue
        lx, ly = label_pos[m_num]
        sys_i = None
        for s in layout["systems"]:
            sx, sy, sw, sh = s["bbox"]
            if sy - 5 <= ly <= sy + sh:
                sys_i = s["idx"]
                break
        if sys_i is None:
            continue
        sys_box = layout["systems"][sys_i]["bbox"]
        out.append({
            "bbox": (lx + 0.5, sys_box[1], timesig_width_mm, sys_box[3]),
            "time": ts,
            "kind": "change",
            "measure": m_num,
            "system": sys_i,
        })

    # Line-start time sig — only at the very first system of the piece
    # (downstream consumers can drop it if not wanted; rendering convention
    # is "show time sig at start of piece, but not at every line").
    if ts_changes and layout["systems"]:
        first_sys = layout["systems"][0]
        sys_box = first_sys["bbox"]
        ts = ts_changes[initial]
        out.append({
            "bbox": (sys_box[0] + offset_after_keysig_mm,
                     sys_box[1], timesig_width_mm, sys_box[3]),
            "time": ts,
            "kind": "line_start",
            "measure": initial,
            "system": first_sys["idx"],
        })
    return out
