"""Per-page keysig extractor v2.

Combines:
  - MusicXML <fifths>-by-measure -> {measure_num: fifths} for every change
  - SVG text labels at barlines -> {measure_num: (x, y)} on the page
  - System bboxes from layout

For each key-change measure that falls on this page:
  - If it's the first measure of a system: bbox = system's leading slab (as before)
  - Otherwise: bbox = small slab to the right of the barline preceding the
    key-change measure (= leftmost portion of the changed measure)
"""
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

NS = "{http://www.w3.org/2000/svg}"


def parse_mxl_key_changes(mxl_path: Path) -> dict[int, int]:
    """Returns {measure_num: fifths} for measures where the key changes
    (including measure 1's initial key)."""
    with zipfile.ZipFile(mxl_path) as z:
        for n in z.namelist():
            if n.endswith(".xml") and not n.startswith("META-INF"):
                text = z.read(n).decode("utf-8", errors="ignore")
                break
        else:
            return {}
    root = ET.fromstring(text)
    out: dict[int, int] = {}
    last = None
    # Use only the first part — global key changes apply across all parts.
    part = root.find(".//part")
    if part is None:
        return {}
    for m in part.findall("measure"):
        try:
            num = int(m.attrib.get("number", "0"))
        except ValueError:
            continue
        f_el = m.find(".//key/fifths")
        if f_el is None:
            continue
        try:
            f = int(f_el.text)
        except (ValueError, TypeError):
            continue
        if f != last:
            out[num] = f
            last = f
    return out


def parse_measure_label_positions(svg_path: Path) -> dict[int, tuple[float, float]]:
    """Returns {measure_num: (x_mm, y_mm)} for every visible bar-number label
    in the SVG. Used to locate measures on the rendered page."""
    def accum(t, base):
        if not t:
            return base
        m = re.search(r"translate\(([-\d.eE]+)[\s,]+([-\d.eE]+)\)", t)
        return (base[0] + float(m.group(1)), base[1] + float(m.group(2))) if m else base

    out: dict[int, tuple[float, float]] = {}
    root = ET.parse(svg_path).getroot()

    def walk(el, acc=(0.0, 0.0)):
        new = accum(el.attrib.get("transform"), acc)
        if el.tag == NS + "text":
            txt = (el.text or "").strip()
            for c in el:
                txt += (c.text or "") + (c.tail or "")
            txt = txt.strip()
            # Plain integer labels only — skip parenthesized cautionaries
            if re.fullmatch(r"\d+", txt):
                num = int(txt)
                x = float(el.attrib.get("x", 0)) + new[0]
                y = float(el.attrib.get("y", 0)) + new[1]
                # Keep the leftmost occurrence per number (per-system labels
                # repeat across staves; we want the topmost-leftmost)
                if num not in out or y < out[num][1]:
                    out[num] = (x, y)
        for c in el:
            walk(c, new)

    walk(root)
    return out


def extract_keysigs(svg_path: Path, mxl_path: Path, layout: dict,
                    keysig_width_mm: float = 8.0,
                    clef_offset_mm: float = 3.5) -> list[dict]:
    """Returns [{'bbox':..., 'fifths':..., 'measure':..., 'system':..., 'kind':...}].

    Two kinds of bboxes per page:
      - 'change'     : a <key><fifths> declaration mid-piece (often after ||)
      - 'line_start' : visual restatement of the current key at each system's
                       leftmost measure (no MusicXML <fifths> change there,
                       but LilyPond redraws the keysig after the clef).
    """
    key_changes = parse_mxl_key_changes(mxl_path)
    label_pos = parse_measure_label_positions(svg_path)
    sorted_changes = sorted(key_changes.items())

    def active_fifths_at(measure_num: int):
        active = None
        for m_chg, f in sorted_changes:
            if m_chg <= measure_num:
                active = f
            else:
                break
        return active

    out: list[dict] = []
    change_keys: set[tuple[int, int]] = set()
    # Pass 1: explicit mid-piece changes.
    for m_num, fifths in key_changes.items():
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
        bx = lx + 0.5
        by = sys_box[1]
        bw = keysig_width_mm
        bh = sys_box[3]
        out.append({
            "bbox": (bx, by, bw, bh),
            "fifths": fifths,
            "kind": "change",
            "measure": m_num,
            "system": sys_i,
        })
        change_keys.add((sys_i, m_num))

    # Pass 2: line-start restatement of the active key at the first measure
    # of each system. Skip systems where pass 1 already placed a 'change' box
    # at the same first measure.
    for s in layout["systems"]:
        sys_box = s["bbox"]
        sy, sh = sys_box[1], sys_box[3]
        in_sys = [(m, p) for m, p in label_pos.items()
                  if sy - 5 <= p[1] <= sy + sh]
        if not in_sys:
            continue
        first_m, _ = min(in_sys, key=lambda kv: kv[0])
        if (s["idx"], first_m) in change_keys:
            continue
        active = active_fifths_at(first_m)
        if active is None:
            continue
        out.append({
            "bbox": (sys_box[0] + clef_offset_mm, sy, keysig_width_mm, sh),
            "fifths": active,
            "kind": "line_start",
            "measure": first_m,
            "system": s["idx"],
        })
    return out


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 3:
        print("usage: keysig_extractor.py <mxl> <svg>", file=sys.stderr)
        sys.exit(1)
    mxl, svg = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, os.path.dirname(__file__))
    from extract_bboxes import extract_layout
    layout = extract_layout(svg)
    ks = extract_keysigs(svg, mxl, layout)
    print(f"{len(ks)} keysigs:")
    for k in ks:
        print(f"  measure {k['measure']:3d} fifths={k['fifths']:+d}  "
              f"kind={k['kind']:10s} sys={k['system']} bbox={k['bbox']}")
