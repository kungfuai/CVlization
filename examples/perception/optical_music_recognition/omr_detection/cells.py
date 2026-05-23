"""Derive measure cells from detector output.

A *cell* is one measure of one staff -- the unit a per-cell transcription
model operates on. Cells are NOT detected directly; each is the staff
bbox intersected with the horizontal span between two consecutive measure
boundaries. Measure boundaries come from barline detections, clustered by
x-position (a measure boundary in a multi-staff system has one barline
per staff at nearly the same x).

Pure geometry -- no torch / ultralytics dependency, so it is importable
by both eval_detector.py and pipeline.py.
"""

from dataclasses import dataclass

Box = tuple[float, float, float, float]  # (x, y, w, h) top-left, page px


@dataclass
class Cell:
    system: int    # system index on the page, top-to-bottom
    staff: int     # staff index within the system, top-to-bottom
    measure: int   # measure index within the system, left-to-right (0-based)
    bbox: Box      # (x, y, w, h) in page pixels


def _center_x(box: Box) -> float:
    return box[0] + box[2] / 2


def _v_overlap_frac(a: Box, b: Box) -> float:
    """Fraction of box a's height that overlaps box b vertically."""
    ay0, ay1 = a[1], a[1] + a[3]
    by0, by1 = b[1], b[1] + b[3]
    ov = max(0.0, min(ay1, by1) - max(ay0, by0))
    return ov / max(a[3], 1e-6)


def _cluster_1d(values: list[float], tol: float) -> list[float]:
    """Group sorted scalars whose neighbours are within tol; return centers."""
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def derive_cells(systems: list[Box], staves: list[Box], barlines: list[Box],
                 x_tol_frac: float = 0.012) -> list[Cell]:
    """Derive measure cells from detector boxes.

    Args:
        systems / staves / barlines: lists of (x, y, w, h) in page pixels.
            barlines may mix single + heavy -- both are measure boundaries.
        x_tol_frac: barline x-clustering tolerance, as a fraction of the
            system's right edge. Barlines closer than this collapse to one
            measure boundary.

    Returns:
        list[Cell], ordered by (system, staff, measure).
    """
    systems = sorted(systems, key=lambda b: b[1])  # top-to-bottom
    cells: list[Cell] = []

    for sys_i, sysb in enumerate(systems):
        sys_staves = sorted(
            (s for s in staves if _v_overlap_frac(s, sysb) > 0.5),
            key=lambda b: b[1])
        if not sys_staves:
            continue
        sys_bars = [b for b in barlines if _v_overlap_frac(b, sysb) > 0.4]

        right_edge = max(s[0] + s[2] for s in sys_staves)
        tol = x_tol_frac * right_edge
        bar_x = _cluster_1d([_center_x(b) for b in sys_bars], tol)

        for st_i, (sx, sy, sw, sh) in enumerate(sys_staves):
            # Measure boundaries within this staff's horizontal span.
            bounds = [bx for bx in bar_x if sx - tol <= bx <= sx + sw + tol]
            # The first measure opens at the staff's left edge.
            if not bounds or bounds[0] - sx > tol:
                bounds = [sx] + bounds
            # Close the final measure at the staff's right edge if the last
            # detected barline fell short (a missed final/heavy barline).
            if bounds[-1] < sx + sw - tol:
                bounds = bounds + [sx + sw]
            for m_i in range(len(bounds) - 1):
                x0, x1 = bounds[m_i], bounds[m_i + 1]
                cells.append(Cell(sys_i, st_i, m_i, (x0, sy, x1 - x0, sh)))

    return cells


def measures_per_system(cells: list[Cell]) -> dict[int, int]:
    """Count distinct measures per system (= cells of its first staff)."""
    out: dict[int, int] = {}
    for c in cells:
        if c.staff == 0:
            out[c.system] = out.get(c.system, 0) + 1
    return out


@dataclass
class Measure:
    """One measure on the page, covering all staves of its system.

    More natural transcription unit than Cell -- a measure crop shows
    vertically-stacked staves of one part-group at once, mirroring how
    the safckylj VLM was trained (whole-page input).
    """
    system: int    # system index on the page
    measure: int   # measure index within the system (0-based)
    bbox: Box      # (x, y, w, h) spanning all staves at this measure


def derive_measures(systems: list[Box], staves: list[Box],
                    barlines: list[Box],
                    x_tol_frac: float = 0.012) -> list[Measure]:
    """Like derive_cells but one box per measure (across all staves)."""
    cells = derive_cells(systems, staves, barlines, x_tol_frac)
    if not cells:
        return []
    # Group by (system, measure); union vertical extent across staves.
    grouped: dict[tuple[int, int], list[Cell]] = {}
    for c in cells:
        grouped.setdefault((c.system, c.measure), []).append(c)
    out: list[Measure] = []
    for (sys_i, m_i), members in grouped.items():
        x0 = min(c.bbox[0] for c in members)
        y0 = min(c.bbox[1] for c in members)
        x1 = max(c.bbox[0] + c.bbox[2] for c in members)
        y1 = max(c.bbox[1] + c.bbox[3] for c in members)
        out.append(Measure(sys_i, m_i, (x0, y0, x1 - x0, y1 - y0)))
    out.sort(key=lambda m: (m.system, m.measure))
    return out
