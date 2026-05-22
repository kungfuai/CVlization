"""DEPRECATED — see mxc3.py for context. Kept for reference only.

For the active per-measure pipeline, use `mxc2_slice.py` (stateless
MXC2-based slicer) once it lands.

MXC3 slicers — extract per-measure / per-aspect targets.

All operations work on flat MXC3 text. Used to build training targets
for per-measure transcription and per-aspect queries.

Public API:
    extract_header(mxc3) -> str
    extract_measure(mxc3, m, p) -> str
    extract_aspect(mxc3, m, p, aspect) -> str    # aspect ∈ {pos, dur, stem, acc, key, time, clef}
    iter_measures(mxc3) -> Iterator[(m, p, block_text)]
    measure_keys(mxc3) -> dict[(m, p), key_int]
"""
import re
from typing import Iterator


_HEADER_RE = re.compile(r"^HEADER\b.*?(?=^M=)", re.MULTILINE | re.DOTALL)
_BLOCK_START_RE = re.compile(r"^M=(\d+) P=(\d+)\b", re.MULTILINE)
_HEADER_FIELD_RE = re.compile(r"(\w[\w-]*)=(\S+)")


def extract_header(mxc3: str) -> str:
    """Return the HEADER + P= lines, stripped of trailing blanks."""
    m = _HEADER_RE.search(mxc3)
    if m:
        return m.group(0).rstrip() + "\n"
    return ""


def iter_measures(mxc3: str) -> Iterator[tuple[int, int, str]]:
    """Yield (measure_number, part_number, block_text) for every measure."""
    matches = list(_BLOCK_START_RE.finditer(mxc3))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(mxc3)
        block = mxc3[start:end].rstrip()
        yield int(m.group(1)), int(m.group(2)), block


def extract_measure(mxc3: str, m: int, p: int) -> str:
    """Return the measure block for M=m P=p (header line + aspect lines)."""
    for mm, pp, block in iter_measures(mxc3):
        if mm == m and pp == p:
            return block + "\n"
    raise KeyError(f"measure M={m} P={p} not found")


def extract_aspect(mxc3: str, m: int, p: int, aspect: str) -> str:
    """Return one aspect of a measure.

    aspect ∈ {'pos', 'dur', 'stem', 'acc'} — the channel content (no prefix).
    aspect ∈ {'key', 'time', 'clef'} — the header field value.
    aspect == 'header' — full header line.
    """
    block = extract_measure(mxc3, m, p).splitlines()
    if not block:
        return ""
    header_line = block[0]
    if aspect == "header":
        return header_line
    if aspect in ("key", "time", "clef"):
        fields = dict(_HEADER_FIELD_RE.findall(header_line))
        return fields.get(aspect, "")
    pref = aspect + ":"
    for line in block[1:]:
        s = line.lstrip()
        if s.startswith(pref):
            return s[len(pref):].strip()
    return ""  # aspect line absent (e.g. no acc overrides)


def measure_keys(mxc3: str) -> dict:
    """Map (m, p) -> int key for every measure. Useful for stitching/eval."""
    out = {}
    for mm, pp, block in iter_measures(mxc3):
        header = block.splitlines()[0]
        fields = dict(_HEADER_FIELD_RE.findall(header))
        if "key" in fields:
            try:
                out[(mm, pp)] = int(fields["key"])
            except ValueError:
                pass
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        mxc3 = open(sys.argv[1]).read()
        if len(sys.argv) == 2:
            print(extract_header(mxc3))
            for mm, pp, _ in iter_measures(mxc3):
                print(f"  M={mm} P={pp}")
        elif len(sys.argv) == 4:
            m, p = int(sys.argv[2]), int(sys.argv[3])
            print(extract_measure(mxc3, m, p))
        elif len(sys.argv) == 5:
            m, p, aspect = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
            print(extract_aspect(mxc3, m, p, aspect))
