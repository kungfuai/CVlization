"""DEPRECATED for its original purpose (MXC3 round-trip oracle) on
2026-05-21. Retained as a general musical-equivalence normalizer for
MXC2 — may still be useful for diff-based eval where engraving-detail
differences shouldn't count as errors.

Musical-equivalence normalizer for MXC2 output.

Used to compare original-XML vs round-trip-XML at the musical level,
ignoring cosmetic / engraving-metadata differences that don't change the
actual music:

  - Direction font styling brackets `[font-size=...,font-weight=...]`.
  - `v=N` voice-state tokens (MXC2 emits these statefully; the same music
    can have different v= patterns depending on how voices interleave).
  - Empty/placement-only `dir @above` / `dir @below` lines.
  - Direction ordering within the same time location (sorted within block).
  - `n0` explicit-natural marker collapsed when the note's alter would be 0
    under the key — these encode "courtesy natural was drawn" which is a
    rendering detail, not a musical content difference.

The goal: ``normalize(mxc2_a) == normalize(mxc2_b)`` iff the two MusicXMLs
represent the same music, irrespective of engraving choices.
"""

import re


_FONT_BRACKET_RE = re.compile(r"\[font-[^\]]*\]\s*")
_V_TOKEN_RE = re.compile(r"\bv=\d+\s*")


def normalize_mxc2(text: str) -> str:
    """Return a normalized MXC2 string for musical-equivalence comparison."""
    out = []
    for line in text.splitlines():
        if line.startswith("print"):
            # `print new-system` / `print new-page` are engraving layout
            # hints, not musical content.
            continue
        if line.startswith("dir"):
            line = _normalize_dir(line)
            if line is None:
                continue
        elif line.startswith("R "):
            line = _normalize_rest(line)
            # Strip v= from rests too — stateful voice emission can cause
            # cosmetic differences that aren't musical.
            line = re.sub(r"\s+v=\d+", "", line)
            # Also strip slur on rests (engraving artifact)
            line = re.sub(r"\s+slur\d+=\w+", "", line)
        elif line.startswith(("N ", "+N ", "gN ", "gR ")):
            line = _normalize_note(line)
            if line.startswith("+N"):
                line = re.sub(r"\s+(tie|tied)=\w+", "", line)
        out.append(line)
    out = _drop_fwd_bak_noops(out)
    out = _reorder_dir_bak(out)
    out = _canonicalize_barline_positions(out)
    return "\n".join(out)


def _canonicalize_barline_positions(lines: list) -> list:
    """Within each measure, move `bar=` lines with `loc=left` to immediately
    after the `M ` start. `loc=right` bars (or default) stay at the end.
    MXC2 emits barlines in source document order, which differs across
    encoders; canonicalizing makes the comparison robust."""
    # Find measure boundaries: lines starting with 'M ' (MXC2 measure header)
    measure_indices = [i for i, l in enumerate(lines) if l.startswith("M ")]
    if not measure_indices:
        return lines

    out = list(lines)
    # Process each measure's range
    for k, start in enumerate(measure_indices):
        end = measure_indices[k + 1] if k + 1 < len(measure_indices) else len(out)
        # Within [start+1, end): pull out bar= loc=left lines, place after M=
        left_bars = []
        rest = []
        for idx in range(start + 1, end):
            ln = out[idx]
            if ln.startswith("bar=") and "loc=left" in ln:
                left_bars.append(ln)
            else:
                rest.append(ln)
        new_slice = [out[start]] + left_bars + rest
        out[start:end] = new_slice
    return out


def _drop_fwd_bak_noops(lines: list) -> list:
    """Drop `fwd DUR ...` followed immediately by `bak DUR` (same DUR) —
    they advance and rewind time with no net change. Often used in
    MuseScore-engraved files for layout positioning; not musical content."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("fwd ") and i + 1 < len(lines) and lines[i + 1].startswith("bak "):
            fwd_dur = " ".join(line.split()[1:])
            # Drop voice/staff modifiers for comparison
            fwd_dur_only = re.sub(r"\s+(?:v=|st=)\S+", "", fwd_dur).strip()
            bak_dur = " ".join(lines[i + 1].split()[1:]).strip()
            if fwd_dur_only == bak_dur:
                i += 2
                continue
        out.append(line)
        i += 1
    return out


def _reorder_dir_bak(lines: list) -> list:
    """Within each measure (between M lines), reorder consecutive runs of
    `dir`/`bak` lines so `bak` always precedes `dir`. Handles the case
    where the encoder vs MusicXML may differ on whether the direction is
    emitted before or after a backup — musically equivalent."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("dir") or line.startswith("bak "):
            j = i
            run = []
            while j < len(lines) and (lines[j].startswith("dir") or
                                       lines[j].startswith("bak ")):
                run.append(lines[j])
                j += 1
            run.sort(key=lambda l: (0 if l.startswith("bak") else 1))
            out.extend(run)
            i = j
        else:
            out.append(line)
            i += 1
    return out


def _normalize_rest(line: str):
    """Collapse `R whole quarter dot ...` to `R whole`. MXC2 emits the
    hardcoded 'whole' for measure-rests *and* the `<type>` element if
    present, plus any `<dot>` tokens — all redundant for musical meaning
    (the measure-rest fills the measure regardless)."""
    parts = line.split()
    DUR_TYPES = {"whole", "half", "quarter", "eighth", "16th", "32nd",
                 "64th", "128th", "breve", "long", "maxima"}
    if len(parts) >= 3 and parts[0] == "R" and parts[1] == "whole":
        kept = parts[:2]
        for p in parts[2:]:
            stripped = p.rstrip(".")
            if stripped in DUR_TYPES or p == "dot" or p.startswith("dot="):
                continue
            kept.append(p)
        return " ".join(kept)
    return line


_NATURAL_PITCH_RE = re.compile(r"\b([A-G])n0(\d+)\b")
_ACC_ANY_RE = re.compile(r"\s+acc=\w+(?:-\w+)*\b")


def _normalize_dir(line: str):
    """Drop font brackets; drop placement-only directions."""
    # Strip font brackets
    line = _FONT_BRACKET_RE.sub("", line)
    line = re.sub(r"\s{2,}", " ", line).rstrip()
    # If the direction has no content beyond placement, drop the line
    parts = line.split()
    if len(parts) <= 2:  # 'dir' + maybe '@above' only
        # Check if any content tokens exist
        content = [p for p in parts[1:] if not p.startswith("@")]
        if not content:
            return None
    return line


def _normalize_note(line: str):
    """Drop v= tokens; collapse explicit-natural markers (Cn04 → C4) and
    `acc=X`. These are engraving conventions, not musical content.
    Also: sort multi-verse lyric tokens (L1, L2, …) into numeric order so
    document-order differences don't surface as diffs."""
    line = re.sub(r"\s+v=\d+", "", line)
    line = _NATURAL_PITCH_RE.sub(r"\1\2", line)
    line = _ACC_ANY_RE.sub("", line)
    # Sort lyric tokens L1:..., L2:... into numeric order
    parts = line.split()
    lyrics = [(i, p) for i, p in enumerate(parts)
              if re.match(r"^L\d+:", p)]
    if len(lyrics) > 1:
        sorted_l = sorted([p for _, p in lyrics],
                          key=lambda p: int(re.match(r"^L(\d+):", p).group(1)))
        for (i, _), new_p in zip(lyrics, sorted_l):
            parts[i] = new_p
        line = " ".join(parts)
    return line


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        print(normalize_mxc2(f.read()))
