"""Stateless per-measure slicer for MXC2.

MXC2 is stateful: voice/staff/stem are emitted only on change, and
key/time/clef appear on the M line only when they change. For per-measure
SFT we need each measure to be a self-contained target — the model
shouldn't need to "remember" prior measures to interpret a slice.

This module walks an MXC2 document, tracks active state, and produces
slices where each requested measure's M line has key/time/clef restated
and the first note of each (voice, staff) within the slice carries
explicit `v=`/`st=` tags.

Public API:
    slice_measure(mxc2: str, m_num: int|str, p_idx: int) -> str
    iter_measures(mxc2: str) -> Iterator[(part_idx, measure_num, slice_text)]
    n_parts(mxc2: str) -> int
"""

import re
from typing import Iterator, Tuple


# Attribute keys MXC2 emits on the M line — capture all so we can restate.
_M_ATTR_KEYS = ("key", "time", "clef", "mode", "staves")


def _parse_m_line(line: str) -> dict:
    """Parse `M 5 key=2 time=4/4 clef=G2 clef2=F4 staves=2` → dict."""
    parts = line.split()
    # parts[0] == "M", parts[1] == measure number
    out = {"_M": parts[0], "_num": parts[1] if len(parts) > 1 else "0", "_extra": {}}
    for tok in parts[2:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out["_extra"][k] = v
        else:
            out["_extra"][tok] = ""
    return out


def _format_m_line(num, attrs: dict) -> str:
    """Reconstruct an M line from accumulated state."""
    toks = [f"M {num}"]
    # Emit in canonical order matching mxc2.py: key, mode, time, clef, clef2..., staves
    if "key" in attrs:
        toks.append(f"key={attrs['key']}")
    if "mode" in attrs:
        toks.append(f"mode={attrs['mode']}")
    if "time" in attrs:
        toks.append(f"time={attrs['time']}")
    # Clefs ordered: clef, clef2, clef3, ...
    clef_keys = sorted([k for k in attrs if k.startswith("clef")],
                       key=lambda k: (len(k), k))
    for ck in clef_keys:
        toks.append(f"{ck}={attrs[ck]}")
    if "staves" in attrs:
        toks.append(f"staves={attrs['staves']}")
    return " ".join(toks)


def _force_voice_staff_on_first_note(slice_lines: list, prior_voice, prior_staff) -> list:
    """If the prior state for (voice, staff) was set, re-assert them on the
    first note line that doesn't already specify. This makes the slice
    self-contained — a downstream re-encoder of just this slice would
    correctly tag voice/staff."""
    out = list(slice_lines)
    # Track within-slice state
    cur_voice = None
    cur_staff = None
    for i, line in enumerate(out):
        stripped = line.strip()
        if not stripped:
            continue
        # Only act on note-like lines (N, +N, gN, R, gR)
        if not (stripped.startswith("N ") or stripped.startswith("+N ")
                or stripped.startswith("gN ") or stripped.startswith("R ")
                or stripped.startswith("gR ") or stripped.startswith("+R ")):
            continue
        # Has v= / st=?
        has_v = re.search(r"\bv=\d+\b", stripped) is not None
        has_s = re.search(r"\bst=\d+\b", stripped) is not None
        # If this is the first note in slice and the prior state was set,
        # force v=/st= to be present.
        additions = []
        if not has_v and prior_voice and prior_voice not in (cur_voice,):
            additions.append(f"v={prior_voice}")
            cur_voice = prior_voice
        if not has_s and prior_staff and prior_staff not in (cur_staff,):
            additions.append(f"st={prior_staff}")
            cur_staff = prior_staff
        if additions:
            out[i] = stripped + " " + " ".join(additions)
        # Update cur_voice/cur_staff from any tokens on this line
        mv = re.search(r"\bv=(\d+)\b", out[i])
        if mv:
            cur_voice = mv.group(1)
        ms = re.search(r"\bst=(\d+)\b", out[i])
        if ms:
            cur_staff = ms.group(1)
        # First note processed — break (we only force on the first eligible note)
        if additions:
            break
        # No additions and no v=/st= — still consider it the "first note seen"
        # and break so we don't over-augment subsequent notes
        break
    return out


def _walk_parts(mxc2: str):
    """Walk MXC2 text, yielding part boundaries and tracking attr/voice/staff state.

    Yields tuples (part_idx, current_state, lines_in_part_so_far) at every
    measure boundary so callers can extract a measure with its prior state.
    """
    lines = mxc2.splitlines()
    # First, locate part boundaries: lines matching part-id pattern after `---`.
    # MXC2 emits `---` then the part id line then measures.
    return lines


def n_parts(mxc2: str) -> int:
    """Count parts in an MXC2 document."""
    return mxc2.count("\n---\n") + (1 if "\n---\n" in mxc2 or "---" in mxc2 else 0)


def slice_measure(mxc2: str, m_num, p_idx: int) -> str:
    """Extract a self-contained slice of measure `m_num` from part `p_idx` (1-indexed).

    Raises KeyError if not found.
    """
    target_num = str(m_num)
    lines = mxc2.splitlines()

    # Walk lines: find the start of part p_idx. Parts are separated by `---`.
    # The first part starts after the first `---`. So part_idx = count of `---`
    # seen so far. (Header before any `---` is part 0 in convention; first
    # actual part is p_idx=1.)
    i = 0
    cur_part = 0
    # Skip to start of part p_idx
    while i < len(lines):
        if lines[i].strip() == "---":
            cur_part += 1
            i += 1
            if cur_part == p_idx:
                break
        else:
            i += 1
    if cur_part < p_idx:
        raise KeyError(f"part {p_idx} not found")

    # Now i points at the line after the `---` for our part — typically the
    # part-id line. Skip it.
    if i < len(lines) and not lines[i].startswith("M "):
        i += 1  # skip part-id line

    # Walk measures of this part, tracking state.
    attrs = {}          # active key/time/clef/...
    cur_voice = None    # last v= emitted
    cur_staff = None    # last st= emitted
    pending_pre = []    # any extra lines (bar=, fwd, bak, dir) before next M
    target_start = None
    target_attrs_at_start = None
    target_voice_at_start = None
    target_staff_at_start = None

    while i < len(lines):
        line = lines[i]
        if line.strip() == "---":
            break  # next part
        if line.startswith("M "):
            parsed = _parse_m_line(line)
            # Update active attrs from this M line (only the keys present)
            for k, v in parsed["_extra"].items():
                attrs[k] = v
            if parsed["_num"] == target_num:
                target_start = i
                target_attrs_at_start = dict(attrs)
                target_voice_at_start = cur_voice
                target_staff_at_start = cur_staff
                # Find end of this measure (next M line or next ---)
                j = i + 1
                while j < len(lines):
                    ln = lines[j]
                    if ln.startswith("M ") or ln.strip() == "---":
                        break
                    j += 1
                end = j
                # Build the slice: M line restated + lines i+1..end-1
                m_line = _format_m_line(target_num, target_attrs_at_start)
                body = lines[i + 1:end]
                # Force v=/st= on first eligible note within body
                body = _force_voice_staff_on_first_note(
                    body, target_voice_at_start, target_staff_at_start
                )
                return "\n".join([m_line] + body) + "\n"
            i += 1
            continue
        # Track voice/staff state from any N/+N/R lines
        mv = re.search(r"\bv=(\d+)\b", line)
        if mv:
            cur_voice = mv.group(1)
        ms = re.search(r"\bst=(\d+)\b", line)
        if ms:
            cur_staff = ms.group(1)
        i += 1

    raise KeyError(f"measure {m_num} not found in part {p_idx}")


def iter_measures(mxc2: str) -> Iterator[Tuple[int, str, str]]:
    """Yield (part_idx, measure_num, slice_text) for every measure in every part.

    part_idx is 1-indexed.
    """
    lines = mxc2.splitlines()
    # Determine part starts (after each `---`)
    part_starts = []
    for i, l in enumerate(lines):
        if l.strip() == "---":
            part_starts.append(i + 1)
    if not part_starts:
        return

    for p_idx, start in enumerate(part_starts, 1):
        # Walk this part to find measure numbers
        i = start
        # Skip part-id line if not an M line
        if i < len(lines) and not lines[i].startswith("M "):
            i += 1
        # Collect measure numbers in this part
        seen_nums = []
        while i < len(lines):
            if lines[i].strip() == "---":
                break
            if lines[i].startswith("M "):
                num = lines[i].split()[1] if len(lines[i].split()) > 1 else "0"
                seen_nums.append(num)
            i += 1
        for m_num in seen_nums:
            try:
                yield p_idx, m_num, slice_measure(mxc2, m_num, p_idx)
            except KeyError:
                continue


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        text = open(sys.argv[1]).read()
        print(slice_measure(text, sys.argv[2], int(sys.argv[3])))
    elif len(sys.argv) == 2:
        text = open(sys.argv[1]).read()
        for p, m, s in iter_measures(text):
            print(f"=== P={p} M={m} ===")
            print(s)
    else:
        print("usage: mxc2_slice.py FILE [M P]")
