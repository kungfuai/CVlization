"""Deterministic accidental re-spelling for MXC2 output.

The Level 7a investigation showed: the model often forms a confident-wrong
key belief and spells every key-implied accidental consistently with the
wrong key. Soft prompt injection of the correct key doesn't help — the
note-spelling pathway re-derives from image perception.

This module fixes that mechanically: given a trusted key from a separate
stage (a focused image→key=N classifier), it walks an MXC2 transcription
and re-spells every note that does NOT carry an explicit `acc=` marker
to match the trusted key. Notes with `acc=` are left alone — they're
explicit chromatic deviations and should be preserved.

Public API:
    respell_mxc2(mxc2: str, trusted_key: int|dict) -> str
        trusted_key can be:
          - int: one key applied to all measures
          - dict[(m_num_str, p_idx)] -> int: per-measure key
"""

import re


SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]

# Same as mxc2.py
ALTER_TO_SYM = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}


def key_implied_alter(step: str, fifths: int) -> int:
    if fifths > 0:
        return 1 if step in SHARP_ORDER[:fifths] else 0
    if fifths < 0:
        return -1 if step in FLAT_ORDER[:-fifths] else 0
    return 0


# Pitch token: <letter>(#|##|b|bb|n0)?<octave>
# Octave can be 0..9 (one digit, sometimes two for "n0" prefix making it like Fn02).
_PITCH_RE = re.compile(r"\b([A-G])(##|#|bb|b|n0)?(\d+)\b")


def _respell_pitch(pitch_tok: str, key: int) -> str:
    """Apply trusted key to a pitch token, producing a new pitch token.

    Whether the existing token has alter info or not, REPLACE the alter
    with the key-implied one. The caller has already verified that this
    note line has no `acc=` marker — meaning whatever alter is on the
    pitch was key-implied, not an explicit chromatic accidental. If the
    model formed a wrong key belief, it would emit a wrong alter here;
    we override.
    """
    m = re.match(r"^([A-G])(##|#|bb|b|n0)?(\d+)$", pitch_tok)
    if not m:
        return pitch_tok
    step, _alter_sym, octave = m.group(1), m.group(2), m.group(3)
    new_alter = key_implied_alter(step, key)
    new_sym = ALTER_TO_SYM.get(new_alter, "")
    return f"{step}{new_sym}{octave}"


def _has_acc_marker(line: str) -> bool:
    return re.search(r"\bacc=\w", line) is not None


def respell_mxc2(mxc2: str, trusted_key) -> str:
    """Re-spell non-explicit-acc notes in an MXC2 string using `trusted_key`.

    trusted_key: either an int (one key for whole document) or a dict
    {(measure_num_str, part_idx): key_int} for per-measure keys.
    """
    use_dict = isinstance(trusted_key, dict)
    lines = mxc2.splitlines()
    out = []
    cur_m = None
    cur_p = 0  # 1-indexed; bumps on `---`

    for raw_line in lines:
        line = raw_line
        stripped = line.strip()

        # Track part index
        if stripped == "---":
            cur_p += 1
            out.append(raw_line)
            continue

        # Track current measure
        if stripped.startswith("M "):
            parts = stripped.split()
            if len(parts) > 1:
                cur_m = parts[1]
            # Also update M-line key if we want — but the model already wrote a
            # key on this line. We trust the trusted_key for note spelling;
            # we also rewrite the M-line's key= to be consistent.
            key_now = trusted_key[(cur_m, cur_p)] if use_dict else trusted_key
            if key_now is not None:
                # Replace or insert key=N in this M line
                if re.search(r"\bkey=-?\d+\b", line):
                    line = re.sub(r"\bkey=-?\d+\b", f"key={key_now}", line)
                else:
                    # Insert key=N after the measure number
                    line = re.sub(r"^(\s*M \S+)", rf"\1 key={key_now}", line)
            out.append(line)
            continue

        # Note-like lines: rewrite pitch tokens that have no acc= marker
        if (stripped.startswith("N ") or stripped.startswith("+N ")
                or stripped.startswith("gN ")):
            key_now = trusted_key[(cur_m, cur_p)] if use_dict else trusted_key
            if key_now is None:
                out.append(raw_line)
                continue
            if _has_acc_marker(line):
                # Explicit accidental → leave alone
                out.append(raw_line)
                continue
            # Find the pitch token (typically the 2nd whitespace-separated tok)
            # but be defensive: scan all tokens, replace any that match _PITCH_RE.
            # MXC2 has at most one pitch per N/+N line.
            tokens = line.split()
            for i, t in enumerate(tokens):
                if i == 0:
                    continue  # skip leading 'N' / '+N' / 'gN'
                if re.match(r"^[A-G](##|#|bb|b|n0)?\d+$", t):
                    tokens[i] = _respell_pitch(t, key_now)
                    break  # only the first pitch token
            out.append(" ".join(tokens))
            continue

        out.append(raw_line)
    return "\n".join(out) + ("\n" if mxc2.endswith("\n") else "")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        text = open(sys.argv[1]).read()
        key = int(sys.argv[2])
        print(respell_mxc2(text, key))
    else:
        print("usage: respell.py MXC2_FILE KEY")
