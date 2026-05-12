"""Compress Audiveris MXC2 hints by stripping unreliable / non-essential markup.

What we KEEP (Audiveris is reasonably good at these):
  - Part declarations (P1, P2)
  - Measure boundaries (M N)
  - Key/time/clef per measure (key=..., time=..., clef=...)
  - Note pitches and durations (N C4 quarter)
  - Chord additions (+N D4)
  - Rests (R whole/half/...)

What we DROP (Audiveris errors here, or not useful):
  - Stem direction (su, sd)
  - Articulations (art=staccato, art=...)
  - Slurs/ties (slur1=start, tie=start, tied=continue)
  - Dynamics directions (dir @below dyn=p)
  - Voice markers (v=1)
  - Backup commands (bak half dot)
  - Print/system formatting (print new-system)
  - Duplicate part-name labels (e.g. "Voice Voice")

Typical size reduction: 50-60%.
"""

import re


_ART = re.compile(r"\s+art=[a-zA-Z\-]+")
_SLUR = re.compile(r"\s+(?:slur\d*=[a-zA-Z]+|tie=[a-zA-Z]+|tied=[a-zA-Z]+|slur=[a-zA-Z]+)")
_VOICE = re.compile(r"\s+v=\d+")
_STEM = re.compile(r"\s+s[ud]\b")
_DIR_LINE = re.compile(r"^\s*dir\b.*$", re.MULTILINE)
_PRINT_LINE = re.compile(r"^\s*print\b.*$", re.MULTILINE)
_BAK_LINE = re.compile(r"^\s*bak\b.*$", re.MULTILINE)


def compress_hint(mxc2: str) -> str:
    """Strip unreliable / non-essential markup from an Audiveris MXC2 hint."""
    # Drop entire noise lines
    text = _DIR_LINE.sub("", mxc2)
    text = _PRINT_LINE.sub("", text)
    text = _BAK_LINE.sub("", text)

    # Drop per-token noise (right side of each line)
    text = _ART.sub("", text)
    text = _SLUR.sub("", text)
    text = _VOICE.sub("", text)
    text = _STEM.sub("", text)

    # De-duplicate "Voice Voice" / "Piano Piano" in part declarations
    text = re.sub(r"\b(Voice) (Voice)\b", r"\1", text)
    text = re.sub(r"\b(Piano) (Piano)\b", r"\1", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


if __name__ == "__main__":
    import sys
    import json
    import statistics

    if len(sys.argv) < 2:
        print("Usage: python hint_compress.py <jsonl> [--sample]")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    sample = "--sample" in sys.argv

    orig_lens = []
    comp_lens = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("audiveris_failed") or not r.get("audiveris_mxc2"):
                continue
            orig = r["audiveris_mxc2"]
            comp = compress_hint(orig)
            orig_lens.append(len(orig))
            comp_lens.append(len(comp))

            if sample and len(orig_lens) == 1:
                print("=== ORIGINAL (first 600 chars) ===")
                print(orig[:600])
                print("\n=== COMPRESSED (first 600 chars) ===")
                print(comp[:600])
                print()

    n = len(orig_lens)
    if n == 0:
        print("No usable hints")
        sys.exit(0)
    avg_orig = sum(orig_lens) / n
    avg_comp = sum(comp_lens) / n
    ratio = avg_comp / avg_orig
    p90_orig = sorted(orig_lens)[int(n*0.9)]
    p90_comp = sorted(comp_lens)[int(n*0.9)]
    max_orig = max(orig_lens)
    max_comp = max(comp_lens)
    print(f"Hints: n={n}")
    print(f"  orig avg={avg_orig:.0f}  p90={p90_orig}  max={max_orig}")
    print(f"  comp avg={avg_comp:.0f}  p90={p90_comp}  max={max_comp}")
    print(f"  reduction: {1-ratio:.1%}  ({avg_comp/avg_orig:.2f}x)")
