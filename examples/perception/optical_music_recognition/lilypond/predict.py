#!/usr/bin/env python3
"""
LilyPond Score Renderer

Converts sheet music notation (kern, MusicXML, or native LilyPond) to
high-quality engraved score images (PNG/PDF/SVG) using LilyPond's
Emmentaler font — the same engraving style used in 19th-century printed scores.

Supported input formats:
  **kern / bekern  (.krn, .txt)  — via music21 humdrum parser
  MusicXML         (.xml, .mxl)  — via music21 MusicXML parser
  LilyPond         (.ly)         — passed directly to lilypond binary

Output formats: png (default), pdf, svg

Usage:
  python predict.py
  python predict.py --input score.krn --format pdf
  python predict.py --input score.xml --output rendered/
  python predict.py --input score.ly  --format svg
"""

import os
import sys
import logging
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("music21").setLevel(logging.ERROR)

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import music21
from music21 import converter

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False
    def resolve_input_path(p, **kw):  return p
    def resolve_output_path(p=None, default_filename="score.png", **kw): return p or default_filename

# ---------------------------------------------------------------------------
# Sample: a short piano piece embedded as kern (no download needed for demo)
# ---------------------------------------------------------------------------
SAMPLE_KERN = """\
!!!OTL: Andante in C
!!!COM: Traditional
**kern\t**kern
*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
*MM72\t*MM72
=1\t=1
4C 4E 4G\t4e
4F 4A\t4f
4G 4B\t4g
4C 4E 4G\t4e
=2\t=2
4F 4A\t4.f
4G 4B\t8e
4A 4C\t4d
4F 4A\t4c
=3\t=3
4G 4B\t4d
4A 4C\t4e
4F 4A\t4f
2G 2B\t2g
=4\t=4
1C 1E 1G\t1c
==\t==
*-\t*-
"""

CACHE_DIR = Path.home() / ".cache" / "huggingface" / "cvl_data" / "lilypond"

# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def detect_format(path: Path) -> str:
    """Detect input format from file extension."""
    ext = path.suffix.lower()
    if ext in (".krn", ".kern"):
        return "kern"
    if ext in (".xml", ".mxl", ".musicxml"):
        return "musicxml"
    if ext == ".ly":
        return "lilypond"
    # Peek at content for ambiguous extensions (.txt etc.)
    try:
        head = path.read_text(errors="ignore")[:200]
        if "**kern" in head or "**ekern" in head or "**dynam" in head:
            return "kern"
        if "<?xml" in head and "score-partwise" in head:
            return "musicxml"
        if r"\version" in head or r"\score" in head or r"\relative" in head:
            return "lilypond"
    except Exception:
        pass
    return "kern"  # default guess


def strip_bekern(kern: str) -> str:
    """Strip bekern encoding artifacts (@, ·) that confuse music21."""
    return kern.replace("@", "").replace("·", "")


# ---------------------------------------------------------------------------
# Conversion: kern / MusicXML → LilyPond .ly
# ---------------------------------------------------------------------------

def _inject_kern_metadata(ly_path: Path, kern_text: str) -> None:
    """
    Inject kern metadata that music21 silently drops into the generated .ly:
      - !!!COM:  → composer field in \\header
      - !!!OMD:  → \\tempo marking before measure 1
      - *>Label  → \\mark \\markup { "Label" } before each section's first measure
    """
    ly = ly_path.read_text()
    kern_lines = kern_text.split("\n")

    # --- composer ---
    for line in kern_lines:
        if line.startswith("!!!COM:"):
            composer = line[7:].strip().replace('"', '\\"')
            ly = re.sub(r'(\\header\s*\{)',
                        r'\1\n  composer = "' + composer + '"',
                        ly, count=1)
            break

    # --- tempo (!!!OMD:) and section labels (*>Label) ---
    # First pass: collect (measure_number, label) pairs and tempo text.
    # A *> record appearing before =N means "inject before measure N".
    tempo_text = None
    pending_label = None
    current_measure = 0
    section_marks = []   # list of (measure_num, label)

    for line in kern_lines:
        col = line.split("\t")[0].strip()
        if col.startswith("!!!OMD:"):
            tempo_text = col[7:].strip().rstrip(".").replace('"', '\\"')
        m = re.match(r"^=(\d+)", col)
        if m:
            current_measure = int(m.group(1))
            if pending_label is not None:
                section_marks.append((current_measure, pending_label))
                pending_label = None
        elif col.startswith("*>") and col[2:]:
            pending_label = col[2:]

    # Second pass: inject marks into the .ly.
    # Marks before measure 1 go after the first \time X/X line.
    # Marks before measure N>1 go after the comment "%{ end measure N-1 %}".
    for mnum, label in section_marks:
        mark = f'\\mark \\markup {{ "{label}" }}'
        if mnum == 1:
            # build insertion: tempo (if any) + section mark
            inserts = []
            if tempo_text:
                inserts.append(f'\\tempo "{tempo_text}"')
                tempo_text = None   # only emit once
            inserts.append(mark)
            insert_str = "\n             ".join(inserts)
            ly = re.sub(
                r'(\\time \d+/\d+\n)',
                lambda m: m.group(1) + "             " + insert_str + "\n",
                ly, count=1,
            )
        else:
            prev = mnum - 1
            tag = f"%{{{{ end measure {prev} %}}}}"
            ly = re.sub(
                r'(%\{ end measure ' + str(prev) + r' %\})',
                lambda m, lbl=label: m.group(1) + f'\n             \\mark \\markup {{ "{lbl}" }}',
                ly,
            )

    # If tempo was set but no section mark preceded measure 1, inject it alone
    if tempo_text:
        ly = re.sub(
            r'(\\time \d+/\d+\n)',
            lambda m: m.group(1) + f'             \\tempo "{tempo_text}"\n',
            ly, count=1,
        )

    ly_path.write_text(ly)


def kern_to_ly(kern_text: str, tmpdir: Path) -> Path:
    """Parse **kern text and write a LilyPond .ly file."""
    kern_clean = strip_bekern(kern_text)
    krn_path = tmpdir / "input.krn"
    krn_path.write_text(kern_clean)

    score = converter.parse(str(krn_path), format="humdrum")
    ly_path = tmpdir / "score.ly"
    score.write("lily", fp=str(ly_path))
    _inject_kern_metadata(ly_path, kern_clean)
    return ly_path


def musicxml_to_ly(xml_path: Path, tmpdir: Path) -> Path:
    """Parse MusicXML and write a LilyPond .ly file."""
    score = converter.parse(str(xml_path))
    ly_path = tmpdir / "score.ly"
    score.write("lily", fp=str(ly_path))
    return ly_path


# ---------------------------------------------------------------------------
# Rendering: LilyPond .ly → PNG / PDF / SVG
# ---------------------------------------------------------------------------

def patch_ly(ly_path: Path) -> None:
    """
    Fix music21-generated LilyPond syntax for compatibility with LilyPond 2.19+.
    music21 9.x emits some deprecated constructs that newer LilyPond rejects,
    and includes lilypond-book-preamble.ly which causes a title-only first page.
    """
    text = ly_path.read_text()
    # Renamed in LilyPond 2.19
    text = text.replace(r"\RemoveEmptyStaffContext", r"\RemoveEmptyStaves")
    # Old property syntax: #'foo → .foo (in \override / \set contexts)
    text = re.sub(r"(\\(?:override|set)\s+\S+)\s+#'(\S+)", r"\1.\2", text)
    # music21 includes lilypond-book-preamble.ly which forces a title-only
    # first page (book layout).  Removing it keeps the title inline with music.
    text = text.replace('\\include "lilypond-book-preamble.ly"', "")
    # Compact layout: smaller staff so more measures fit per line/page
    text = text.replace('\\version', '#(set-global-staff-size 16)\n\n\\version', 1)
    ly_path.write_text(text)


def render_ly(ly_path: Path, output_format: str, tmpdir: Path) -> list[Path]:
    """
    Run LilyPond on a .ly file and return paths to output files.
    LilyPond writes output alongside the input file by default.
    """
    patch_ly(ly_path)

    stem = ly_path.stem
    flag = f"--{output_format}"

    cmd = ["lilypond", flag, str(ly_path)]
    result = subprocess.run(
        cmd, cwd=str(ly_path.parent),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Print LilyPond errors so user can diagnose
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"LilyPond exited with code {result.returncode}")

    # LilyPond may produce multiple pages: stem.png, stem-page2.png, etc.
    if output_format == "png":
        outputs = sorted(ly_path.parent.glob(f"{stem}*.png"))
    elif output_format == "pdf":
        outputs = sorted(ly_path.parent.glob(f"{stem}*.pdf"))
    else:  # svg
        outputs = sorted(ly_path.parent.glob(f"{stem}*.svg"))

    return outputs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: Path | None, output_format: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)

        # --- Determine source .ly ---
        if input_path is None:
            # Use embedded sample kern
            ly_path = kern_to_ly(SAMPLE_KERN, tmpdir)
            source_label = "<embedded sample>"
        else:
            fmt = detect_format(input_path)
            source_label = str(input_path)
            if fmt == "kern":
                ly_path = kern_to_ly(input_path.read_text(), tmpdir)
            elif fmt == "musicxml":
                ly_path = musicxml_to_ly(input_path, tmpdir)
            else:  # lilypond
                ly_path = tmpdir / "score.ly"
                shutil.copy(input_path, ly_path)

        print(f"Input : {source_label}")
        print(f"Format: {output_format.upper()}")

        # --- Render ---
        rendered = render_ly(ly_path, output_format, tmpdir)
        if not rendered:
            raise RuntimeError("LilyPond produced no output files.")

        # --- Copy to output dir ---
        saved = []
        for p in rendered:
            dest = output_dir / p.name
            shutil.copy(p, dest)
            saved.append(dest)
            print(f"Saved : {dest}")

        return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render sheet music to PNG/PDF/SVG via LilyPond",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py                               # render embedded sample
  python predict.py --input score.krn             # kern file
  python predict.py --input score.xml --format pdf
  python predict.py --input score.ly  --format svg
""",
    )
    parser.add_argument("--input",  default=None,
        help="Input file (.krn, .xml, .mxl, .ly). Default: embedded sample.")
    parser.add_argument("--output", default=None,
        help="Output directory (default: current working directory)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
        help="Output format (default: png)")
    args = parser.parse_args()

    input_path = None
    if args.input:
        input_path = Path(resolve_input_path(args.input))
        if not input_path.exists():
            sys.exit(f"ERROR: input file not found: {input_path}")

    out_dir = Path(resolve_output_path(args.output, default_filename=".") or ".")
    if out_dir.suffix:          # looks like a file, use its parent
        out_dir = out_dir.parent

    print("=" * 50)
    print("LilyPond Score Renderer")
    print("=" * 50)

    saved = run(input_path, args.format, out_dir)

    print("=" * 50)
    print(f"Done — {len(saved)} file(s) written.")


if __name__ == "__main__":
    main()
