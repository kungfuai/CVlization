#!/usr/bin/env python3
"""
Render ekern/bekern transcriptions from comparison JSONs to PNG images.

Usage:
    python render_ekern.py
    python render_ekern.py --models smt gemini3 claude_opus
    python render_ekern.py --models gemini3 --renderer lilypond
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

COMPARISON_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Parsers for each JSON format
# ---------------------------------------------------------------------------

def extract_code_block(text: str) -> str:
    """Strip ```...``` fences from a response string."""
    m = re.search(r"```[a-z]*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def sanitize_kern(kern: str) -> str:
    """
    Remove spine-split/join sections that verovio can't handle,
    enforce a uniform column count, and fix single-column spine records
    (section labels *>X, terminator *-) by duplicating them for both spines.
    """
    lines = kern.split("\n")
    result = []
    base_cols = 2  # standard two-staff piano score

    for line in lines:
        # Skip spine-operation lines (*^, *v)
        if re.match(r"^\*[\^v]", line) or re.match(r".*\t\*[\^v]", line):
            continue
        # !!LO: and !!! lines are global comments — pass through unchanged
        if line.startswith("!!") or line.startswith("!!!"):
            # Ensure !!LO:TX lines have a placement directive (,a = above staff)
            # Normalise to !!LO:TX,a,t=<text> — verovio needs placement before text
            if line.startswith("!!LO:TX"):
                # Extract the t= value and rebuild in canonical order
                import re as _re
                t_match = _re.search(r",t=([^,]+)", line)
                if t_match:
                    text_val = t_match.group(1).rstrip()
                    line = f"!!LO:TX,a,t={text_val}"
            result.append(line)
            continue
        cols = line.split("\t")
        # Single-column spine records (*>label, *-) need duplicating for each spine
        if len(cols) == 1 and line.startswith("*"):
            cols = [line] * base_cols
        # Trim to base_cols if too wide
        elif len(cols) > base_cols:
            cols = cols[:base_cols]
        result.append("\t".join(cols))

    return "\n".join(result)


def ekern_to_kern(ekern: str) -> str:
    """Convert ekern/bekern text to renderable **kern format."""
    # Strip code fences
    ekern = extract_code_block(ekern)
    # Replace **ekern_1.0 header with **kern
    ekern = ekern.replace("**ekern_1.0", "**kern")
    # Strip bekern encoding artifacts
    ekern = ekern.replace("@", "").replace("·", "")
    # If no **kern header anywhere, prepend one (SMT raw bekern)
    if "**kern" not in ekern:
        ekern = "**kern\t**kern\n" + ekern
    # Remove spine-split sections that crash verovio
    ekern = sanitize_kern(ekern)
    return ekern


def load_smt() -> tuple[str, str]:
    p = COMPARISON_DIR / "smt_omr.json"
    data = json.loads(p.read_text())
    return "smt_omr", data["bekern"]


def load_gemini3() -> tuple[str, str]:
    # Prefer the rich_kern run (with embedded metadata) if available
    for fname in ("gemini3_rich_kern.json", "gemini3_ekern4k.json"):
        p = COMPARISON_DIR / fname
        if p.exists():
            data = json.loads(p.read_text())
            return "gemini3", data["results"]["ekern_transcription"]["response"]
    raise FileNotFoundError("No gemini3 JSON found")


def load_claude_opus() -> tuple[str, str]:
    p = COMPARISON_DIR / "claude_opus_ekern4k.json"
    data = json.loads(p.read_text())
    return "claude_opus", data["results"]["ekern_transcription"]["response"]


LOADERS = {
    "smt": load_smt,
    "gemini3": load_gemini3,
    "claude_opus": load_claude_opus,
}

# ---------------------------------------------------------------------------
# Rendering worker (runs in subprocess to isolate verovio crashes)
# ---------------------------------------------------------------------------

def _render_worker(kern_text: str, out_path_str: str, page_width: int):
    """Worker entry point — called in a fresh subprocess."""
    import verovio
    from cairosvg import svg2png

    out_path = Path(out_path_str)
    verovio.enableLog(verovio.LOG_OFF)
    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": page_width,
        "footer": "none",
        "header": "encoded",   # renders !!!OTL: / !!!COM: humdrum records
    })
    tk.loadData(kern_text)
    n_pages = tk.getPageCount()
    print(f"  → {n_pages} page(s)", flush=True)

    for pg in range(1, n_pages + 1):
        svg = tk.renderToSVG(pg)
        svg = svg.replace('overflow="inherit"', 'overflow="visible"')
        png_bytes = svg2png(bytestring=svg.encode(), background_color="white")
        if n_pages == 1:
            page_path = out_path
        else:
            page_path = out_path.with_stem(out_path.stem + f"_p{pg:02d}")
        page_path.write_bytes(png_bytes)
        print(f"  Saved: {page_path}", flush=True)


def render_kern_to_png(kern_text: str, out_path: Path, page_width: int = 2100):
    """Render a **kern string to PNG, isolated in a subprocess."""
    # Write kern to temp file so subprocess can read it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".kern", delete=False) as tf:
        tf.write(kern_text)
        tf_path = tf.name

    # Launch subprocess running this same file with --worker flag
    cmd = [
        sys.executable, __file__,
        "--worker", tf_path, str(out_path), str(page_width)
    ]
    result = subprocess.run(cmd, capture_output=False)
    Path(tf_path).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"Rendering subprocess exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# LilyPond rendering (via cvlization/lilypond Docker image)
# ---------------------------------------------------------------------------

LILYPOND_IMAGE = "cvlization/lilypond:latest"


def render_kern_to_png_lilypond(kern_text: str, out_path: Path):
    """
    Render **kern to PNG using the cvlization/lilypond Docker container.
    The container has music21 + lilypond pre-installed; kern_to_ly() converts
    kern → .ly, then LilyPond engraves with its Emmentaler font.
    Multi-page outputs are saved as out_stem_p01.png, out_stem_p02.png, etc.
    """
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        krn_file = tmp / "score.krn"
        krn_file.write_text(kern_text)

        cmd = [
            "docker", "run", "--rm",
            "--mount", f"type=bind,src={tmp},dst=/mnt/work",
            LILYPOND_IMAGE,
            "python3", "/workspace/predict.py",
            "--input", "/mnt/work/score.krn",
            "--output", "/mnt/work",
            "--format", "png",
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError("LilyPond Docker container failed")

        pngs = sorted(tmp.glob("score*.png"))
        if not pngs:
            raise RuntimeError("No PNG produced by LilyPond")

        if len(pngs) == 1:
            shutil.copy(pngs[0], out_path)
        else:
            for i, pg in enumerate(pngs, 1):
                dest = out_path.with_stem(out_path.stem + f"_p{i:02d}")
                shutil.copy(pg, dest)
                print(f"  Saved: {dest}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render OMR comparison results to PNG")
    parser.add_argument(
        "--models", nargs="+",
        choices=list(LOADERS.keys()), default=list(LOADERS.keys()),
        help="Which models to render (default: all)"
    )
    parser.add_argument(
        "--renderer", choices=["verovio", "lilypond"], default="verovio",
        help="Rendering backend (default: verovio)"
    )
    parser.add_argument(
        "--page-width", type=int, default=2100,
        help="Verovio page width in px (default: 2100; ignored for lilypond)"
    )
    # Internal: worker mode (called by subprocess for verovio isolation)
    parser.add_argument("--worker", nargs=3, metavar=("KERN_FILE", "OUT_PATH", "PAGE_WIDTH"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        kern_file, out_path_str, page_width_str = args.worker
        kern_text = Path(kern_file).read_text()
        _render_worker(kern_text, out_path_str, int(page_width_str))
        return

    for key in args.models:
        loader = LOADERS[key]
        try:
            name, raw = loader()
        except FileNotFoundError as e:
            print(f"[SKIP] {key}: {e}")
            continue

        print(f"\n[{key}] Rendering with {args.renderer}...")
        kern = ekern_to_kern(raw)

        # Dump kern for inspection
        debug_path = COMPARISON_DIR / f"{name}_debug.kern"
        debug_path.write_text(kern)

        suffix = "_ly" if args.renderer == "lilypond" else ""
        out_path = COMPARISON_DIR / f"{name}_rendered{suffix}.png"
        try:
            if args.renderer == "lilypond":
                render_kern_to_png_lilypond(kern, out_path)
            else:
                render_kern_to_png(kern, out_path, page_width=args.page_width)
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Kern dumped to: {debug_path}")


if __name__ == "__main__":
    main()
