#!/usr/bin/env python3
"""
Audiveris — classical rule-based Optical Music Recognition baseline.

Takes a scanned sheet music image and outputs MusicXML (.mxl) using
Audiveris 5.9.0 in headless batch mode. No GPU required.

Repository: https://github.com/Audiveris/audiveris
License: AGPL-3.0
"""

import os
import sys
import shutil
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

# CVL dual-mode execution support
try:
    from cvlization.paths import (
        get_input_dir,
        get_output_dir,
        resolve_input_path,
        resolve_output_path,
    )
except ImportError:
    def get_input_dir():
        return os.getcwd()

    def get_output_dir():
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path, base_dir):
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    def resolve_output_path(path, base_dir):
        return path if os.path.isabs(path) else os.path.join(base_dir, path)


AUDIVERIS_VERSION = "5.9.0"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "cvl_data" / "audiveris"


def find_audiveris() -> str:
    """Locate the Audiveris binary."""
    # Try PATH first (deb installs a symlink there)
    binary = shutil.which("Audiveris") or shutil.which("audiveris")
    if binary:
        return binary
    # jpackage deb installs to /opt/audiveris/bin/Audiveris
    candidates = [
        "/opt/audiveris/bin/Audiveris",
        "/usr/bin/Audiveris",
        "/usr/local/bin/Audiveris",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        "Audiveris binary not found. Is the .deb installed?"
    )


def ensure_sample_image() -> Path:
    """Download sample score image from zzsi/cvl."""
    cache_path = CACHE_DIR / "sample_page.png"
    if cache_path.exists():
        return cache_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id="zzsi/cvl",
        repo_type="dataset",
        filename="audiveris/sample_page.png",
        local_dir=str(CACHE_DIR.parent),
    )
    return Path(local)


def run_audiveris(audiveris_bin: str, image_path: Path, out_dir: Path, verbose: bool = False) -> Path:
    """
    Run Audiveris in batch mode and return path to the output .mxl file.

    Wraps with xvfb-run because Audiveris initialises Java AWT even in
    -batch mode and fails without a display.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "xvfb-run", "--auto-servernum",
        audiveris_bin,
        "-batch",
        "-transcribe",
        "-export",
        "-output", str(out_dir),
        "--", str(image_path),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
        timeout=300,
    )

    if verbose or result.returncode != 0:
        if result.stdout:
            print(result.stdout[-3000:])
        if result.stderr:
            print(result.stderr[-3000:], file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Audiveris exited with code {result.returncode}")

    # Audiveris outputs <out_dir>/<input_stem>.mxl
    mxl = out_dir / (image_path.stem + ".mxl")
    if not mxl.exists():
        # Fallback: find any .mxl in output dir
        candidates = list(out_dir.rglob("*.mxl"))
        if not candidates:
            raise FileNotFoundError(f"No .mxl output found in {out_dir}")
        mxl = candidates[0]

    return mxl


def main():
    parser = argparse.ArgumentParser(
        description="Audiveris OMR: classical rule-based sheet music transcription to MusicXML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe auto-downloaded sample image
  python predict.py

  # Transcribe your own score
  python predict.py --image score.jpg

  # Save output to specific path
  python predict.py --image score.jpg --output score.mxl
        """,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input score image (default: auto-download sample from HuggingFace)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .mxl file path (default: outputs/result.mxl)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full Audiveris log output",
    )
    args = parser.parse_args()

    INP = get_input_dir()
    OUT = get_output_dir()

    if args.image is None:
        image_path = ensure_sample_image()
    else:
        image_path = Path(resolve_input_path(args.image, INP))

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 1

    output_path = Path(resolve_output_path(args.output or "result.mxl", OUT))

    try:
        audiveris_bin = find_audiveris()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"\n{'='*60}")
    print("Audiveris — OMR (classical rule-based baseline)")
    print("="*60)
    print(f"  Version: {AUDIVERIS_VERSION}")
    print(f"  Image:   {image_path}")
    print(f"  Binary:  {audiveris_bin}")
    print(f"  Output:  {output_path}")
    print("="*60 + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            mxl_path = run_audiveris(audiveris_bin, image_path, tmp_dir, verbose=args.verbose)
        except Exception as e:
            print(f"Error during Audiveris processing: {e}")
            return 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mxl_path, output_path)

    size_kb = output_path.stat().st_size / 1024
    print(f"\nOutput saved to: {output_path}")
    print(f"Output size: {size_kb:.1f} KB")
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
