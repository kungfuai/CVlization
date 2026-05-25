"""Re-rasterize every cached openscore SVG to PNG at width=1280 via cairosvg.

The HF openscore dataset was rendered with `lilypond --png` at default 101 DPI
(835 wide), which silently diverges from `lilypond --svg` layout on complex
scores. This script overwrites the cached PNGs with renders made from the
same SVG that bbox extraction uses -> guaranteed image-consistency.

Run as: python3 /tmp/regen_openscore_cache.py [--workers N] [--dry-run]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cairosvg

PNG_WIDTH = 1280


SVG_ROOT = Path(os.path.expanduser("~/.cache/openscores/svg"))
OUT_ROOT = Path(os.path.expanduser("~/.cache/openscores_v2"))


def _rasterize(svg_path: Path) -> tuple[Path, bool, str]:
    stem = svg_path.stem  # e.g. 'score-1'
    try:
        idx = int(stem.split("-")[-1])
    except ValueError:
        return svg_path, False, "bad-name"
    rel = svg_path.parent.relative_to(SVG_ROOT)
    out_dir = OUT_ROOT / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"score-page{idx}.png"
    if png_path.exists():
        return svg_path, True, "skip"
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                         output_width=PNG_WIDTH, background_color="white")
        return svg_path, True, "ok"
    except Exception as e:
        return svg_path, False, repr(e)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg-root", default=os.path.expanduser(
        "~/.cache/openscores/svg"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.svg_root)
    svgs = sorted(root.rglob("score-*.svg"))
    print(f"Found {len(svgs)} SVG pages under {root}", flush=True)
    if args.dry_run:
        for s in svgs[:5]:
            print(f"  would render: {s}")
        return

    ok = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_rasterize, s) for s in svgs]
        for i, fut in enumerate(as_completed(futs), 1):
            _, success, msg = fut.result()
            if success:
                ok += 1
            else:
                err += 1
                if err <= 10:
                    print(f"  FAIL {msg}", flush=True)
            if i % 500 == 0:
                print(f"  [{i}/{len(svgs)}] ok={ok} err={err}", flush=True)
    print(f"Done: {ok} ok, {err} err", flush=True)


if __name__ == "__main__":
    main()
