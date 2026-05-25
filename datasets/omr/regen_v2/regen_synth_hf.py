"""Re-render synthetic-scores HF pages with the new SVG+cairosvg pipeline.

For each HF row, write the embedded musicxml to disk, then call
synthetic_scores.batch_render in chunks. Output goes to ~/.cache/synthetic_v2/<config>/<score_id>.png.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = "/home/zsi/projects/worktrees/CVlization/vintage-music-sheet"
sys.path.insert(0, f"{REPO}/datasets/omr/synthetic_scores")
from generate import batch_render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="level7a")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-root", default=os.path.expanduser(
        "~/.cache/synthetic_v2"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/tmp/zsi_hf_make")
    from datasets import load_dataset
    ds = load_dataset("zzsi/synthetic-scores", args.config,
                      split=args.split, streaming=True)
    out_root = Path(args.out_root) / args.config / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    chunk_dir = out_root / "_chunk"
    chunk_dir.mkdir(exist_ok=True)
    names: list[str] = []
    total_ok = 0
    i = 0

    def _flush():
        nonlocal names, total_ok
        if not names:
            return
        n = batch_render(str(chunk_dir), names)
        total_ok += n
        # Move PNGs to final location
        for name in names:
            src = chunk_dir / f"{name}.png"
            dst = out_root / f"{name}.png"
            if src.exists():
                src.replace(dst)
        # Drop musicxml staging files
        for f in chunk_dir.glob("*.musicxml"):
            f.unlink()
        names = []

    for r in ds:
        if args.limit and i >= args.limit:
            break
        sid = r["score_id"]
        png_final = out_root / f"{sid}.png"
        if png_final.exists():
            i += 1
            continue
        (chunk_dir / f"{sid}.musicxml").write_text(r["musicxml"])
        names.append(sid)
        i += 1
        if len(names) >= args.chunk:
            _flush()
            print(f"  [{i}] total_ok={total_ok}", flush=True)
    _flush()
    print(f"Done: {total_ok} rendered to {out_root}", flush=True)


if __name__ == "__main__":
    main()
