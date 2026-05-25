"""Push regenerated PNGs (cairosvg-from-SVG, 1280-wide) back to
zzsi/synthetic-scores, ONE level at a time.

Strategy: stream the existing HF dataset for each level's train split,
replace the `image` column with the v2 PNG (matching by score_id), keep
everything else, then push back to the SAME config name.

Note: this currently rewrites only the TRAIN split. dev/test splits stay
as-is in their original 150-DPI rendering. (Run separately for those when
needed.)
"""
import argparse
import os
import sys
from pathlib import Path

V2_ROOT = Path(os.path.expanduser("~/.cache/synthetic_v2"))


def push_level(level: str, split: str = "train",
               repo: str = "zzsi/synthetic-scores",
               dry_run: bool = False) -> None:
    from datasets import Dataset, load_dataset
    from datasets import Image as HFImage

    cfg = f"level{level.lstrip('level')}" if not level.startswith("level") else level
    png_dir = V2_ROOT / cfg / split
    if not png_dir.is_dir():
        print(f"SKIP {cfg}/{split}: {png_dir} missing")
        return
    have = {p.stem: p for p in png_dir.glob("*.png")}
    print(f"{cfg}/{split}: {len(have)} v2 PNGs on disk")

    print(f"  Loading existing rows from {repo}/{cfg}/{split} ...")
    src = load_dataset(repo, cfg, split=split)
    print(f"  source: {len(src)} rows, columns: {src.column_names}")

    rows = []
    missing = 0
    for r in src:
        sid = r["score_id"]
        png = have.get(sid)
        if png is None:
            missing += 1
            continue
        # Build row with all original fields except image -> v2 path
        new_row = dict(r)
        new_row["image"] = str(png)
        rows.append(new_row)
    print(f"  {len(rows)} rows kept, {missing} skipped (no v2 PNG)")
    if not rows:
        return

    ds = Dataset.from_list(rows).cast_column("image", HFImage())
    print(f"  Built new {split} split: {len(ds)} rows")

    if dry_run:
        print(f"  [DRY] would push to {repo} config={cfg} split={split}")
        return

    print(f"  Pushing to {repo} config={cfg} split={split} ...")
    ds.push_to_hub(repo, config_name=cfg, split=split)
    print(f"  Pushed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True,
                    help="e.g. level1, level7a, level9, or 'all'")
    ap.add_argument("--split", default="train")
    ap.add_argument("--repo", default="zzsi/synthetic-scores")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_levels = [
        "level1", "level2", "level3", "level4", "level5",
        "level6", "level6b", "level7", "level7a", "level7b",
        "level7c", "level8", "level9",
    ]
    levels = all_levels if args.level == "all" else [args.level]
    for lvl in levels:
        try:
            push_level(lvl, args.split, args.repo, args.dry_run)
        except Exception as e:
            print(f"FAIL {lvl}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
