#!/usr/bin/env python3
"""
Push synthetic scores to HuggingFace as zzsi/synthetic-scores.

Each difficulty level is a separate config (level1, level2, ...).
All configs share the same schema: image, musicxml, score_id, level.

Usage:
    python push_hf.py --input /tmp/synthetic_scores/level1 --level 1
    python push_hf.py --input /tmp/synthetic_scores/level1 --level 1 --repo zzsi/synthetic-scores
"""

import argparse
from pathlib import Path
from datasets import Dataset, Image as HFImage


def build_dataset(input_dir: str, level: int) -> Dataset:
    input_dir = Path(input_dir)
    rows = []
    for mxml_path in sorted(input_dir.glob("*.musicxml")):
        png_path = mxml_path.with_suffix(".png")
        if not png_path.exists():
            continue
        score_id = mxml_path.stem  # e.g. "L1_00042"
        rows.append({
            "image": str(png_path),
            "musicxml": mxml_path.read_text(),
            "score_id": score_id,
            "level": level,
        })

    ds = Dataset.from_list(rows)
    ds = ds.cast_column("image", HFImage())
    return ds


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Directory with .musicxml + .png files")
    parser.add_argument("--level", type=int, required=True, help="Difficulty level (1-8)")
    parser.add_argument("--repo", default="zzsi/synthetic-scores", help="HuggingFace repo ID")
    parser.add_argument("--split", default="train", help="Split name (default: train)")
    args = parser.parse_args()

    print(f"Building dataset from {args.input} (level {args.level})...")
    ds = build_dataset(args.input, args.level)
    print(f"  {len(ds)} samples")

    config_name = f"level{args.level}"
    print(f"Pushing to {args.repo} config={config_name} split={args.split}...")
    ds.push_to_hub(args.repo, config_name=config_name, split=args.split)
    print(f"Done — https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
