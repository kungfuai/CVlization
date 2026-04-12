#!/usr/bin/env python3
"""
Push synthetic scores to HuggingFace as zzsi/synthetic-scores.

Each difficulty level is a separate config (level1, level2, ...).
All configs share the same schema: image, musicxml, score_id, level.
Auto-creates train/dev/test splits (80/10/10).

Usage:
    python push_hf.py --input /tmp/synthetic_scores/level1 --level 1
    python push_hf.py --input /tmp/synthetic_scores/level1 --level 1 --repo zzsi/synthetic-scores
"""

import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, Image as HFImage


def build_dataset(input_dir: str, level, train_frac: float = 0.8, dev_frac: float = 0.1):
    """Build train/dev/test splits from a directory of .musicxml + .png pairs."""
    input_dir = Path(input_dir)
    rows = []
    for mxml_path in sorted(input_dir.glob("*.musicxml")):
        png_path = mxml_path.with_suffix(".png")
        if not png_path.exists():
            continue
        rows.append({
            "image": str(png_path),
            "musicxml": mxml_path.read_text(),
            "score_id": mxml_path.stem,
            "level": level,
        })

    n = len(rows)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)

    splits = {}
    splits["train"] = Dataset.from_list(rows[:n_train]).cast_column("image", HFImage())
    splits["dev"] = Dataset.from_list(rows[n_train:n_train + n_dev]).cast_column("image", HFImage())
    splits["test"] = Dataset.from_list(rows[n_train + n_dev:]).cast_column("image", HFImage())

    return DatasetDict(splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Directory with .musicxml + .png files")
    def _level_type(v):
        try:
            return int(v)
        except ValueError:
            return v
    parser.add_argument("--level", type=_level_type, required=True, help="Difficulty level (1-7, 6b, 7a, 7b, 7c)")
    parser.add_argument("--repo", default="zzsi/synthetic-scores", help="HuggingFace repo ID")
    args = parser.parse_args()

    print(f"Building dataset from {args.input} (level {args.level})...")
    dd = build_dataset(args.input, args.level)
    for split, ds in dd.items():
        print(f"  {split}: {len(ds)} samples")

    config_name = f"level{args.level}"
    print(f"Pushing to {args.repo} config={config_name}...")
    dd.push_to_hub(args.repo, config_name=config_name)
    print(f"Done — https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
