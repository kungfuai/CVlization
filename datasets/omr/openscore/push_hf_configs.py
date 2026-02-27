#!/usr/bin/env python3
"""
Push HuggingFace configs for zzsi/openscore.

Builds and pushes the following configs:
  pages_transcribed  — (image, per-page-musicxml) pairs for SFT, lieder corpus
  pages              — image-only rows, all corpora (rename of current 'default')

Usage:
    python push_hf_configs.py --pages-transcribed --repo zzsi/openscore
    python push_hf_configs.py --pages            --repo zzsi/openscore
    python push_hf_configs.py --all              --repo zzsi/openscore

The 'default' config is left unchanged.  After verifying the new configs,
delete 'default' manually from the HF Hub UI if desired.
"""

import argparse
import sys
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "openscores"

RENDER_ROOTS = {
    "lieder":    (CACHE_DIR / "rendered" / "lieder"    / "scores", "scores"),
    "quartets":  (CACHE_DIR / "rendered" / "quartets"  / "scores", "scores"),
    "orchestra": (CACHE_DIR / "rendered" / "orchestra" / "data",   "data"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def build_png_lookup(corpus_name: str) -> dict:
    """Return {(score_id, page): {"path": Path, "composer": str, "opus": str, "title": str}}."""
    render_base, _ = RENDER_ROOTS[corpus_name]
    if not render_base.exists():
        print(f"  WARN: render dir not found: {render_base}")
        return {}

    lookup = {}
    for png in render_base.rglob("score-page*.png"):
        try:
            rel   = png.parent.relative_to(render_base)
            parts = rel.parts          # e.g. ('Composer', 'Opus', 'Title', 'lc1234567')
            score_id = parts[-1]
            page_num = int(png.stem.replace("score-page", ""))

            composer = parts[0].replace("_", " ") if len(parts) >= 1 else ""
            if len(parts) >= 4:
                opus_raw = parts[1]
                opus  = "" if opus_raw == "_" else opus_raw.replace("_", " ")
                title = parts[2].replace("_", " ")
            elif len(parts) >= 3:
                opus  = ""
                title = parts[1].replace("_", " ")
            else:
                opus  = ""
                title = ""

            lookup[(score_id, page_num)] = {
                "path": png,
                "composer": composer,
                "opus": opus,
                "title": title,
            }
        except Exception as e:
            print(f"  WARN: could not parse {png}: {e}")
    return lookup


# ── pages_transcribed ─────────────────────────────────────────────────────────

def push_pages_transcribed(musicxml_dir: Path, repo_id: str, corpus: str = "lieder") -> None:
    from datasets import load_from_disk, DatasetDict
    from datasets import Image as HFImage

    print(f"\nBuilding pages_transcribed for {corpus} ...")
    print(f"  Loading musicxml dataset from {musicxml_dir} ...")
    dd = load_from_disk(str(musicxml_dir))
    print(f"  {dd}")

    png_lookup = build_png_lookup(corpus)
    print(f"  PNG lookup: {len(png_lookup)} entries")

    def add_image_and_meta(row):
        key = (row["score_id"], row["page"])
        entry = png_lookup.get(key)
        if entry:
            row["image"]    = str(entry["path"])
            row["composer"] = entry["composer"]
            row["opus"]     = entry["opus"]
            row["title"]    = entry["title"]
        else:
            row["image"]    = None
            row["composer"] = ""
            row["opus"]     = ""
            row["title"]    = ""
        return row

    new_splits = {}
    for split_name, ds in dd.items():
        print(f"  Mapping {split_name} ({len(ds)} rows) ...")
        ds = ds.map(add_image_and_meta, desc=f"  {split_name}")
        missing = sum(1 for r in ds if r["image"] is None)
        if missing:
            print(f"    WARN: {missing} rows with no matching PNG — dropping")
            ds = ds.filter(lambda r: r["image"] is not None)
        ds = ds.cast_column("image", HFImage())
        new_splits[split_name] = ds

    result = DatasetDict(new_splits)
    print(f"\n  Final dataset:\n{result}")

    print(f"\n  Pushing to {repo_id} config=pages_transcribed ...")
    result.push_to_hub(repo_id, config_name="pages_transcribed", private=False)
    print("  Done.")


# ── pages (image-only, all corpora) ──────────────────────────────────────────

def push_pages(repo_id: str, corpora: list[str] | None = None) -> None:
    """Rebuild the image-only 'pages' config from local PNG cache (all corpora)."""
    from datasets import Dataset, DatasetDict, load_dataset
    from datasets import Image as HFImage

    if corpora is None:
        corpora = list(RENDER_ROOTS.keys())

    print(f"\nBuilding pages config for corpora: {corpora} ...")

    # Load split assignments from existing default config (no images)
    print("  Loading split assignments from default config ...")
    source = load_dataset("zzsi/openscore", "default",
                          columns=["score_id", "corpus", "page", "n_pages"])

    # Build PNG lookups
    png_lookups = {c: build_png_lookup(c) for c in corpora}
    total_pngs  = sum(len(v) for v in png_lookups.values())
    print(f"  PNG entries: {total_pngs}")

    split_dicts: dict[str, list] = {}
    for split_name, split_ds in source.items():
        rows = []
        missing = 0
        for row in split_ds:
            corpus = row["corpus"]
            if corpus not in corpora:
                continue
            key   = (row["score_id"], row["page"])
            entry = png_lookups[corpus].get(key)
            if entry is None:
                missing += 1
                continue
            rows.append({
                "image":    str(entry["path"]),
                "score_id": row["score_id"],
                "corpus":   corpus,
                "page":     row["page"],
                "n_pages":  row["n_pages"],
                "composer": entry["composer"],
                "opus":     entry["opus"],
                "title":    entry["title"],
            })
        if missing:
            print(f"  WARN {split_name}: {missing} rows with no PNG")
        ds = Dataset.from_list(rows)
        ds = ds.cast_column("image", HFImage())
        split_dicts[split_name] = ds
        print(f"  {split_name}: {len(ds)} rows")

    result = DatasetDict(split_dicts)
    print(f"\n  Final dataset:\n{result}")
    print(f"\n  Pushing to {repo_id} config=pages ...")
    result.push_to_hub(repo_id, config_name="pages", private=False)
    print("  Done.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo",               default="zzsi/openscore")
    parser.add_argument("--pages-transcribed",  action="store_true",
                        help="Build and push pages_transcribed config (lieder)")
    parser.add_argument("--musicxml-dir",       default="/tmp/openscore_pages_lieder/lieder",
                        help="Directory with the page_musicxml.py output DatasetDict")
    parser.add_argument("--pages",              action="store_true",
                        help="Rebuild and push image-only pages config (all corpora)")
    parser.add_argument("--all",                action="store_true",
                        help="Run all push steps")
    args = parser.parse_args()

    if not (args.pages_transcribed or args.pages or args.all):
        parser.print_help()
        sys.exit(1)

    if args.pages_transcribed or args.all:
        push_pages_transcribed(Path(args.musicxml_dir), args.repo)

    if args.pages or args.all:
        push_pages(args.repo)


if __name__ == "__main__":
    main()
