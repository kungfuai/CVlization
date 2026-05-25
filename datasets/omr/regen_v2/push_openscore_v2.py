"""Push regenerated openscore PNGs back to zzsi/openscore.

Mirrors push_synth_v2.py: stream the existing config, replace `image`
with the v2 PNG (matched by score_id + page), keep all other fields,
push back. Touches `pages_transcribed` and `pages` configs.
"""
import argparse
import os
from pathlib import Path

V2_ROOT = Path(os.path.expanduser("~/.cache/openscores_v2"))


def _build_lookup() -> dict:
    """Map (score_id, page) -> Path. Mirrors push_hf_configs.build_png_lookup
    but reads from openscores_v2 instead of openscores/rendered."""
    out: dict[tuple[str, int], Path] = {}
    for png in V2_ROOT.rglob("score-page*.png"):
        try:
            sid = png.parent.name
            page = int(png.stem.replace("score-page", ""))
            out[(sid, page)] = png
        except Exception:
            continue
    return out


def push_config(repo: str, config: str, split: str,
                lookup: dict, dry_run: bool) -> None:
    from datasets import Dataset, load_dataset
    from datasets import Image as HFImage

    print(f"\n=== {repo}/{config}/{split} ===", flush=True)
    src = load_dataset(repo, config, split=split)
    print(f"  source: {len(src)} rows, columns: {src.column_names}", flush=True)
    rows = []
    missing = 0
    for r in src:
        key = (r["score_id"], r["page"])
        png = lookup.get(key)
        if png is None:
            missing += 1
            continue
        new = dict(r)
        new["image"] = str(png)
        rows.append(new)
    print(f"  kept {len(rows)} rows ({missing} missing v2 png)", flush=True)
    if not rows:
        return
    ds = Dataset.from_list(rows).cast_column("image", HFImage())
    if dry_run:
        print(f"  [DRY] would push {len(ds)} rows", flush=True)
        return
    print(f"  Pushing {len(ds)} rows to {config}/{split} ...", flush=True)
    ds.push_to_hub(repo, config_name=config, split=split)
    print(f"  Pushed.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="zzsi/openscore")
    ap.add_argument("--config", required=True,
                    help="e.g. pages_transcribed, pages")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("Building PNG lookup from openscores_v2 ...", flush=True)
    lookup = _build_lookup()
    print(f"  {len(lookup)} (score_id, page) entries", flush=True)
    push_config(args.repo, args.config, args.split, lookup, args.dry_run)


if __name__ == "__main__":
    main()
