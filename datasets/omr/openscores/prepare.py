#!/usr/bin/env python3
"""
Prepare OpenScore corpora for HuggingFace Hub.

Downloads MusicXML/MuseScore sources from GitHub, renders full-page score
images via the cvlization/lilypond Docker image, and pushes to Hub.

Corpora
-------
  lieder    OpenScore Lieder corpus — 1,460 voice + piano songs (.mxl)
  quartets  OpenScore String Quartets — 122 quartets (.mscx)
  orchestra OpenScore Orchestra / Hauptstimme — 94 orchestral movements (.mxl)

The script runs in two modes:

  Host mode (default):
    1. Download & extract zips from GitHub into CACHE_DIR/raw/
    2. Spin up cvlization/lilypond:latest Docker image once per corpus to
       batch-render all scores (no per-file Docker overhead).
    3. Collect rendered PNGs, build HuggingFace DatasetDict, push to Hub.

  Batch-render mode (--batch-render, runs INSIDE Docker):
    Called automatically by the host; not intended for direct use.

Usage
-----
    python prepare.py --inspect
    python prepare.py --corpus lieder    --push-to-hub zzsi/openscore
    python prepare.py --corpus quartets  --push-to-hub zzsi/openscore
    python prepare.py --corpus orchestra --push-to-hub zzsi/openscore
    python prepare.py --corpus all       --push-to-hub zzsi/openscore
"""

import argparse
import json
import os
import random
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Corpus definitions
# ---------------------------------------------------------------------------

CORPORA = {
    "lieder": {
        "url":         "https://github.com/OpenScore/Lieder/archive/refs/heads/main.zip",
        "zip_subdir":  "Lieder-main",
        "score_glob":  "scores/**/*.mxl",
        "scores_root": "scores",   # relative to corpus_root
        "fmt":         "mxl",
        "description": "OpenScore Lieder corpus — voice + piano",
        "instruments": ["voice", "piano"],
        "exclude":     [],
    },
    "quartets": {
        "url":         "https://github.com/OpenScore/StringQuartets/archive/refs/heads/main.zip",
        "zip_subdir":  "StringQuartets-main",
        # .mscx files are converted to .mxl before rendering (requires MuseScore 3.6.x)
        "score_glob":  "scores/**/*.mxl",
        "mscx_glob":   "scores/**/*.mscx",   # source format before conversion
        "scores_root": "scores",
        "fmt":         "mxl",
        "description": "OpenScore String Quartets — violin I/II + viola + cello",
        "instruments": ["violin", "violin", "viola", "cello"],
        "exclude":     [],
    },
    "orchestra": {
        # MarkGotham/Hauptstimme — same team as OpenScore, CC0
        "url":         "https://github.com/MarkGotham/Hauptstimme/archive/refs/heads/main.zip",
        "zip_subdir":  "Hauptstimme-main",
        "score_glob":  "data/**/*.mxl",
        "scores_root": "data",
        "fmt":         "mxl",
        "description": "OpenScore Orchestra (Hauptstimme) — 94 orchestral movements",
        "instruments": ["orchestra"],
        # Each movement also ships a *_melody.mxl extract — skip those
        "exclude":     ["_melody.mxl"],
    },
}

CACHE_DIR   = Path.home() / ".cache" / "openscores"
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
LILYPOND_DIR = (REPO_ROOT / "examples" / "perception" /
                "optical_music_recognition" / "lilypond")

# MuseScore 3.6.2 AppImage — downloaded on demand for .mscx → .mxl conversion
MSCORE_APPIMAGE_URL = (
    "https://github.com/musescore/MuseScore/releases/download/v3.6.2/"
    "MuseScore-3.6.2.548021370-x86_64.AppImage"
)
MSCORE_DIR = CACHE_DIR / "musescore"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> None:
    """Download url to dest with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} …")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def download_zip(url: str, dest_dir: Path, zip_subdir: str) -> Path:
    """Download a GitHub zip archive and extract it; return extracted root."""
    extracted = dest_dir / zip_subdir
    if extracted.exists() and any(extracted.iterdir()):
        print(f"  Already extracted: {extracted}")
        return extracted

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} …")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    buf = BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="  downloading") as bar:
        for chunk in r.iter_content(chunk_size=65536):
            buf.write(chunk)
            bar.update(len(chunk))

    buf.seek(0)
    print(f"  Extracting …")
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest_dir)

    return extracted


# ---------------------------------------------------------------------------
# MuseScore .mscx → .mxl conversion
# ---------------------------------------------------------------------------

def find_or_install_mscore() -> Path:
    """
    Return path to a working mscore binary (≥3.6).
    Preference order:
      1. Already-extracted AppImage under MSCORE_DIR/squashfs-root/
      2. System binary if version ≥ 3.6
      3. Download + extract AppImage
    """
    extracted = MSCORE_DIR / "squashfs-root" / "usr" / "bin" / "mscore-portable"
    if extracted.exists():
        return extracted

    # Check system binary version
    for candidate in ("mscore3", "mscore", "musescore3"):
        path = subprocess.run(["which", candidate], capture_output=True, text=True).stdout.strip()
        if path:
            ver = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
            # Accept if version string contains 3.6 or 4.x
            if any(f".{v}." in ver or ver.startswith(f"MuseScore {v}") for v in ("3.6", "4.")):
                return Path(path)

    # Download and extract AppImage
    MSCORE_DIR.mkdir(parents=True, exist_ok=True)
    appimage = MSCORE_DIR / "mscore.AppImage"
    if not appimage.exists():
        download(MSCORE_APPIMAGE_URL, appimage)
    appimage.chmod(0o755)
    print("  Extracting MuseScore AppImage …")
    subprocess.run([str(appimage), "--appimage-extract"],
                   cwd=str(MSCORE_DIR), check=True, capture_output=True)
    return extracted


def convert_mscx_to_mxl(corpus_root: Path, mscx_glob: str) -> int:
    """
    Convert all .mscx files to sibling .mxl files in-place.
    Skips files whose .mxl already exists.
    Returns number of files converted.
    """
    mscx_files = sorted(corpus_root.glob(mscx_glob))
    to_convert = [f for f in mscx_files if not f.with_suffix(".mxl").exists()]
    if not to_convert:
        print(f"  All {len(mscx_files)} .mscx already converted to .mxl")
        return 0

    print(f"  Converting {len(to_convert)} .mscx → .mxl via MuseScore …")
    mscore = find_or_install_mscore()

    converted = 0
    for mscx in tqdm(to_convert, desc="  mscx→mxl"):
        mxl = mscx.with_suffix(".mxl")
        result = subprocess.run(
            ["xvfb-run", "-a", str(mscore), "--export-to", str(mxl), str(mscx)],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and mxl.exists():
            converted += 1
        else:
            print(f"    WARN: failed to convert {mscx.name}: {result.stderr[-200:]}")

    print(f"  Converted {converted}/{len(to_convert)}")
    return converted


def parse_path_metadata(score_path: Path, corpus_root: Path,
                        scores_root: str = "scores") -> dict:
    """
    Extract composer / opus / title / score_id from the relative path.

    Lieder layout:    scores/{Composer}/{Opus}/{Title}/{id}.mxl
    Quartets layout:  scores/{Composer}/{Title}/{id}.mscx
    Orchestra layout: data/{Composer}/{Work}/{movement_num}/{id}.mxl
                      → title = "{Work} / {movement_num}"
    """
    rel = score_path.relative_to(corpus_root / scores_root)
    parts = rel.parts   # depth varies by corpus
    score_id = score_path.stem
    composer = parts[0].replace("_", " ") if len(parts) >= 1 else ""
    if len(parts) >= 4:
        # Lieder: Op / Title / id  OR  Orchestra: Work / mvt / id
        opus  = parts[1].replace("_", " ")
        title = parts[2].replace("_", " ")
    elif len(parts) >= 3:
        opus  = ""
        title = parts[1].replace("_", " ")
    else:
        opus  = ""
        title = ""
    return dict(score_id=score_id, composer=composer, opus=opus, title=title)


# ---------------------------------------------------------------------------
# Batch-render mode  (runs INSIDE the cvlization/lilypond Docker container)
# ---------------------------------------------------------------------------

def batch_render(raw_dir: Path, render_dir: Path, score_glob: str,
                 exclude: list[str] | None = None) -> None:
    """
    Render every score file found under raw_dir → render_dir/<score_id>/*.png.
    This function is called when --batch-render is active (inside Docker).
    """
    # predict.run() lives in /workspace/predict.py inside the container
    sys.path.insert(0, "/workspace")
    from predict import run as lilypond_run   # noqa: PLC0415

    exclude = exclude or []
    all_files = sorted(raw_dir.glob(score_glob))
    score_files = [f for f in all_files
                   if not any(pat in f.name for pat in exclude)]
    print(f"  batch-render: {len(score_files)} files under {raw_dir}"
          f" ({len(all_files) - len(score_files)} excluded)", flush=True)
    ok = err = skipped = 0

    for sf in tqdm(score_files, desc="  rendering"):
        rel = sf.relative_to(raw_dir)
        out_dir = render_dir / rel.parent / rel.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already rendered
        if list(out_dir.glob("*.png")):
            skipped += 1
            continue

        try:
            pages = lilypond_run(sf, "png", out_dir)
            if pages:
                ok += 1
            else:
                print(f"    WARN no pages: {rel}", flush=True)
                err += 1
        except Exception as e:
            print(f"    ERR {rel}: {e}", flush=True)
            err += 1

    print(f"  Done: {ok} ok, {err} errors, {skipped} skipped", flush=True)


# ---------------------------------------------------------------------------
# Host mode: call Docker once per corpus for batch rendering
# ---------------------------------------------------------------------------

def render_corpus_via_docker(corpus_root: Path, render_dir: Path,
                             score_glob: str, corpus_name: str,
                             exclude: list[str] | None = None) -> None:
    """Mount corpus_root + render_dir into Docker and run batch_render."""
    render_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()

    cmd = [
        "docker", "run", "--rm",
        "--mount", f"type=bind,src={corpus_root},dst=/raw,readonly",
        "--mount", f"type=bind,src={render_dir},dst=/rendered",
        "--mount", f"type=bind,src={LILYPOND_DIR},dst=/workspace,readonly",
        "--mount", f"type=bind,src={script_path.parent},dst=/prepare,readonly",
        "--mount", f"type=bind,src={REPO_ROOT},dst=/cvlization_repo,readonly",
        "--env", "PYTHONPATH=/cvlization_repo",
        "cvlization/lilypond:latest",
        "python3", "/prepare/prepare.py",
        "--batch-render",
        "--raw-dir",    "/raw",
        "--render-dir", "/rendered",
        "--score-glob", score_glob,
    ]
    for pat in (exclude or []):
        cmd += ["--exclude", pat]

    print(f"  Running Docker batch-render for {corpus_name} …")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"  WARNING: Docker exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Build HuggingFace dataset
# ---------------------------------------------------------------------------

def collect_rows(corpus_name: str, corpus_root: Path, render_dir: Path,
                 score_glob: str, meta: dict) -> list[dict]:
    """Walk rendered PNGs and build one row per page."""
    exclude      = meta.get("exclude", [])
    scores_root  = meta.get("scores_root", "scores")
    all_files    = sorted(corpus_root.glob(score_glob))
    score_files  = [f for f in all_files
                    if not any(pat in f.name for pat in exclude)]
    rows = []
    missing_renders = 0

    for sf in tqdm(score_files, desc=f"  collecting {corpus_name}"):
        rel = sf.relative_to(corpus_root)
        page_dir = render_dir / rel.parent / rel.stem
        pages = sorted(page_dir.glob("*.png")) if page_dir.exists() else []
        if not pages:
            missing_renders += 1
            continue

        path_meta = parse_path_metadata(sf, corpus_root, scores_root)

        for page_idx, png in enumerate(pages):
            rows.append({
                "image":       str(png),
                "score_id":    path_meta["score_id"],
                "composer":    path_meta["composer"],
                "opus":        path_meta["opus"],
                "title":       path_meta["title"],
                "corpus":      corpus_name,
                "instruments": meta["instruments"],
                "page":        page_idx + 1,
                "n_pages":     len(pages),
            })

    if missing_renders:
        print(f"  WARNING: {missing_renders} scores had no rendered pages")
    print(f"  {corpus_name}: {len(rows)} page rows from "
          f"{len(score_files) - missing_renders} scores")
    return rows


def build_dataset(all_rows: list[dict]):
    """Split rows by score_id (score-level), cast images, return DatasetDict."""
    from datasets import Dataset, DatasetDict
    from datasets import Image as HFImage

    # Group pages by score_id
    score_ids = list({r["score_id"] for r in all_rows})
    random.shuffle(score_ids)
    n = len(score_ids)
    n_test = max(1, int(n * 0.05))
    n_dev  = max(1, int(n * 0.05))

    test_ids = set(score_ids[:n_test])
    dev_ids  = set(score_ids[n_test:n_test + n_dev])
    train_ids = set(score_ids[n_test + n_dev:])

    split_rows: dict[str, list] = {"train": [], "dev": [], "test": []}
    for row in all_rows:
        sid = row["score_id"]
        if sid in test_ids:
            split_rows["test"].append(row)
        elif sid in dev_ids:
            split_rows["dev"].append(row)
        else:
            split_rows["train"].append(row)

    split_datasets = {}
    for split, rows in split_rows.items():
        if not rows:
            continue
        ds = Dataset.from_list(rows)
        ds = ds.cast_column("image", HFImage())
        split_datasets[split] = ds
        print(f"  {split}: {len(ds)} pages")

    return DatasetDict(split_datasets)


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

DATASET_CARD = """\
---
license: cc-by-sa-4.0
task_categories:
  - image-to-text
language:
  - en
tags:
  - music
  - optical-music-recognition
  - omr
  - sheet-music
  - musicxml
  - lilypond
size_categories:
  - 1K<n<10K
---

# OpenScore — Full-Page Score Images

Full-page score images rendered from the [OpenScore](https://github.com/OpenScore)
corpora via [LilyPond](https://lilypond.org), paired with source MusicXML.

## Corpora

| Corpus | Scores | Staves/system | Source |
|--------|--------|---------------|--------|
| Lieder | ~1,460 | 3 (voice + piano) | [OpenScore/Lieder](https://github.com/OpenScore/Lieder) |
| Quartets | ~122 | 4 (violin I/II + viola + cello) | [OpenScore/StringQuartets](https://github.com/OpenScore/StringQuartets) |
| Orchestra | ~94 movements | 10–20+ (full orchestra) | [MarkGotham/Hauptstimme](https://github.com/MarkGotham/Hauptstimme) |

## Format

```python
{
    "image":       PIL.Image,   # full-page score render (LilyPond Emmentaler font)
    "score_id":    str,         # e.g. "lc6583477"
    "composer":    str,
    "opus":        str,
    "title":       str,
    "corpus":      str,         # "lieder" or "quartets"
    "instruments": list[str],
    "page":        int,         # 1-indexed
    "n_pages":     int,
}
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("zzsi/openscore")
example = ds["train"][0]
example["image"].show()
```

## License

Source scores: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
LilyPond renders: same license as source scores.

## Attribution

- OpenScore Lieder corpus: https://github.com/OpenScore/Lieder
- OpenScore String Quartets: https://github.com/OpenScore/StringQuartets
- OpenScore Orchestra (Hauptstimme): https://github.com/MarkGotham/Hauptstimme
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare OpenScore dataset for HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", default="all",
        choices=["lieder", "quartets", "orchestra", "all"],
        help="Which corpus to process (default: all)")
    parser.add_argument("--inspect", action="store_true",
        help="Download only, print structure, then exit")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
        help="Push to HuggingFace Hub repo (e.g. zzsi/openscore)")
    parser.add_argument("--output", default=None,
        help="Save dataset locally to this directory")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for train/dev/test split")

    # Internal args used when running inside Docker
    parser.add_argument("--batch-render", action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument("--raw-dir",    default=None, help=argparse.SUPPRESS)
    parser.add_argument("--render-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--score-glob", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--exclude", action="append", default=[],
        help=argparse.SUPPRESS)

    args = parser.parse_args()
    random.seed(args.seed)

    # -----------------------------------------------------------------
    # Batch-render mode: runs inside Docker
    # -----------------------------------------------------------------
    if args.batch_render:
        batch_render(
            raw_dir    = Path(args.raw_dir),
            render_dir = Path(args.render_dir),
            score_glob = args.score_glob,
            exclude    = args.exclude,
        )
        return

    # -----------------------------------------------------------------
    # Host mode
    # -----------------------------------------------------------------
    corpora_to_run = (
        list(CORPORA.keys()) if args.corpus == "all"
        else [args.corpus]
    )

    all_rows = []

    for corpus_name in corpora_to_run:
        meta = CORPORA[corpus_name]
        print(f"\n=== {corpus_name}: {meta['description']} ===")

        # 1. Download & extract
        raw_dir = CACHE_DIR / "raw" / corpus_name
        corpus_root = download_zip(meta["url"], raw_dir, meta["zip_subdir"])

        if args.inspect:
            exclude     = meta.get("exclude", [])
            scores_root = meta.get("scores_root", "scores")
            all_files   = sorted(corpus_root.glob(meta["score_glob"]))
            score_files = [f for f in all_files
                           if not any(pat in f.name for pat in exclude)]
            print(f"  Score files: {len(score_files)}"
                  f" ({len(all_files) - len(score_files)} excluded)")
            for sf in score_files[:5]:
                pm = parse_path_metadata(sf, corpus_root, scores_root)
                print(f"    {pm['composer']} / {pm['opus']} / {pm['title']} ({pm['score_id']})")
            continue

        # 2. Convert .mscx → .mxl if needed (quartets corpus)
        if meta.get("mscx_glob"):
            convert_mscx_to_mxl(corpus_root, meta["mscx_glob"])

        # 3. Render via LilyPond Docker
        render_dir = CACHE_DIR / "rendered" / corpus_name
        render_corpus_via_docker(
            corpus_root = corpus_root,
            render_dir  = render_dir,
            score_glob  = meta["score_glob"],
            corpus_name = corpus_name,
            exclude     = meta.get("exclude", []),
        )

        # 3. Collect rows
        rows = collect_rows(corpus_name, corpus_root, render_dir,
                            meta["score_glob"], meta)
        all_rows.extend(rows)

    if args.inspect:
        return

    if not all_rows:
        print("\nNo rows collected — check render logs above.")
        return

    print(f"\nTotal: {len(all_rows)} page rows across all corpora")
    print("Building HuggingFace DatasetDict …")
    dd = build_dataset(all_rows)
    print(f"\nDataset:\n{dd}")

    if args.output:
        out = Path(args.output)
        print(f"\nSaving to {out} …")
        dd.save_to_disk(str(out))

    if args.push_to_hub:
        repo_id = args.push_to_hub
        print(f"\nPushing to {repo_id} …")
        dd.push_to_hub(repo_id, private=False)

        from huggingface_hub import HfApi
        api = HfApi()
        card_path = CACHE_DIR / "README.md"
        card_path.write_text(DATASET_CARD)
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Done — https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
