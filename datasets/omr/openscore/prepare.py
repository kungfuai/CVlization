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
        # Standard orchestral engraving: hide staves with only rests
        "hide_empty_staves": True,
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

PNG_WIDTH = 1280   # all PNGs are rasterized from SVG at this width


def batch_render(raw_dir: Path, render_dir: Path, score_glob: str,
                 exclude: list[str] | None = None,
                 hide_empty_staves: bool = False) -> None:
    """
    Render every score file under raw_dir to SVG via LilyPond, then
    rasterize each SVG page to PNG via cairosvg at fixed PNG_WIDTH.
    The PNG that downstream training sees and the SVG that bbox extraction
    reads come from the same render -> bboxes land on the actual ink.
    """
    # predict.run() lives in /workspace/predict.py inside the container
    sys.path.insert(0, "/workspace")
    from predict import run as lilypond_run   # noqa: PLC0415
    import cairosvg                            # noqa: PLC0415
    from PIL import Image as PILImage          # noqa: PLC0415

    exclude = exclude or []
    all_files = sorted(raw_dir.glob(score_glob))
    score_files = [f for f in all_files
                   if not any(pat in f.name for pat in exclude)]
    print(f"  batch-render: {len(score_files)} files under {raw_dir}"
          f" ({len(all_files) - len(score_files)} excluded)"
          f"{' [hide-empty-staves]' if hide_empty_staves else ''}", flush=True)
    ok = err = skipped = 0

    def _rasterize_dir(out_dir: Path) -> int:
        """Rasterize every score-N.svg in out_dir to score-pageN.png at PNG_WIDTH.
        Returns number of pages rasterized."""
        n = 0
        for svg in sorted(out_dir.glob("score-*.svg")):
            # score-N.svg -> score-pageN.png
            try:
                idx = int(svg.stem.split("-")[-1])
            except ValueError:
                continue
            png = out_dir / f"score-page{idx}.png"
            try:
                cairosvg.svg2png(url=str(svg), write_to=str(png),
                                 output_width=PNG_WIDTH,
                                 background_color="white")
                n += 1
            except Exception as e:
                print(f"    rasterize fail {svg.name}: {e}", flush=True)
        return n

    for sf in tqdm(score_files, desc="  rendering"):
        rel = sf.relative_to(raw_dir)
        out_dir = render_dir / rel.parent / rel.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already rendered (PNGs present).
        if list(out_dir.glob("*.png")):
            skipped += 1
            continue

        try:
            # Step 1: SVG render via LilyPond (vector, DPI-independent).
            svg_pages = lilypond_run(sf, "svg", out_dir,
                                     hide_empty_staves=hide_empty_staves)
            if not svg_pages:
                print(f"    WARN no svg: {rel}", flush=True)
                err += 1
                continue
            # Step 2: rasterize each SVG page to PNG at PNG_WIDTH.
            n_png = _rasterize_dir(out_dir)
            if n_png == 0:
                err += 1
            else:
                ok += 1
        except Exception as e:
            print(f"    ERR {rel}: {e}", flush=True)
            err += 1

    print(f"  Done: {ok} ok, {err} errors, {skipped} skipped", flush=True)


# ---------------------------------------------------------------------------
# Host mode: call Docker once per corpus for batch rendering
# ---------------------------------------------------------------------------

def render_corpus_via_docker(corpus_root: Path, render_dir: Path,
                             score_glob: str, corpus_name: str,
                             exclude: list[str] | None = None,
                             hide_empty_staves: bool = False) -> None:
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
    if hide_empty_staves:
        cmd += ["--hide-empty-staves"]

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

DATASET_CARD = r"""---
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
- 10K<n<100K
configs:
- config_name: pages
  data_files:
  - split: dev
    path: pages/dev-*
  - split: test
    path: pages/test-*
  - split: train
    path: pages/train-*
- config_name: pages-lieder
  data_files:
  - split: train
    path: pages-lieder/train-*
  - split: dev
    path: pages-lieder/dev-*
  - split: test
    path: pages-lieder/test-*
- config_name: pages_transcribed
  data_files:
  - split: dev
    path: pages_transcribed/dev-*
  - split: test
    path: pages_transcribed/test-*
  - split: train
    path: pages_transcribed/train-*
- config_name: scores
  data_files:
  - split: train
    path: scores/train-*
  - split: test
    path: scores/test-*
  - split: dev
    path: scores/dev-*
dataset_info:
- config_name: pages
  features:
  - name: image
    dtype: image
  - name: score_id
    dtype: string
  - name: corpus
    dtype: string
  - name: page
    dtype: int64
  - name: n_pages
    dtype: int64
  - name: composer
    dtype: string
  - name: opus
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 27048193
    num_examples: 339
  - name: test
    num_bytes: 31131909
    num_examples: 478
  - name: train
    num_bytes: 1181562028
    num_examples: 16225
  download_size: 977001946
  dataset_size: 1239742130
- config_name: pages-lieder
  features:
  - name: score_id
    dtype: string
  - name: corpus
    dtype: string
  - name: page
    dtype: int64
  - name: n_pages
    dtype: int64
  - name: bar_start
    dtype: int64
  - name: bar_end
    dtype: int64
  - name: musicxml
    dtype: string
  splits:
  - name: train
    num_bytes: 511960929
    num_examples: 3415
  - name: dev
    num_bytes: 28971937
    num_examples: 195
  - name: test
    num_bytes: 31504305
    num_examples: 218
  download_size: 53578761
  dataset_size: 572437171
- config_name: pages_transcribed
  features:
  - name: score_id
    dtype: string
  - name: corpus
    dtype: string
  - name: page
    dtype: int64
  - name: n_pages
    dtype: int64
  - name: bar_start
    dtype: int64
  - name: bar_end
    dtype: int64
  - name: musicxml
    dtype: string
  - name: image
    dtype: image
  - name: composer
    dtype: string
  - name: opus
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 81654956
    num_examples: 329
  - name: test
    num_bytes: 89237566
    num_examples: 455
  - name: train
    num_bytes: 3160588563
    num_examples: 14129
  download_size: 1135929995
  dataset_size: 3331481085
- config_name: scores
  features:
  - name: score_id
    dtype: string
  - name: composer
    dtype: string
  - name: opus
    dtype: string
  - name: title
    dtype: string
  - name: corpus
    dtype: string
  - name: instruments
    list: string
  - name: page
    dtype: int64
  - name: n_pages
    dtype: int64
  - name: musicxml
    dtype: string
  splits:
  - name: train
    num_bytes: 1468576392
    num_examples: 1424
  - name: test
    num_bytes: 54739890
    num_examples: 79
  - name: dev
    num_bytes: 54992179
    num_examples: 79
  download_size: 154745773
  dataset_size: 1578308461
---

# zzsi/openscore — OpenScore Sheet Music Pages

Rendered sheet music pages from three open-score corpora, paired with
per-page MusicXML ground truth. Intended for optical music recognition (OMR)
research and supervised fine-tuning of vision-language models.

Images are rendered from source MusicXML via
[LilyPond](https://lilypond.org) (Emmentaler font). Per-page MusicXML is
extracted by parsing bar numbers from the rendered SVGs and slicing the source
score with [music21](https://web.mit.edu/music21/).

---

## Corpora

| Corpus | Scores | Instrumentation | Source |
|--------|-------:|-----------------|--------|
| `lieder` | ~1,460 | Voice + piano (3 staves) | [OpenScore/Lieder](https://github.com/OpenScore/Lieder) |
| `quartets` | ~122 | String quartet (4 staves) | [OpenScore/StringQuartets](https://github.com/OpenScore/StringQuartets) |
| `orchestra` | ~94 movements | Full orchestra (10–20+ staves) | [MarkGotham/Hauptstimme](https://github.com/MarkGotham/Hauptstimme) |

---

## ⚠️ Known issue — broken `musicxml` labels in the `quartets` corpus

In the `pages_transcribed` config, the per-page `musicxml` slicing
**fails on a large fraction of `quartets` pages**: the page image shows
real music, but the `musicxml` label is an empty stub (one measure per
part, `<attributes>` + end barline, **no notes**). Verified 2026-05-18
by a full scan of all splits.

| Corpus | Scores w/ empty pages | Empty pages | Genuine failed slices |
|--------|----------------------:|-------------|----------------------:|
| `quartets`  | 97 / 109   | 3,746 / 6,410 (58.4%) | 3,578 |
| `lieder`    | 13 / 1,018 | 70 / 3,392 (2.1%)     | 13 |
| `orchestra` | 14 / 91    | 251 / 5,111 (4.9%)    | 0 (spurious `bar_start > bar_end` pages) |

**Do not train on the `quartets` portion of `pages_transcribed`** until
the slicer is fixed — empty labels teach a model "page of music →
empty output". `lieder` and `orchestra` are usable (`lieder` has 13
stray failed-slice pages, listed below; `orchestra` has zero). The
`pages` config (images only) and `scores` config are unaffected. Key
signatures in non-empty labels are correct.

<details>
<summary>Broken quartet scores — <code>score_id (empty_pages/total_pages)</code> (97)</summary>

sq10307350 (14/26), sq10313029 (53/88), sq10328092 (8/16),
sq10372717 (12/25), sq10381459 (9/20), sq10406164 (53/128),
sq10414906 (13/22), sq10490761 (5/10), sq10517302 (116/198),
sq10527526 (67/115), sq10675759 (74/101), sq11154985 (79/119),
sq11164006 (134/204), sq11539384 (23/61), sq12113164 (21/34),
sq12536479 (50/115), sq12701461 (9/25), sq12772795 (9/25),
sq14720995 (12/25), sq15049456 (18/26), sq15230467 (14/19),
sq15624112 (38/60), sq15730717 (16/22), sq16138966 (49/79),
sq7103818 (46/62), sq7108150 (65/129), sq7114183 (9/21),
sq7123582 (52/88), sq7127785 (44/54), sq7224846 (54/68),
sq7249986 (21/34), sq7284122 (12/20), sq7294793 (33/48),
sq7295726 (63/81), sq7300376 (9/18), sq7302602 (47/59),
sq7302710 (33/65), sq7330550 (52/77), sq7353137 (16/28),
sq7354505 (166/299), sq7358579 (95/125), sq7358708 (22/40),
sq7383977 (14/38), sq7384409 (80/139), sq7397765 (63/161),
sq7434431 (11/17), sq7471661 (14/36), sq7483523 (118/156),
sq7524617 (15/33), sq7541288 (39/68), sq7551068 (17/26),
sq7556360 (81/118), sq7577795 (44/60), sq7588853 (25/38),
sq7872392 (79/127), sq8071278 (23/41), sq8075304 (9/18),
sq8088531 (10/22), sq8437280 (5/14), sq8437358 (4/11),
sq8438840 (2/9), sq8438915 (1/5), sq8438999 (2/10),
sq8454356 (42/68), sq8455808 (15/26), sq8461409 (9/13),
sq8509238 (17/26), sq8556926 (6/12), sq8561633 (42/62),
sq8623643 (64/104), sq8630159 (83/92), sq8796660 (13/20),
sq8806134 (10/14), sq8806746 (7/15), sq8806881 (9/13),
sq8807040 (13/23), sq8807667 (11/17), sq8811375 (74/102),
sq8818128 (68/109), sq8823783 (187/314), sq8853405 (12/31),
sq8885439 (79/175), sq8885571 (72/129), sq8907120 (18/33),
sq8913219 (46/61), sq8938822 (107/151), sq8940236 (21/57),
sq9010547 (50/62), sq9094235 (78/162), sq9137469 (26/52),
sq9146376 (9/17), sq9199617 (6/14), sq9396439 (16/24),
sq9529900 (70/80), sq9608209 (19/43), sq9719026 (41/55),
sq9961690 (15/28).

</details>

The 12 unaffected quartet scores: sq10502527, sq14387632, sq7070319,
sq7070781, sq7075297, sq7078259, sq7082029, sq7093885, sq7095930,
sq7236909, sq7648382, sq8812200.

Lieder scores with one stray failed-slice page each: lc6197282,
lc6486038, lc6593095, lc6613436, lc6613481, lc6614760, lc6624112,
lc6625925, lc6667483, lc6669339, lc6670960, lc6764425, lc8873154.

---

## Configs

### `pages_transcribed` — image + per-page MusicXML (SFT-ready)

Each row is one rendered page paired with the MusicXML for the bars on that
page. Suitable for supervised fine-tuning of OMR models.

| Split | Rows |
|-------|-----:|
| train | 14,129 |
| test  |    455 |
| dev   |    329 |

Fields:

| Field | Type | Description |
|-------|------|-------------|
| `image` | PIL.Image | Full-page score render |
| `score_id` | str | Score identifier (e.g. `lc6583477`) |
| `corpus` | str | `lieder`, `quartets`, or `orchestra` |
| `composer` | str | |
| `opus` | str | |
| `title` | str | |
| `page` | int | 1-indexed page number |
| `n_pages` | int | Total pages in the score |
| `bar_start` | int | First bar number on this page |
| `bar_end` | int | Last bar number on this page (inclusive) |
| `musicxml` | str | MusicXML for `bar_start`–`bar_end` |

---

### `pages` — image only, all corpora

Same rows as `pages_transcribed` but without the `musicxml`, `bar_start`, and
`bar_end` fields. Useful for unsupervised pre-training or image-only tasks.

| Split | Rows |
|-------|-----:|
| train | 16,225 |
| test  |    478 |
| dev   |    339 |

---

### `scores` — full MusicXML per score

One row per score (not per page). Contains the complete MusicXML for the
entire piece plus metadata.

| Split | Rows |
|-------|-----:|
| train | 1,424 |
| test  |    79 |
| dev   |    79 |

Fields: `score_id`, `composer`, `opus`, `title`, `corpus`, `instruments`
(list), `page` (total pages), `n_pages`, `musicxml` (full score).

---

## Usage

### Load `pages_transcribed`

```python
from datasets import load_dataset

ds = load_dataset("zzsi/openscore", "pages_transcribed")
example = ds["train"][0]
example["image"].show()
print(example["musicxml"][:500])
```

### Filter by corpus (streaming)

The dataset is sorted by `corpus` within each split, so row groups in the
parquet files are corpus-homogeneous. This means streaming with a corpus
filter is efficient: non-matching row groups are skipped without being
downloaded.

```python
from datasets import load_dataset

# Lieder only
ds = load_dataset("zzsi/openscore", "pages_transcribed",
                  streaming=True, split="train")
ds = ds.filter(lambda r: r["corpus"] == "lieder")

# Lieder + quartets (no orchestra)
ds = ds.filter(lambda r: r["corpus"] in {"lieder", "quartets"})
```

### Quick subset for testing

```python
# First 100 rows (any corpus)
ds = load_dataset("zzsi/openscore", "pages_transcribed",
                  streaming=True, split="train")
sample = list(ds.take(100))
```

### Fine-tuning example (Qwen-VL style)

```python
from datasets import load_dataset

ds = load_dataset("zzsi/openscore", "pages_transcribed", split="train")

def to_chat(row):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text",  "text": "Transcribe this sheet music page to MusicXML."},
            ]},
            {"role": "assistant", "content": row["musicxml"]},
        ]
    }

ds = ds.map(to_chat)
```

---

## Construction

1. **Render**: Source MusicXML is converted to LilyPond (`.ly`) format and
   rendered to SVG pages using a Docker image containing LilyPond 2.24.
   Bar numbers are made visible on every bar (`all-bar-numbers-visible`).
2. **Align**: Bar numbers are parsed from each SVG page to determine which
   bars appear on each page.
3. **Slice**: music21 slices the source MusicXML to the bar range for each
   page and re-exports it as a self-contained MusicXML fragment.

Pages whose bar numbers could not be reliably parsed (e.g. continuation
pages with no bar number printed) are excluded.

---

## Known Limitations

- **Pickup bars**: Scores with a pickup bar (anacrusis) have an implicit
  measure 0 that is accounted for in `bar_start`/`bar_end`.
- **Orchestra page alignment**: Orchestra scores frequently render to a
  different page count than the original due to `\RemoveEmptyStaves` in
  LilyPond. Alignment is based on bar numbers embedded in the rendered SVG,
  not on page index.
- **MusicXML slice quality**: Sliced MusicXML may be missing some cross-page
  spanners (slurs, hairpins). Inexpressible rhythms (rare) cause individual
  pages to be dropped.
- **Render failures**: ~6% of lieder scores, 3% of quartet scores, and 2
  orchestra movements failed to render and are absent from the dataset.

---

## License

Source scores are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
LilyPond renders and derived MusicXML slices carry the same license.

---

## Attribution

- **OpenScore Lieder** — scores transcribed by the OpenScore community:
  https://github.com/OpenScore/Lieder
- **OpenScore String Quartets** — scores transcribed by the OpenScore community:
  https://github.com/OpenScore/StringQuartets
- **Hauptstimme (Orchestra)** — scores curated by Mark Gotham:
  https://github.com/MarkGotham/Hauptstimme

Rendering pipeline uses [LilyPond](https://lilypond.org) and
[music21](https://web.mit.edu/music21/). Dataset construction code:
https://github.com/zhudotexe/CVlization
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
    parser.add_argument("--hide-empty-staves", action="store_true",
        help=argparse.SUPPRESS)

    args = parser.parse_args()
    random.seed(args.seed)

    # -----------------------------------------------------------------
    # Batch-render mode: runs inside Docker
    # -----------------------------------------------------------------
    if args.batch_render:
        batch_render(
            raw_dir            = Path(args.raw_dir),
            render_dir         = Path(args.render_dir),
            score_glob         = args.score_glob,
            exclude            = args.exclude,
            hide_empty_staves  = args.hide_empty_staves,
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
            corpus_root        = corpus_root,
            render_dir         = render_dir,
            score_glob         = meta["score_glob"],
            corpus_name        = corpus_name,
            exclude            = meta.get("exclude", []),
            hide_empty_staves  = meta.get("hide_empty_staves", False),
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
