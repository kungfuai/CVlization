#!/usr/bin/env python3
"""
Zero-shot OMR probing with frontier VLMs via LiteLLM.

Sends 5 music-theory prompts to a configurable frontier VLM (default:
gemini/gemini-2.5-flash) on a sheet-music image. Supports single-image
mode (JSON output) and batch/directory mode (JSONL output with --resume).

Usage:
  python predict.py                                    # Gemini 2.5 Flash, default sample
  python predict.py --image my_score.jpg
  python predict.py --model gpt-4o
  python predict.py --model anthropic/claude-opus-4-6
  python predict.py --model anthropic/claude-sonnet-4-6
  python predict.py --input-dir ./scans/ --output omr_results.jsonl --resume
  python predict.py --prompts key_signature time_signature
"""

import argparse
import base64
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import litellm
from PIL import Image

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except ImportError:
    def resolve_input_path(p, **kw):
        return p

    def resolve_output_path(p=None, default_filename="result.txt", **kw):
        return p or default_filename


# ---------------------------------------------------------------------------
# Prompts — identical to qwen3_vl for direct comparison
# ---------------------------------------------------------------------------
PROMPTS = [
    {
        "id": "key_signature",
        "label": "Key signature",
        "text": (
            "Look at this sheet music image. What is the key signature? "
            "Count the sharps or flats shown at the beginning of each staff line "
            "and name the major or minor key (e.g. 'G major', 'D minor', "
            "'no sharps or flats — C major')."
        ),
        "expectation": "Expected to work well — key signature is a clear visual feature.",
    },
    {
        "id": "time_signature",
        "label": "Time signature",
        "text": (
            "Look at this sheet music image. What is the time signature? "
            "Read the two stacked numbers shown at the start of the piece "
            "(e.g. 4/4, 3/4, 6/8) and explain what it means rhythmically."
        ),
        "expectation": "Expected to work well — time signature digits are visually salient.",
    },
    {
        "id": "musical_era",
        "label": "Musical era",
        "text": (
            "Looking at the engraving style, paper quality, typography, and notation "
            "conventions in this sheet music image, what musical era or approximate time "
            "period does it appear to be from? Give your best estimate (e.g. Baroque, "
            "Classical, Romantic, early 20th century)."
        ),
        "expectation": "Moderate — depends on whether the image has clear period-specific features.",
    },
    {
        "id": "dynamics_tempo",
        "label": "Dynamics and tempo markings",
        "text": (
            "List all dynamics markings (e.g. p, f, mf, pp, ff, crescendo, decrescendo) "
            "and tempo / expression markings (e.g. Allegro, Andante, poco a poco, dolce) "
            "visible in this sheet music image. For each, note approximately where it appears."
        ),
        "expectation": "Moderate — depends on image resolution and marking visibility.",
    },
    {
        "id": "ekern_transcription",
        "label": "Full kern transcription with embedded metadata",
        "text": (
            "Transcribe this piano sheet music into a complete **kern file. "
            "Output ONLY the kern file — no prose, no explanation, no code fences.\n\n"
            "RULES:\n"
            "- Every data row has exactly TWO tab-separated columns: left=bass clef, right=treble clef.\n"
            "- *> section lines must appear in BOTH columns: *>Label TAB *>Label\n"
            "- Barlines must also have two columns: =1<TAB>=1\n\n"
            "CONCRETE EXAMPLE — illustrates FORMAT only, not content "
            "(TAB = actual tab character between the two columns):\n"
            "!!!OTL: Title exactly as printed\n"
            "!!!COM: Full composer attribution line exactly as printed\n"
            "!!!OMD: Andante                <- opening tempo word read from image\n"
            "!!!YEC: Copyright line if visible, else omit\n"
            "**kern TAB **kern\n"
            "*clefF4 TAB *clefG2\n"
            "*k[f#] TAB *k[f#]       <- read key from image: *k[]=C, *k[f#]=G, *k[b-]=F, etc.\n"
            "*M4/4 TAB *M4/4         <- read time signature from image\n"
            "*>Section_A TAB *>Section_A  <- section label; use underscore for spaces; read name from image\n"
            "=1 TAB =1\n"
            "2G 4D TAB 4d 4f# 4a\n"
            "4G 4D TAB 2g\n"
            "=2 TAB =2\n"
            "...\n"
            "*k[b-b-] TAB *k[b-b-]       <- if key changes mid-piece, emit new *k[] here\n"
            "*>Section_B TAB *>Section_B  <- new section label read from image\n"
            "=9 TAB =9\n"
            "4A 4E TAB 4a 4cc\n"
            "...\n"
            "*- TAB *-\n\n"
            "PITCH GUIDE:\n"
            "- Uppercase = below middle C: C D E F G A B (one octave below)\n"
            "  Double uppercase = two octaves below: CC DD FF GG (e.g. CC = low C)\n"
            "- Lowercase = above middle C: c d e f g a b\n"
            "  Double lowercase = two octaves above: cc dd ee ff gg\n"
            "- Durations: 1=whole 2=half 4=quarter 8=eighth 16=sixteenth\n"
            "- Rests: r (e.g. 4r = quarter rest)\n"
            "- Beaming: L=beam start, J=beam end (e.g. 8cL 8dJ)\n"
            "- Chords: space-separated (e.g. 4C 4E 4G = C major chord)\n"
            "- Repeat barline: =:|!\n\n"
            "Now transcribe the actual score image. "
            "Read the title, composer, copyright, section names, key, and tempo directly from the image. "
            "If the key changes mid-piece, emit the new *k[] record at that point. "
            "Transcribe as many measures as you can read accurately."
        ),
        "expectation": (
            "Metadata and structure expected to succeed; exact pitches may hallucinate. "
            "Output uses music21-native kern (*> section labels, !!!OMD: tempo) "
            "for direct rendering via LilyPond (kern_to_ly pipeline)."
        ),
    },
]

PROMPT_IDS = [p["id"] for p in PROMPTS]
DEFAULT_MODEL = "gemini/gemini-2.5-flash"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "cvl_data" / "qwen3_omr"
SAMPLE_HF_PATH = "qwen3_omr/vintage_score_1884.jpg"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def encode_image_base64(image_path: str) -> tuple[str, str]:
    """Load image and return (base64_string, mime_type) as JPEG."""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def _is_openai_model(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")


def make_message(prompt_text: str, b64: str, mime_type: str, model: str = "") -> dict:
    """Build a LiteLLM-compatible user message with image + text."""
    image_url: dict = {"url": f"data:{mime_type};base64,{b64}"}
    if _is_openai_model(model):
        image_url["detail"] = "high"
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": image_url},
            {"type": "text", "text": prompt_text},
        ],
    }


# ---------------------------------------------------------------------------
# LiteLLM call
# ---------------------------------------------------------------------------

def run_prompt(model: str, message: dict, max_tokens: int = 1024) -> str:
    response = litellm.completion(
        model=model,
        messages=[message],
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Sample image
# ---------------------------------------------------------------------------

def ensure_sample_image() -> Path:
    """Download vintage 1884 sample from zzsi/cvl (same as qwen3_vl example)."""
    cache_path = CACHE_DIR / Path(SAMPLE_HF_PATH).name
    if cache_path.exists():
        return cache_path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading vintage sample from HuggingFace (zzsi/cvl) ...")
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id="zzsi/cvl",
        repo_type="dataset",
        filename=SAMPLE_HF_PATH,
        local_dir=str(CACHE_DIR.parent),
    )
    return Path(local)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _prompt_map() -> dict:
    return {p["id"]: p for p in PROMPTS}


def select_prompts(ids: list[str] | None) -> list[dict]:
    if not ids:
        return PROMPTS
    pm = _prompt_map()
    return [pm[i] for i in ids if i in pm]


def _preview(text: str, width: int = 120) -> str:
    flat = text.replace("\n", " ")
    return flat[:width] + ("..." if len(flat) > width else "")


def print_summary(record: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Model : {record['model']}")
    print(f"Image : {record['file']}")
    for pid, r in record["results"].items():
        print(f"\n[{pid}] {r['label']}")
        print(_preview(r["response"], 400))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Single-image mode
# ---------------------------------------------------------------------------

def run_single(args) -> None:
    if args.image:
        image_path = resolve_input_path(args.image)
    else:
        image_path = str(ensure_sample_image())
        print(f"Using vintage sample: {image_path}")
        print("  Source: Biddle's Piano Waltz (1884), Library of Congress")

    output_path = Path(resolve_output_path(args.output, default_filename="omr_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Encoding image ...")
    b64, mime = encode_image_base64(image_path)

    selected = select_prompts(args.prompts)
    results = {}
    for i, prompt in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {prompt['label']} ...")
        msg = make_message(prompt["text"], b64, mime, args.model)
        try:
            response = run_prompt(args.model, msg, args.max_tokens)
        except Exception as e:
            response = f"ERROR: {e}"
        results[prompt["id"]] = {
            "label": prompt["label"],
            "expectation": prompt["expectation"],
            "prompt": prompt["text"],
            "response": response,
        }
        print(f"  {_preview(response)}")

    record = {
        "file": str(image_path),
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {output_path}")
    print_summary(record)


# ---------------------------------------------------------------------------
# Batch/directory mode
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def run_batch(args) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"ERROR: --input-dir '{input_dir}' is not a directory.")

    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        sys.exit(f"No image files found in {input_dir}")

    output_path = Path(resolve_output_path(args.output, default_filename="omr_results.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect filenames already in output
    done: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(Path(json.loads(line)["file"]).name)
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resume: {len(done)} files already done, skipping.")

    selected = select_prompts(args.prompts)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for idx, img_path in enumerate(image_files):
            fname = img_path.name
            if fname in done:
                print(f"[{idx+1}/{len(image_files)}] Skip: {fname}")
                continue

            print(f"[{idx+1}/{len(image_files)}] {fname}")
            try:
                b64, mime = encode_image_base64(str(img_path))
            except Exception as e:
                print(f"  ERROR encoding: {e}", file=sys.stderr)
                continue

            results = {}
            for prompt in selected:
                msg = make_message(prompt["text"], b64, mime, args.model)
                try:
                    response = run_prompt(args.model, msg, args.max_tokens)
                except Exception as e:
                    response = f"ERROR: {e}"
                results[prompt["id"]] = {"label": prompt["label"], "response": response}
                print(f"  [{prompt['id']}] {_preview(response, 80)}")

            record = {
                "file": fname,
                "model": args.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()  # durable after each image

    print(f"\nBatch complete. Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot OMR probing with frontier VLMs via LiteLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  gemini/gemini-2.5-flash      (default, cheapest)
  gpt-4o
  anthropic/claude-opus-4-6
  anthropic/claude-sonnet-4-6

API keys (set in host environment, forwarded by predict.sh):
  GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
        """,
    )
    parser.add_argument("--image", default=None,
        help="Path to a single sheet music image (single-image mode)")
    parser.add_argument("--input-dir", default=None,
        help="Directory of images for batch mode (outputs JSONL)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
        help=f"LiteLLM model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=None,
        help="Output path (default: omr_results.json / omr_results.jsonl)")
    parser.add_argument("--max-tokens", type=int, default=1024,
        help="Max tokens per response (default: 1024)")
    parser.add_argument("--prompts", nargs="+", choices=PROMPT_IDS, default=None,
        help="Run only specific prompts (default: all 5)")
    parser.add_argument("--resume", action="store_true",
        help="Batch mode: skip files already in output JSONL")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_dir:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
