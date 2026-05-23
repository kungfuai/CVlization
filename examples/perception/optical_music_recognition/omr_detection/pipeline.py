#!/usr/bin/env python3
"""End-to-end detection-first OMR pipeline.

Inference flow:
  page image
    -> YOLO layout detector  -> {systems, staves, barlines}
    -> derive_cells()        -> measure cells (staff x measure)
    -> [transcribe each cell]  (HOOK -- not wired here; transcription
       lives in vlm_omr_sft, per the detection-workstream scope)
    -> stitch                -> page-level MXC2

This module implements detection + cell derivation. `transcribe_cell` is
a documented integration point that raises NotImplementedError.

It doubles as an inspection tool: with --crops-dir it crops every
detected instance (and every derived cell) to disk, and with --inspect
it builds a per-class montage of a random subset so the detections can
be eyeballed.

Usage (inside Docker via detect.sh):
    python pipeline.py --image page.png --checkpoint best.pt \\
        --crops-dir /tmp/crops --inspect 12
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from cells import Cell, Measure, derive_cells, derive_measures, measures_per_system

# Where vlm_omr_sft lives (sibling folder). Imported lazily by the
# transcription helpers; detection-only paths don't touch it.
_THIS_DIR = Path(__file__).resolve().parent
_VLM_DIR = _THIS_DIR.parent / "vlm_omr_sft"

# Detector class ids (see train_detector.CLASS_NAMES).
CLS_SYSTEM, CLS_STAFF, CLS_BARLINE, CLS_BARLINE_HEAVY = 0, 1, 2, 3

# Horizontal/vertical padding (px) applied when cropping for inspection,
# per class. Barlines are ~8 px wide -- useless to eyeball without context.
INSPECT_PAD = {
    "systems": 6,
    "staves": 6,
    "barlines": 40,
    "cells": 4,
}


def load_detector(checkpoint: str):
    from ultralytics import YOLO  # lazy: keeps --help torch-free
    return YOLO(checkpoint)


def detect_layout(model, image_path: str, imgsz: int = 1280,
                  conf: float = 0.25) -> dict:
    """Run the detector and return boxes grouped by class.

    Returns dict with lists of (x, y, w, h) page-pixel boxes:
        {"systems", "staves", "barlines", "barlines_heavy"}
    `barlines` merges single + heavy (both are measure boundaries);
    `barlines_heavy` is kept separately for reporting.
    """
    res = model.predict(source=image_path, imgsz=imgsz, conf=conf,
                         verbose=False)[0]
    out = {"systems": [], "staves": [], "barlines": [], "barlines_heavy": []}
    for box, cls in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
        x1, y1, x2, y2 = box
        xywh = (x1, y1, x2 - x1, y2 - y1)
        cls = int(cls)
        if cls == CLS_SYSTEM:
            out["systems"].append(xywh)
        elif cls == CLS_STAFF:
            out["staves"].append(xywh)
        elif cls == CLS_BARLINE:
            out["barlines"].append(xywh)
        elif cls == CLS_BARLINE_HEAVY:
            out["barlines"].append(xywh)
            out["barlines_heavy"].append(xywh)
    return out


# ---------------------------------------------------------------------------
# Transcription via the safckylj VLM (vlm_omr_sft).
# ---------------------------------------------------------------------------
# These helpers import unsloth + the vlm_omr_sft module lazily so the
# detection-only entry points stay light. They must run inside the
# cvlization/omr-pipeline image (vlm-omr-sft base + ultralytics).


def load_vlm(checkpoint: str):
    """Load the safckylj VLM checkpoint via unsloth FastVisionModel."""
    sys.path.insert(0, str(_VLM_DIR))
    from unsloth import FastVisionModel  # type: ignore
    model, processor = FastVisionModel.from_pretrained(
        checkpoint, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    return model, processor


def _vlm_generate(model, processor, image_pil, instruction: str,
                  max_new_tokens: int = 8192) -> str:
    """One inference call. Returns the model's generated text."""
    sys.path.insert(0, str(_VLM_DIR))
    from train import prepare_inference_inputs  # type: ignore
    import torch
    inputs = prepare_inference_inputs(processor, image_pil,
                                       instruction).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              use_cache=True, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


_PAGE_INSTRUCTION = "Transcribe this sheet music page to MXC2 (compact MusicXML)."
_MEASURE_INSTRUCTION = (
    "Transcribe this single measure of sheet music to MXC2 "
    "(compact MusicXML). Output only the measure(s) shown.")


def transcribe_page(model, processor, image_pil) -> str:
    """Whole-page transcription -- the model's native training mode."""
    return _vlm_generate(model, processor, image_pil, _PAGE_INSTRUCTION)


def transcribe_measure(model, processor, page_image_pil, measure: Measure,
                       pad: int = 8) -> str:
    """Per-measure transcription on a measure crop.

    The model was trained on whole pages; cropping to a single measure is
    off-distribution and quality is expected to degrade. Kept here so the
    architectural plumbing is real and future per-cell-trained models can
    drop in.
    """
    from PIL import Image  # noqa: F401
    crop = _crop(page_image_pil, measure.bbox, pad)
    return _vlm_generate(model, processor, crop, _MEASURE_INSTRUCTION,
                          max_new_tokens=2048)


def stitch_measures_naive(measure_outputs: list[tuple[Measure, str]]) -> str:
    """Best-effort stitch of per-measure MXC2 fragments into a page MXC2.

    Strategy: take the part declarations + separator from the first
    non-empty fragment, then concatenate the per-part-per-measure bodies
    in reading order with `M <num>` renumbered globally. Honestly: the
    underlying per-measure outputs are off-distribution, so this is more
    of a wiring demonstration than a quality path.
    """
    if not measure_outputs:
        return ""
    # Use header (P-declarations + ---) from the first fragment that has one.
    header_lines: list[str] = []
    for _, text in measure_outputs:
        if "---" in text:
            head = text.split("---", 1)[0].rstrip()
            header_lines = head.splitlines()
            break
    out = []
    if header_lines:
        out.extend(header_lines)
        out.append("---")
    out.append("P1")
    running = 1
    for _, text in measure_outputs:
        body = text.split("---", 1)[1] if "---" in text else text
        # Drop any leading P-line; keep only this measure's M body.
        body = re.sub(r"^\s*P\d+.*\n", "", body, count=1)
        # Take just the first M-section if multiple were emitted.
        chunks = re.split(r"(?=^M )", body, flags=re.MULTILINE)
        m_chunk = next((c for c in chunks if c.strip().startswith("M ")), "")
        if not m_chunk:
            continue
        m_chunk = re.sub(r"^M\s+\S+", f"M {running}", m_chunk, count=1)
        out.append(m_chunk.rstrip())
        running += 1
    return "\n".join(out) + "\n"


def transcribe_cell(cell_crop) -> str:
    """Deprecated: prefer transcribe_measure (per-measure across staves)."""
    raise NotImplementedError(
        "Per-(staff,measure) cell transcription is not wired -- use "
        "transcribe_measure on a derived Measure box instead.")


def _crop(img, box, pad: int):
    """Crop a padded (x,y,w,h) region, clamped to the image."""
    x, y, w, h = box
    x0 = max(0, int(x - pad))
    y0 = max(0, int(y - pad))
    x1 = min(img.width, int(x + w + pad))
    y1 = min(img.height, int(y + h + pad))
    return img.crop((x0, y0, x1, y1))


def dump_crops(image_path: str, layout: dict, cells: list[Cell],
               out_dir: Path) -> dict[str, int]:
    """Crop every detected instance + derived cell to <out_dir>/<class>/."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    groups = {
        "systems": [(b, b) for b in layout["systems"]],
        "staves": [(b, b) for b in layout["staves"]],
        "barlines": [(b, b) for b in layout["barlines"]],
        "cells": [(c.bbox, c) for c in cells],
    }
    counts: dict[str, int] = {}
    for name, items in groups.items():
        cls_dir = out_dir / name
        cls_dir.mkdir(parents=True, exist_ok=True)
        pad = INSPECT_PAD[name]
        for i, (box, meta) in enumerate(items):
            crop = _crop(img, box, pad)
            if name == "cells":
                fname = f"sys{meta.system}_st{meta.staff}_m{meta.measure}.png"
            else:
                fname = f"{name[:-1]}_{i:03d}.png"
            crop.save(cls_dir / fname)
        counts[name] = len(items)
    return counts


def build_montage(crops_dir: Path, klass: str, n: int, seed: int = 0) -> Path:
    """Tile a random subset of one class's crops into a single image."""
    from PIL import Image
    cls_dir = crops_dir / klass
    files = sorted(cls_dir.glob("*.png"))
    if not files:
        return None
    random.Random(seed).shuffle(files)
    files = files[:n]

    row_h = 90  # normalize every crop to this height
    tiles = []
    for f in files:
        im = Image.open(f).convert("RGB")
        scale = row_h / im.height
        tiles.append((f.stem, im.resize((max(1, int(im.width * scale)), row_h))))

    pad, label_h = 6, 14
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    col_w = max(t.width for _, t in tiles) + pad
    cell_h = row_h + label_h + pad
    canvas = Image.new("RGB", (cols * col_w + pad, rows * cell_h + pad),
                       "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for idx, (name, tile) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = pad + c * col_w
        y = pad + r * cell_h
        draw.text((x, y), name, fill="black")
        canvas.paste(tile, (x, y + label_h))
    out = crops_dir / f"_montage_{klass}.png"
    canvas.save(out)
    return out


def run_pipeline(image_path: str, checkpoint: str, imgsz: int = 1280,
                 conf: float = 0.25) -> dict:
    """Detect + derive cells for one page. Returns a layout dict."""
    model = load_detector(checkpoint)
    layout = detect_layout(model, image_path, imgsz=imgsz, conf=conf)
    cells = derive_cells(layout["systems"], layout["staves"],
                         layout["barlines"])
    return {"layout": layout, "cells": cells}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="page image (PNG)")
    p.add_argument("--checkpoint", required=True, help="YOLO .pt weights")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--crops-dir", type=Path, default=None,
                   help="dump detected instances + cells here")
    p.add_argument("--inspect", type=int, default=0,
                   help="if >0, build a per-class montage of N random crops")
    p.add_argument("--out-json", type=Path, default=None,
                   help="write the layout + cells as JSON")
    args = p.parse_args()

    result = run_pipeline(args.image, args.checkpoint, args.imgsz, args.conf)
    layout, cells = result["layout"], result["cells"]

    print(f"Detected: {len(layout['systems'])} systems, "
          f"{len(layout['staves'])} staves, "
          f"{len(layout['barlines'])} barlines "
          f"({len(layout['barlines_heavy'])} heavy)", flush=True)
    print(f"Derived:  {len(cells)} cells; "
          f"measures/system = {measures_per_system(cells)}", flush=True)

    if args.out_json:
        args.out_json.write_text(json.dumps({
            "image": args.image,
            "layout": layout,
            "cells": [{"system": c.system, "staff": c.staff,
                       "measure": c.measure, "bbox": c.bbox} for c in cells],
        }, indent=2))
        print(f"Wrote {args.out_json}", flush=True)

    if args.crops_dir:
        counts = dump_crops(args.image, layout, cells, args.crops_dir)
        print(f"Crops -> {args.crops_dir}: {counts}", flush=True)
        if args.inspect > 0:
            for klass in ("systems", "staves", "barlines", "cells"):
                m = build_montage(args.crops_dir, klass, args.inspect)
                if m:
                    print(f"  montage: {m}", flush=True)


if __name__ == "__main__":
    main()
