#!/usr/bin/env python3
"""End-to-end per-measure OMR pipeline.

  page image
    -> multi-task YOLO (structural classes + 15 keysig sub-classes)
       -> systems, staves, barlines, keysig boxes (with key value)
    -> cells.derive_measures   -> list[Measure]   (one bbox per measure,
                                                   spanning all staves)
    -> per-measure VLM, one call per Measure bbox
       (instruction = 'Transcribe this single measure ...')
    -> stitch_measures: concatenate per-measure outputs into page MXC2
    -> respell.respell_mxc2(page_mxc2, page_key) -> corrected MXC2

The detected `key_signature` sub-class id encodes the key (class id 4
= key_-7, class id 18 = key_+7); page key = majority vote across all
detected keysig boxes.

Usage (inside Docker via the per-measure docker image):
    python pipeline_per_measure.py --image page.png \\
        --det-ckpt /det/outputs/detector_mt/run/weights/best.pt \\
        --vlm-ckpt outputs/per_measure_v2/final_model \\
        --out page.mxc2
"""

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Local + sibling-folder imports
_THIS = Path(__file__).resolve()
_DET = _THIS.parent.parent / "omr_detection"
_VLM = _THIS.parent.parent / "vlm_omr_sft"
sys.path.insert(0, str(_DET))
sys.path.insert(0, str(_VLM))

from cells import derive_measures  # noqa: E402
from respell import respell_mxc2  # noqa: E402


INSTRUCTION = ("Transcribe this single measure of sheet music to MXC2 "
               "(compact MusicXML). Output only this measure, including "
               "every part's content for it.")
# Detector keysig sub-classes are 4..18 with fifths = class - 4 - 7.
_KEYSIG_BASE = 4


def _default_parts_for_n_staves(n_staves: int) -> list[tuple[str, str]]:
    """Heuristic (part_name, clef_spec) for systems whose part structure
    is unknown at inference. Covers the common openscore / synthetic
    lieder layouts: 1 staff = solo voice; 2 = voice + piano; 3 = voice +
    grand-staff piano; 4 = voice+voice+grand-staff piano.
    """
    if n_staves <= 1:
        return [("Voice", "G2")]
    if n_staves == 2:
        return [("Voice", "G2"), ("Piano", "G2,F4")]
    if n_staves == 3:
        return [("Voice", "G2"), ("Piano RH", "G2"), ("Piano LH", "F4")]
    if n_staves == 4:
        return [("Voice 1", "G2"), ("Voice 2", "G2"),
                ("Piano RH", "G2"), ("Piano LH", "F4")]
    # 5+: voice + piano + extras (default everything else to treble)
    out = [("Voice", "G2"), ("Piano RH", "G2"), ("Piano LH", "F4")]
    for i in range(n_staves - 3):
        out.append((f"Other {i+1}", "G2"))
    return out


def _build_active_header(key_fifths: int | None, n_staves: int,
                         time_str: str = "4/4") -> str:
    """Match training-time prompt format. Without per-page time/clef
    detection we use defaults: 4/4 time and the most common clef layout
    for the staff count."""
    if key_fifths is None:
        key_fifths = 0
    parts = _default_parts_for_n_staves(n_staves)
    lines = ["Active header per part:"]
    for i, (name, clef) in enumerate(parts, 1):
        lines.append(f"P{i} {name}: key={key_fifths} time={time_str} "
                     f"clef={clef}")
    return "\n".join(lines)


def _fifths_from_cls(cls: int) -> int | None:
    if cls < _KEYSIG_BASE:
        return None
    return (cls - _KEYSIG_BASE) - 7


def detect_layout_and_key(det, image_path: str, imgsz: int = 1280,
                          conf: float = 0.25):
    """Run multi-task detector; return (systems, staves, barlines,
    predicted_key_or_none)."""
    res = det.predict(source=image_path, imgsz=imgsz, conf=conf,
                       verbose=False)[0]
    systems, staves, barlines = [], [], []
    keysig_votes: list[int] = []
    for box, cls in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
        x1, y1, x2, y2 = box
        xywh = (x1, y1, x2 - x1, y2 - y1)
        c = int(cls)
        if c == 0:
            systems.append(xywh)
        elif c == 1:
            staves.append(xywh)
        elif c in (2, 3):
            barlines.append(xywh)
        elif c >= _KEYSIG_BASE:
            f = _fifths_from_cls(c)
            if f is not None:
                keysig_votes.append(f)
    key = (Counter(keysig_votes).most_common(1)[0][0]
           if keysig_votes else None)
    return systems, staves, barlines, key, keysig_votes


def transcribe_measure(model, processor, page_pil, measure, pad_px=4,
                       pad_vertical_frac: float = 0.3,
                       active_header: str | None = None):
    """Crop the measure region and run the per-measure VLM.

    `pad_vertical_frac` must match the value used at training time
    (build_per_measure_dataset.py default 0.3 for v4); otherwise the
    inference crop omits the lyrics / dynamics that live below the staff
    and the model produces note hallucinations.

    `active_header` is the per-part key/time/clef context the v4 model
    was trained to read (see train_per_measure._make_prompt). When None,
    falls back to the bare instruction (matches pre-v4 behavior)."""
    import torch
    x, y, w, h = measure.bbox
    pad_x = max(pad_px, int(0.01 * page_pil.width))
    pad_y = max(pad_px, int(0.01 * page_pil.height + pad_vertical_frac * h))
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(page_pil.width, int(x + w + pad_x))
    bottom = min(page_pil.height, int(y + h + pad_y))
    crop = page_pil.crop((left, top, right, bottom)).convert("RGB")
    prompt = INSTRUCTION + ("\n\n" + active_header if active_header else "")
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image"}]}]
    txt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp = processor(crop, txt, add_special_tokens=False,
                     return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=512, use_cache=True,
                              do_sample=False)
    return processor.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()


_RE_M_LINE = re.compile(r"^M\s+(\S+)(.*)$")


def _per_measure_parts(measure_mxc2: str) -> dict[int, list[str]]:
    """Split one per-measure MXC2 output into {part_idx: [body lines]}.

    Expects the format produced by build_per_measure_dataset:
        P1 X
        P2 X
        ---
        P1
        M N ...
        N ...
        P2
        M N ...
    Returns {1: [...], 2: [...]} -- part body lines after 'PN'.
    Skips header (everything before '---') and the 'M N' lines themselves
    -- caller will renumber.
    """
    out: dict[int, list[str]] = {}
    lines = measure_mxc2.splitlines()
    # skip until '---'
    i = 0
    while i < len(lines) and lines[i].strip() != "---":
        i += 1
    i += 1  # past the ---
    cur_part = None
    cur_body: list[str] = []
    cur_m_attrs: str | None = None
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^P(\d+)\s*$", line.strip())
        if m:
            if cur_part is not None:
                out[cur_part] = (cur_m_attrs, cur_body)
            cur_part = int(m.group(1))
            cur_body = []
            cur_m_attrs = None
        else:
            ml = _RE_M_LINE.match(line)
            if ml is not None and cur_m_attrs is None:
                cur_m_attrs = ml.group(2).strip()
            else:
                cur_body.append(line)
        i += 1
    if cur_part is not None:
        out[cur_part] = (cur_m_attrs, cur_body)
    return out


def stitch_measures(per_measure_outputs: list[tuple],
                     header_block: str | None = None) -> str:
    """Concatenate per-measure outputs in reading order into one page MXC2.

    Each `per_measure_outputs` entry is (measure_num, measure_mxc2_str).
    Returns:
        P1 ...
        P2 ...
        ---
        P1
        M 1 <attrs>
        <body>
        M 2 <attrs>
        ...
        P2
        M 1 <attrs>
        ...
    """
    by_part: dict[int, list[tuple[int, str | None, list[str]]]] = {}
    for m_num, txt in per_measure_outputs:
        parts = _per_measure_parts(txt)
        for p_idx, (m_attrs, body) in parts.items():
            by_part.setdefault(p_idx, []).append((m_num, m_attrs, body))

    if not by_part:
        return ""

    out: list[str] = []
    if header_block:
        for line in header_block.splitlines():
            out.append(line)
    else:
        # synthesize minimal header from observed part indices
        for p_idx in sorted(by_part):
            out.append(f"P{p_idx}")
        out.append("---")
    for p_idx in sorted(by_part):
        out.append(f"P{p_idx}")
        for m_num, m_attrs, body in by_part[p_idx]:
            attrs = f" {m_attrs}" if m_attrs else ""
            out.append(f"M {m_num}{attrs}".rstrip())
            for line in body:
                if line.strip():
                    out.append(line)
    return "\n".join(out) + "\n"


def run(image_path: str, det_ckpt: str, vlm_ckpt: str,
        imgsz: int = 1280, conf: float = 0.25,
        bar_start: int = 1, verbose: bool = False) -> str:
    """End-to-end per-measure inference. Returns final MXC2."""
    from ultralytics import YOLO
    from unsloth import FastVisionModel
    from PIL import Image

    det = YOLO(det_ckpt)
    vlm, processor = FastVisionModel.from_pretrained(vlm_ckpt,
                                                       load_in_4bit=True)
    FastVisionModel.for_inference(vlm)

    systems, staves, barlines, key, votes = detect_layout_and_key(
        det, image_path, imgsz=imgsz, conf=conf)
    measures = derive_measures(systems, staves, barlines)
    if verbose:
        print(f"  detected {len(systems)} systems, {len(staves)} staves, "
              f"{len(barlines)} barlines, key votes={votes} -> {key}",
              flush=True)
        print(f"  derived {len(measures)} measures", flush=True)

    page_pil = Image.open(image_path).convert("RGB")
    per_measure: list[tuple] = []
    t0 = time.time()
    # Build active header per system based on its staff count.
    # All measures within a system share the same header layout.
    header_by_sys: dict[int, str] = {}
    staves_per_sys: dict[int, int] = {}
    for s in staves:
        # staves are (sys_i, idx, x, y, w, h)
        staves_per_sys[s[0]] = staves_per_sys.get(s[0], 0) + 1
    for sys_i, n_st in staves_per_sys.items():
        header_by_sys[sys_i] = _build_active_header(key, n_st)
    for i, m in enumerate(measures):
        abs_m = bar_start + i
        header = header_by_sys.get(m.system, _build_active_header(key, 3))
        try:
            txt = transcribe_measure(vlm, processor, page_pil, m,
                                     active_header=header)
        except Exception as e:
            if verbose:
                print(f"  m={abs_m} transcribe err: {e}", flush=True)
            continue
        per_measure.append((abs_m, txt))
        if verbose and (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(measures)}] {time.time()-t0:.1f}s",
                  flush=True)

    stitched = stitch_measures(per_measure)
    if key is None:
        return stitched
    try:
        return respell_mxc2(stitched, key)
    except Exception as e:
        if verbose:
            print(f"  respell err: {e}", flush=True)
        return stitched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--det-ckpt", required=True)
    p.add_argument("--vlm-ckpt", required=True)
    p.add_argument("--out", default=None,
                   help="write final MXC2 here (else print)")
    p.add_argument("--bar-start", type=int, default=1,
                   help="for openscore pages where measures start > 1")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    mxc = run(args.image, args.det_ckpt, args.vlm_ckpt,
              imgsz=args.imgsz, conf=args.conf,
              bar_start=args.bar_start, verbose=args.verbose)
    if args.out:
        Path(args.out).write_text(mxc)
        print(f"wrote {args.out}", flush=True)
    else:
        print(mxc)


if __name__ == "__main__":
    main()
