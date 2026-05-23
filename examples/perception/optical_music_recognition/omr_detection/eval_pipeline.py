#!/usr/bin/env python3
"""End-to-end pipeline quality: detect -> transcribe -> stitch.

Runs the full pipeline on N L7a dev pages and reports MXC2 similarity
vs ground truth, in two modes:

  page  -- whole-image VLM call (the model's native training mode)
  measure -- per-measure VLM calls on cell crops, then naive stitch
             (off-distribution for safckylj; reported for comparison)

Usage (inside the omr-pipeline image via eval_pipeline.sh):
    python eval_pipeline.py --data /data --det-ckpt <yolo> \\
        --vlm-ckpt <safckylj> --n 10 --mode page
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Importing pipeline transcribes via vlm_omr_sft; ensure that path is on
# sys.path so its `train`, `mxc2`, `eval_mxc` are importable here too.
_THIS = Path(__file__).resolve().parent
_VLM_DIR = _THIS.parent / "vlm_omr_sft"
sys.path.insert(0, str(_VLM_DIR))
sys.path.insert(0, str(_THIS))

from cells import derive_measures  # noqa: E402
from pipeline import (  # noqa: E402
    detect_layout, load_detector, load_vlm,
    predict_keys_from_detections,
    transcribe_page, transcribe_measure, stitch_measures_naive,
)


def _load_gt_mxc2(mxl_text: str, drop_beams: bool = True) -> str:
    from mxc2 import xml_to_mxc2  # type: ignore
    from train import strip_musicxml_header  # type: ignore
    return xml_to_mxc2(strip_musicxml_header(mxl_text), drop_beams=drop_beams)


def _load_keysig_cnn(ckpt_path: str, on_crop: bool = False):
    """Load the small CNN key classifier.

    on_crop=False: expects the full page and uses the legacy fixed
        top-left fraction (crop_keysig).
    on_crop=True:  expects an already-cropped region (e.g. a YOLO-
        detected key_signature box). Skips crop_keysig and applies only
        the resize+normalize transform.
    """
    import torch
    from train_keyclf_cnn import (  # type: ignore
        SmallKeyCNN, crop_keysig, _EVAL_TFM, label_to_fifths,
    )
    cnn = SmallKeyCNN().cuda()
    state = torch.load(ckpt_path, map_location="cuda")
    cnn.load_state_dict(state["state_dict"])
    cnn.eval()

    if on_crop:
        def predict(crop_pil) -> int:
            x = _EVAL_TFM(crop_pil.convert("RGB")).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = cnn(x)
            return label_to_fifths(int(logits.argmax(1).item()))
    else:
        def predict(image_pil) -> int:
            x = _EVAL_TFM(crop_keysig(image_pil.convert("RGB"))).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = cnn(x)
            return label_to_fifths(int(logits.argmax(1).item()))
    return predict


def _gt_key(mxl_text: str) -> int:
    """Extract the score's key signature from MusicXML (sign of fifths)."""
    import re
    m = re.search(r"<fifths>(-?\d+)</fifths>", mxl_text)
    return int(m.group(1)) if m else 0


def _hf_index() -> dict:
    """Fetch L7a dev split: {score_id: (PIL image, musicxml)}.

    The transcriber (safckylj) was trained on these HF-stored images, so
    the pipeline must use them at inference. Our re-rendered PNGs in
    `data_dir` are at a different scale and inject all-bar-numbers, which
    is out-of-distribution for the VLM.
    """
    import os
    os.environ.setdefault("HF_HOME", "/tmp/hf_user")
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("zzsi/synthetic-scores", "level7a", split="dev")
    return {row["score_id"]: (row["image"], row["musicxml"]) for row in ds}


def evaluate(data_dir: Path, det_ckpt: str, vlm_ckpt: str,
             n: int, mode: str, imgsz: int,
             cnn_ckpt: str | None = None,
             use_gt_key: bool = False,
             detector_key: bool = False) -> None:
    from eval_mxc import evaluate_pair  # type: ignore
    from respell import respell_mxc2  # type: ignore

    hf = _hf_index()

    labels = data_dir / "labels_dev.jsonl"
    with labels.open() as f:
        recs = [json.loads(line) for line in f][:n]
    if use_gt_key:
        key_src = "GT"
    elif detector_key and cnn_ckpt:
        key_src = "YOLO-keysig+CNN"
    elif cnn_ckpt:
        key_src = "fixed-crop CNN"
    else:
        key_src = "no-respell"
    print(f"Evaluating {len(recs)} dev pages, mode={mode}, key={key_src}",
          flush=True)

    print("Loading detector ...", flush=True)
    det = load_detector(det_ckpt) if (mode == "measure" or detector_key) else None
    print("Loading VLM ...", flush=True)
    vlm_model, processor = load_vlm(vlm_ckpt)
    cnn_predict = None
    cnn_predict_oncrop = None
    if cnn_ckpt and not use_gt_key:
        print(f"Loading key-sig CNN {cnn_ckpt} ...", flush=True)
        if detector_key:
            cnn_predict_oncrop = _load_keysig_cnn(cnn_ckpt, on_crop=True)
        else:
            cnn_predict = _load_keysig_cnn(cnn_ckpt, on_crop=False)

    pitched_base, pitched_resp, rhythm, times = [], [], [], []
    key_correct = key_seen = 0
    for i, rec in enumerate(recs):
        sid = rec["score_id"]
        if sid not in hf:
            print(f"  [{i:2d}] {sid}: not in HF index, skipped", flush=True)
            continue
        hf_img, mxl = hf[sid]
        img = hf_img.convert("RGB")
        t0 = time.time()

        if mode == "page":
            pred = transcribe_page(vlm_model, processor, img)
        else:
            # For detection we use our re-rendered image (det was trained
            # on it); for transcription crops we use the HF image. The
            # bbox coords scale by width ratio.
            our_img_path = data_dir / rec["image"]
            layout = detect_layout(det, str(our_img_path), imgsz=imgsz)
            measures = derive_measures(layout["systems"], layout["staves"],
                                       layout["barlines"])
            # Scale measure bboxes from re-rendered px to HF-image px.
            sx = img.width / rec["width"]
            sy = img.height / rec["height"]
            outs: list[tuple] = []
            for m in measures:
                x, y, w, h = m.bbox
                from cells import Measure as _M
                m_hf = _M(m.system, m.measure,
                          (x * sx, y * sy, w * sx, h * sy))
                txt = transcribe_measure(vlm_model, processor, img, m_hf)
                outs.append((m_hf, txt))
            pred = stitch_measures_naive(outs)

        gt = _load_gt_mxc2(mxl)
        try:
            r_base = evaluate_pair(pred, gt, score_id=sid)
        except Exception as e:
            print(f"  [{i:2d}] {sid}: eval error {e!r}", flush=True)
            continue

        # Stage 3: respell with predicted/oracle key
        if use_gt_key:
            key = _gt_key(mxl)
        elif cnn_predict_oncrop is not None:
            # Detect on the HF image *directly*. The detector trained on
            # our re-rendered PNGs generalises here, and box coordinates
            # are then already in HF-image pixel space -- no scaling.
            # (Linear scaling from re-rendered px would mis-place boxes
            # because the two are *separate renderings*, not the same
            # image at different scales.)
            hf_tmp = Path(f"/tmp/_eval_hf_{sid}.png")
            img.save(hf_tmp)
            try:
                layout = detect_layout(det, str(hf_tmp), imgsz=imgsz)
            finally:
                hf_tmp.unlink(missing_ok=True)
            preds = predict_keys_from_detections(layout, img,
                                                  cnn_predict_oncrop)
            if not preds:
                key = None
            else:
                from collections import Counter
                key = Counter(preds).most_common(1)[0][0]
        elif cnn_predict is not None:
            key = cnn_predict(img)
        else:
            key = None
        true_key = _gt_key(mxl)
        if key is not None:
            key_seen += 1
            if key == true_key:
                key_correct += 1
            try:
                fixed = respell_mxc2(pred, key)
                r_resp = evaluate_pair(fixed, gt, score_id=sid)
            except Exception as e:
                print(f"  [{i:2d}] {sid}: respell err {e!r}", flush=True)
                r_resp = r_base
        else:
            r_resp = r_base

        dt = time.time() - t0
        pitched_base.append(r_base.pitched_only_similarity)
        pitched_resp.append(r_resp.pitched_only_similarity)
        rhythm.append(r_base.rhythm_similarity)
        times.append(dt)
        keytag = (f"key={key:+d}/{true_key:+d}"
                  if key is not None else f"key=GT{true_key:+d}")
        print(f"  [{i:2d}] {sid}  {keytag}  "
              f"pitch {r_base.pitched_only_similarity:.1%}->"
              f"{r_resp.pitched_only_similarity:.1%}  "
              f"rhy {r_base.rhythm_similarity:.1%}  ({dt:.0f}s)",
              flush=True)

    if pitched_base:
        n_ = len(pitched_base)
        mean = lambda xs: sum(xs) / n_
        med = lambda xs: sorted(xs)[n_ // 2]
        print("\n=== Summary ({} pages, mode={}, key-src={}) ==="
              .format(n_, mode, key_src))
        print(f"pitched base     mean={mean(pitched_base):.1%}  "
              f"median={med(pitched_base):.1%}")
        print(f"pitched respell  mean={mean(pitched_resp):.1%}  "
              f"median={med(pitched_resp):.1%}")
        print(f"rhythm           mean={mean(rhythm):.1%}  "
              f"median={med(rhythm):.1%}")
        if key_seen:
            print(f"key accuracy:    {key_correct}/{key_seen} "
                  f"({100*key_correct/key_seen:.1f}%)")
        print(f"avg time/page:   {sum(times)/n_:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="dataset dir from labels/make_dataset.py")
    p.add_argument("--det-ckpt", required=True, help="YOLO checkpoint")
    p.add_argument("--vlm-ckpt", required=True,
                   help="safckylj checkpoint (.../final_model)")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--mode", default="page", choices=["page", "measure"])
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--cnn-ckpt", default=None,
                   help="keysig CNN .pt; if given, respell with predicted key")
    p.add_argument("--use-gt-key", action="store_true",
                   help="respell with GT key from MusicXML (oracle)")
    p.add_argument("--detector-key", action="store_true",
                   help="use YOLO key_signature boxes + CNN per crop "
                        "(majority vote) instead of fixed top-left crop")
    args = p.parse_args()
    evaluate(args.data, args.det_ckpt, args.vlm_ckpt, args.n, args.mode,
             args.imgsz, cnn_ckpt=args.cnn_ckpt, use_gt_key=args.use_gt_key,
             detector_key=args.detector_key)


if __name__ == "__main__":
    main()
