#!/usr/bin/env python3
"""Probe: does the key classifier do better on a cropped+zoomed key-sig region?

Tests whether the off-by-one perception error on 3-vs-4 accidentals is a
*resolution* problem. We crop the top-left of the page (where all three
staves' key signatures appear) and feed it to the existing classifier.

If accuracy on +2/+3/-4 jumps, the issue is resolution → solvable by
training on crops. If it stays bad, the off-by-one is representational
and we need a different approach.

Usage:
    python eval_keyclassifier_crop.py --checkpoint outputs/<keyclf>/final_model \\
        --crop-frac 0.30 --upscale 3
"""
import argparse
import re
import statistics
import time

import torch


KEY_QUESTION = (
    "What is the key signature of this score? Answer in the form "
    "key=N, where N is the number of sharps (positive) or flats "
    "(negative), or key=0 for no sharps or flats."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("-n", "--n-samples", type=int, default=50)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dataset-config", default="level7a")
    ap.add_argument("--crop-frac-w", type=float, default=0.30,
                    help="fraction of image width to keep from left")
    ap.add_argument("--crop-frac-h", type=float, default=0.55,
                    help="fraction of image height to keep from top (covers 3 staves)")
    ap.add_argument("--upscale", type=int, default=3)
    args = ap.parse_args()

    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import prepare_inference_inputs

    print(f"Loading {args.checkpoint} ...")
    model, processor = FastVisionModel.from_pretrained(
        args.checkpoint, load_in_4bit=True
    )
    FastVisionModel.for_inference(model)

    ds = load_dataset("zzsi/synthetic-scores", args.dataset_config, split=args.split)
    n = min(args.n_samples, len(ds))
    print(f"Evaluating {n} samples with crop_w={args.crop_frac_w}, "
          f"crop_h={args.crop_frac_h}, upscale={args.upscale}x.\n")

    rows = []
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        sid = sample.get("score_id", f"{args.split}_{i}")
        m = re.search(r"<fifths>(-?\d+)</fifths>", sample["musicxml"] or "")
        gt_key = int(m.group(1)) if m else None

        img = sample["image"]
        w, h = img.size
        cw, ch = int(w * args.crop_frac_w), int(h * args.crop_frac_h)
        crop = img.crop((0, 0, cw, ch))
        if args.upscale > 1:
            crop = crop.resize((crop.width * args.upscale,
                                crop.height * args.upscale))

        inputs = prepare_inference_inputs(processor, crop, KEY_QUESTION).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, use_cache=True,
                                 do_sample=False)
        pred = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        m2 = re.search(r"key=(-?\d+)", pred)
        pred_key = int(m2.group(1)) if m2 else None
        correct = (gt_key == pred_key)
        rows.append((gt_key, pred_key, correct))
        elapsed = time.time() - t0
        print(f"  [{i:3d}] gt={gt_key:+d}  pred={pred_key}  "
              f"{'✓' if correct else '✗'}  id={sid}  "
              f"({elapsed/(i+1):.1f}s/sample)")

    if not rows:
        return
    n_correct = sum(1 for _, _, c in rows if c)
    print(f"\n===== {args.checkpoint} CROP w={args.crop_frac_w} h={args.crop_frac_h} ×{args.upscale} =====")
    print(f"  overall: {n_correct}/{len(rows)} = {n_correct/len(rows):.1%}")
    by_key = {}
    for gt, pred, c in rows:
        by_key.setdefault(gt, []).append(c)
    print(f"  {'key':>4}  {'n':>3}  {'correct':>8}")
    for k in sorted(by_key):
        v = by_key[k]
        acc = sum(v) / len(v)
        print(f"  {k:>+4d}  {len(v):>3}  {acc:>7.1%}")


if __name__ == "__main__":
    main()
