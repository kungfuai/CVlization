#!/usr/bin/env python3
"""Evaluate the Stage-1 key classifier: image → key=N.

For each sample, asks KEY_QUESTION, parses `key=N` from output, compares
to GT `<fifths>`. Reports per-key accuracy across all 9 keys (-4..+4).

Usage:
    python eval_keyclassifier.py --checkpoint outputs/<keyclf>/final_model
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
    print(f"Evaluating {n} samples on key-only task.\n")

    rows = []
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        sid = sample.get("score_id", f"{args.split}_{i}")
        m = re.search(r"<fifths>(-?\d+)</fifths>", sample["musicxml"] or "")
        gt_key = int(m.group(1)) if m else None
        inputs = prepare_inference_inputs(
            processor, sample["image"], KEY_QUESTION
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=32, use_cache=True, do_sample=False
            )
        pred = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
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
    print(f"\n===== {args.checkpoint} on {args.dataset_config}/{args.split} =====")
    n_correct = sum(1 for _, _, c in rows if c)
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
