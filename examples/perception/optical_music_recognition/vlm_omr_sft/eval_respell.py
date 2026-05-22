#!/usr/bin/env python3
"""End-to-end eval of: page transcription → respell with trusted key.

For Stage 1 we use GROUND-TRUTH KEY from the dataset (perfect Stage-1).
This validates the pipeline ceiling — if the GT-key + respell run gives a
strong pitched accuracy, then the pipeline is viable given a reliable key
classifier. The remaining gap will then be in Stage 1's per-key accuracy.

Usage:
    python eval_respell.py --checkpoint outputs/safckylj/final_model \\
        --split dev -n 50
"""

import argparse
import re
import statistics
import time

import torch


def gt_fifths(musicxml):
    m = re.search(r"<fifths>(-?\d+)</fifths>", musicxml or "")
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Stage-2 transcription model")
    ap.add_argument("--key-checkpoint", default=None,
                    help="Stage-1 VLM key classifier. If absent and "
                         "--cnn-checkpoint also absent, uses GT key.")
    ap.add_argument("--cnn-checkpoint", default=None,
                    help="Stage-1 CNN key classifier (e.g. best.pt).")
    ap.add_argument("-n", "--n-samples", type=int, default=50)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dataset-config", default="level7a")
    args = ap.parse_args()

    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import prepare_inference_inputs, INSTRUCTION_MXC2, strip_musicxml_header
    from eval_mxc import evaluate_pair
    from mxc2 import xml_to_mxc2
    from respell import respell_mxc2

    print(f"Loading transcription {args.checkpoint} ...")
    model, processor = FastVisionModel.from_pretrained(args.checkpoint, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    key_model = key_proc = None
    if args.key_checkpoint:
        print(f"Loading VLM key classifier {args.key_checkpoint} ...")
        key_model, key_proc = FastVisionModel.from_pretrained(
            args.key_checkpoint, load_in_4bit=True
        )
        FastVisionModel.for_inference(key_model)

    cnn_predict_key = None
    if args.cnn_checkpoint:
        print(f"Loading CNN key classifier {args.cnn_checkpoint} ...")
        from train_keyclf_cnn import (
            SmallKeyCNN, crop_keysig, _EVAL_TFM, label_to_fifths,
        )
        cnn = SmallKeyCNN().cuda()
        ckpt = torch.load(args.cnn_checkpoint, map_location="cuda")
        cnn.load_state_dict(ckpt["state_dict"])
        cnn.eval()

        def _cnn_predict(img):
            x = _EVAL_TFM(crop_keysig(img.convert("RGB"))).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = cnn(x)
            return label_to_fifths(int(logits.argmax(1).item()))
        cnn_predict_key = _cnn_predict

    KEY_QUESTION = (
        "What is the key signature of this score? Answer in the form "
        "key=N, where N is the number of sharps (positive) or flats "
        "(negative), or key=0 for no sharps or flats."
    )

    ds = load_dataset("zzsi/synthetic-scores", args.dataset_config, split=args.split)
    n = min(args.n_samples, len(ds))
    if args.cnn_checkpoint:
        src = "CNN classifier"
    elif args.key_checkpoint:
        src = "VLM classifier"
    else:
        src = "GT key (oracle)"
    print(f"Evaluating {n} samples; Stage-1 = {src}.\n")

    rows = []  # (gt_key, pitch_base, pos_base, pitch_respell, pos_respell)
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        sid = sample.get("score_id", f"{args.split}_{i}")
        gt_key = gt_fifths(sample["musicxml"])

        # Stage 1: get the trusted key
        if cnn_predict_key is not None:
            key = cnn_predict_key(sample["image"])
        elif key_model is not None:
            kin = prepare_inference_inputs(
                key_proc, sample["image"], KEY_QUESTION
            ).to("cuda")
            with torch.no_grad():
                kout = key_model.generate(
                    **kin, max_new_tokens=64, use_cache=True, do_sample=False
                )
            kpred = key_proc.decode(
                kout[0][kin["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            mk = re.search(r"key=(-?\d+)", kpred)
            key = int(mk.group(1)) if mk else gt_key
        else:
            key = gt_key

        # Stage 2: transcription
        inputs = prepare_inference_inputs(
            processor, sample["image"], INSTRUCTION_MXC2
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8192,
                                 use_cache=True, do_sample=False)
        pred = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        ref = strip_musicxml_header(sample["musicxml"])
        try:
            ref = xml_to_mxc2(ref, drop_beams=True)
        except Exception:
            pass

        try:
            r_base = evaluate_pair(pred, ref, score_id=sid)
        except Exception as e:
            print(f"  [{i:3d}] base eval error: {e}")
            continue
        try:
            fixed = respell_mxc2(pred, key) if key is not None else pred
            r_fixed = evaluate_pair(fixed, ref, score_id=sid)
        except Exception as e:
            print(f"  [{i:3d}] respell eval error: {e}")
            continue

        rows.append((gt_key,
                     r_base.pitched_only_similarity,
                     r_base.position_only_similarity,
                     r_fixed.pitched_only_similarity,
                     r_fixed.position_only_similarity))
        elapsed = time.time() - t0
        key_tag = (f"pred_key={key:+d}{'' if key == gt_key else f'(gt {gt_key:+d})'}"
                   if key_model is not None else f"key={gt_key:+d}")
        print(f"  [{i:3d}] {key_tag}  "
              f"pitched: {r_base.pitched_only_similarity:.1%} → "
              f"{r_fixed.pitched_only_similarity:.1%}  "
              f"pos: {r_base.position_only_similarity:.1%}  "
              f"({elapsed/(i+1):.0f}s/sample)")

    if not rows:
        return
    print(f"\n===== {args.checkpoint} on {args.dataset_config}/{args.split} =====")
    pb = [r[1] for r in rows]
    pa = [r[3] for r in rows]
    posb = [r[2] for r in rows]
    print(f"  n={len(rows)}")
    print(f"  pitched   base mean={statistics.mean(pb):.1%}  median={statistics.median(pb):.1%}")
    print(f"  pitched respell mean={statistics.mean(pa):.1%}  median={statistics.median(pa):.1%}")
    print(f"  position       mean={statistics.mean(posb):.1%}  median={statistics.median(posb):.1%}")

    # Per-key
    by_key = {}
    for k, pb_, posb_, pa_, posa_ in rows:
        by_key.setdefault(k, []).append((pb_, pa_, posb_))
    print(f"\n  {'key':>4}  {'n':>3}  {'pitch_base':>10}  {'pitch_respell':>13}  {'gain':>5}  {'pos':>5}")
    for k in sorted(by_key):
        v = by_key[k]
        mb = statistics.mean(p for p, _, _ in v)
        ma = statistics.mean(p for _, p, _ in v)
        mp = statistics.mean(p for _, _, p in v)
        gain = ma - mb
        print(f"  {k:>+4d}  {len(v):>3}  {mb:>9.1%}  {ma:>12.1%}  {gain:>+5.1%}  {mp:>4.1%}")


if __name__ == "__main__":
    main()
