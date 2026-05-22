#!/usr/bin/env python3
"""Evaluate L7a transcription with the ground-truth key injected into the prompt.

Tests the key-as-context decomposition premise. Run with --inject to hand the
model the key signature; run without it for the baseline. Reports overall and
per-key pitched-note accuracy.

Usage:
    python eval_keyinprompt.py --checkpoint outputs/safckylj/final_model --inject
    python eval_keyinprompt.py --checkpoint outputs/safckylj/final_model
    python eval_keyinprompt.py --checkpoint outputs/<keyinprompt_run>/final_model --inject
"""

import argparse
import re
import statistics
import time

import torch


def gt_fifths(musicxml):
    m = re.search(r"<fifths>(-?\d+)</fifths>", musicxml or "")
    return int(m.group(1)) if m else None


def run(checkpoint, n_samples, split, inject, dataset_config="level7a"):
    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import prepare_inference_inputs, INSTRUCTION_MXC2, strip_musicxml_header
    from eval_mxc import evaluate_pair
    from mxc2 import xml_to_mxc2

    print(f"Loading model from {checkpoint} ...")
    model, processor = FastVisionModel.from_pretrained(checkpoint, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    ds = load_dataset("zzsi/synthetic-scores", dataset_config, split=split)
    n_samples = min(n_samples, len(ds))
    print(f"Evaluating {n_samples}/{len(ds)} samples from {dataset_config}/{split} "
          f"(inject_gt_key={inject})\n")

    rows = []  # (gt_key, pitched_similarity, correct_key)
    t0 = time.time()
    for i in range(n_samples):
        sample = ds[i]
        key = gt_fifths(sample["musicxml"])
        instruction = INSTRUCTION_MXC2
        if inject and key is not None:
            instruction = (f"{instruction} The key signature is key={key} "
                           f"(N = sharps if positive, flats if negative).")

        inputs = prepare_inference_inputs(processor, sample["image"], instruction).to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=8192,
                                    use_cache=True, do_sample=False)
        pred = processor.decode(output[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

        ref = strip_musicxml_header(sample["musicxml"])
        try:
            ref = xml_to_mxc2(ref, drop_beams=True)
        except Exception:
            pass

        sid = sample.get("score_id", f"{split}_{i}")
        try:
            r = evaluate_pair(pred, ref, score_id=sid)
            rows.append((key, r.pitched_only_similarity,
                         r.position_only_similarity, r.correct_key))
            print(f"  [{i:3d}] key={key:+d}  pitch={r.pitched_only_similarity:.1%}  "
                  f"pos={r.position_only_similarity:.1%}  "
                  f"key_ok={r.correct_key}  id={sid}  "
                  f"({(time.time()-t0)/(i+1):.1f}s/sample)")
        except Exception as e:
            print(f"  [{i:3d}] Error: {e}")

    return rows


def summarize(rows, label):
    pitch = [p for _, p, _, _ in rows]
    pos = [q for _, _, q, _ in rows]
    print(f"\n===== {label} =====")
    print(f"  n={len(rows)}  pitched mean={statistics.mean(pitch):.1%} "
          f"median={statistics.median(pitch):.1%}  |  "
          f"position mean={statistics.mean(pos):.1%} "
          f"median={statistics.median(pos):.1%}")
    by_key = {}
    for k, p, q, ok in rows:
        by_key.setdefault(k, []).append((p, q, ok))
    print(f"  {'key':>4}  {'n':>3}  {'pitch_mean':>10}  {'pos_mean':>9}  {'key_acc':>7}")
    for k in sorted(by_key):
        v = by_key[k]
        kacc = sum(1 for _, _, ok in v if ok) / len(v)
        print(f"  {k:>+4d}  {len(v):>3}  {statistics.mean(p for p,_,_ in v):>9.1%}  "
              f"{statistics.mean(q for _,q,_ in v):>8.1%}  {kacc:>6.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("-n", "--n-samples", type=int, default=50)
    p.add_argument("--split", default="dev")
    p.add_argument("--dataset-config", default="level7a")
    p.add_argument("--inject", action="store_true",
                   help="Inject ground-truth key=N into the prompt")
    args = p.parse_args()

    rows = run(args.checkpoint, args.n_samples, args.split, args.inject,
               args.dataset_config)
    summarize(rows, f"checkpoint={args.checkpoint} inject={args.inject}")


if __name__ == "__main__":
    main()
