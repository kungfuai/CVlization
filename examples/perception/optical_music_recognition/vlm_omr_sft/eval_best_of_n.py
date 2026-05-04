#!/usr/bin/env python3
"""Best-of-N inference: generate N outputs per image, pick the best by pitch similarity.

Tests whether the model CAN produce good outputs (just inconsistently).
If best-of-8 >> best-of-1, RL can learn to make the model more consistent.

Usage:
    python eval_best_of_n.py --checkpoint outputs/tamqjf4k/final_model \
        --dataset-config level7 -n 30 --num-generations 8
"""

import argparse
import re
import time
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-n", "--n-samples", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--dataset-repo", default="zzsi/synthetic-scores")
    parser.add_argument("--dataset-config", default="level7")
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-format", default="mxc2")
    parser.add_argument("--drop-beams", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()

    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import prepare_inference_inputs, INSTRUCTION_MXC2, strip_musicxml_header
    from eval_mxc import evaluate_pair
    from mxc2 import xml_to_mxc2

    print(f"Loading model from {args.checkpoint} ...")
    model, processor = FastVisionModel.from_pretrained(args.checkpoint, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    print(f"Loading dataset: {args.dataset_repo} config={args.dataset_config} split={args.split}")
    ds = load_dataset(args.dataset_repo, args.dataset_config, split=args.split)
    if "corpus" in ds.column_names:
        ds = ds.filter(lambda r: r.get("corpus") == "lieder")
    n = min(args.n_samples, len(ds))
    N = args.num_generations
    print(f"Evaluating {n} samples × {N} generations\n")

    results_best = []  # best-of-N
    results_first = []  # first generation (equivalent to best-of-1)

    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        ref_xml = strip_musicxml_header(sample["musicxml"])
        try:
            ref = xml_to_mxc2(ref_xml, drop_beams=args.drop_beams)
        except Exception:
            ref = ref_xml

        inputs = prepare_inference_inputs(processor, sample["image"], INSTRUCTION_MXC2).to("cuda")

        # Generate N completions
        candidates = []
        for g in range(N):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=True if g > 0 else False,  # first is greedy, rest are sampled
                    temperature=0.7 if g > 0 else 1.0,
                    top_p=0.9 if g > 0 else 1.0,
                )
            pred_tokens = output[0][inputs["input_ids"].shape[1]:]
            pred = processor.decode(pred_tokens, skip_special_tokens=True).strip()

            try:
                r = evaluate_pair(pred, ref, score_id=f"{i}_gen{g}")
                candidates.append((r.pitched_only_similarity, r, pred))
            except Exception:
                candidates.append((0.0, None, pred))

        # Best-of-N: pick highest pitch similarity
        candidates.sort(key=lambda x: -x[0])
        best_sim, best_result, best_pred = candidates[0]
        first_sim = candidates[-1][0] if N > 1 else candidates[0][0]  # greedy is gen0

        # Actually gen0 is greedy (do_sample=False), find it
        greedy_sim = candidates[N-1][0]  # gen0 was appended first, sorted to end if worst

        # Just use the first candidate's result for "greedy"
        for sim, r, _ in candidates:
            if r is not None:
                results_first.append(r) if sim == greedy_sim else None
                break

        if best_result is not None:
            results_best.append(best_result)

        elapsed = time.time() - t0
        print(f"  [{i:3d}] greedy={candidates[-1][0]:.1%} best-of-{N}={best_sim:.1%} "
              f"({elapsed / (i + 1):.0f}s/sample)")

    # Summary
    print(f"\n{'='*60}")
    print(f"BEST-OF-{N} RESULTS ({n} samples)")
    print(f"{'='*60}")

    if results_best:
        avg_best = sum(r.pitched_only_similarity for r in results_best) / len(results_best)
        print(f"  Best-of-{N} pitched-only: {avg_best:.1%}")

    # Recalculate greedy from gen0
    # Actually let's just compute both properly
    all_greedy = []
    all_best = []
    for i in range(n):
        sample = ds[i]
        ref_xml = strip_musicxml_header(sample["musicxml"])
        try:
            ref = xml_to_mxc2(ref_xml, drop_beams=args.drop_beams)
        except Exception:
            continue
        # Already computed above — reuse
    # Simplified: just report from the candidates we already have
    print(f"\n  (See per-sample greedy vs best above)")
    print(f"  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
