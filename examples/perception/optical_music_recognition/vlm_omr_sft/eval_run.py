#!/usr/bin/env python3
"""Batch evaluation of a trained OMR model on a test split.

Loads a checkpoint, runs inference on N samples, and prints accuracy metrics.

Usage:
    python eval_run.py --checkpoint outputs/checkpoint-2979
    python eval_run.py --checkpoint outputs/final_model --n 100 --dataset-config level3
    python eval_run.py --checkpoint outputs/checkpoint-2979 --split dev
"""

import argparse
import time
import sys

import torch


def run_eval(checkpoint, n_samples=50, dataset_repo="zzsi/synthetic-scores",
             dataset_config="level7", split="test", target_format="mxc",
             drop_beams=False):
    """Run inference + accuracy evaluation. Returns list of EvalResult."""
    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import (prepare_inference_inputs, INSTRUCTION_MXC, INSTRUCTION_MXC2,
                       INSTRUCTION_XML, strip_musicxml_header)
    from eval_mxc import evaluate_pair
    from mxc import xml_to_mxc
    from mxc2 import xml_to_mxc2

    instruction = (INSTRUCTION_MXC2 if target_format == "mxc2"
                   else INSTRUCTION_MXC if target_format == "mxc"
                   else INSTRUCTION_XML)

    print(f"Loading model from {checkpoint} ...")
    model, processor = FastVisionModel.from_pretrained(checkpoint, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    print(f"Loading dataset: {dataset_repo} config={dataset_config} split={split} ...")
    print(f"Target format: {target_format} (drop_beams={drop_beams})")
    ds = load_dataset(dataset_repo, dataset_config, split=split)
    n_samples = min(n_samples, len(ds))
    print(f"Evaluating {n_samples} / {len(ds)} samples\n")

    results = []
    t0 = time.time()
    for i in range(n_samples):
        sample = ds[i]
        inputs = prepare_inference_inputs(processor, sample["image"], instruction).to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=8192, use_cache=True, do_sample=False)
        pred_tokens = output[0][inputs["input_ids"].shape[1]:]
        pred = processor.decode(pred_tokens, skip_special_tokens=True).strip()

        ref = strip_musicxml_header(sample["musicxml"])
        try:
            if target_format == "mxc2":
                ref = xml_to_mxc2(ref, drop_beams=drop_beams)
            elif target_format == "mxc":
                ref = xml_to_mxc(ref)
        except Exception:
            pass

        sid = sample.get("score_id", f"{split}_{i}")
        try:
            r = evaluate_pair(pred, ref, score_id=sid)
            results.append(r)
            elapsed = time.time() - t0
            per_sample = elapsed / (i + 1)
            print(f"  [{i:3d}] pitch={r.pitched_only_similarity:.1%}  "
                  f"rhythm={r.rhythm_similarity:.1%}  "
                  f"id={sid}  ({per_sample:.1f}s/sample)")
        except Exception as e:
            print(f"  [{i:3d}] Error: {e}")

    elapsed = time.time() - t0
    if results:
        print(f"\nTotal: {elapsed:.0f}s for {len(results)} samples ({elapsed / len(results):.1f}s/sample)")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="outputs/checkpoint-2979",
                        help="Path to model checkpoint")
    parser.add_argument("-n", "--n-samples", type=int, default=50,
                        help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--dataset-repo", default="zzsi/synthetic-scores")
    parser.add_argument("--dataset-config", default="level7")
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-format", default="mxc",
                        choices=["xml", "mxc", "mxc2"])
    parser.add_argument("--drop-beams", action="store_true")
    args = parser.parse_args()

    from eval_mxc import format_summary

    results = run_eval(
        checkpoint=args.checkpoint,
        n_samples=args.n_samples,
        dataset_repo=args.dataset_repo,
        dataset_config=args.dataset_config,
        target_format=args.target_format,
        drop_beams=args.drop_beams,
        split=args.split,
    )
    print()
    print(format_summary(results))


if __name__ == "__main__":
    main()
