#!/usr/bin/env python3
"""In-context VLM evaluation with Audiveris hints.

For each test sample, prepend the Audiveris MXC2 transcription to the prompt
as a "hint" and ask the model to provide the corrected MXC2 transcription.

Tests whether Audiveris's pitch information (~88% correct values, but wrong
order due to voice/staff confusion) can be exploited as in-context hints
WITHOUT retraining the SFT model.

Usage:
    python eval_with_hints.py --checkpoint outputs/tamqjf4k/final_model \
        --hints audiveris_hints_level9_n50.jsonl -n 50

The hints JSONL is produced by audiveris/extract_hints.py.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

INSTRUCTION_WITH_HINT = (
    "Transcribe this sheet music page to MXC2 (compact MusicXML). "
    "A classical OMR system (Audiveris) provides the following draft transcription. "
    "It may have errors — especially in voice assignment, chord grouping, and "
    "pitch ordering — but it likely has correct measure boundaries, key/time "
    "signatures, clefs, and many correct pitch values. Use it as a hint and "
    "verify against the image.\n\n"
    "Audiveris draft:\n{hint}\n\n"
    "Corrected MXC2 transcription:"
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hints", required=True, help="JSONL from audiveris/extract_hints.py")
    p.add_argument("-n", "--n-samples", type=int, default=None)
    p.add_argument("--dataset-repo", default="zzsi/synthetic-scores")
    p.add_argument("--dataset-config", default="level9")
    p.add_argument("--split", default="test")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-hint-chars", type=int, default=8000,
                   help="Truncate Audiveris hint to this many chars to stay in context")
    p.add_argument("--output", default="eval_with_hints_results.jsonl")
    args = p.parse_args()

    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import prepare_inference_inputs, strip_musicxml_header
    from eval_mxc import evaluate_pair
    from mxc2 import xml_to_mxc2

    # Load Audiveris hints (indexed by score_id)
    hints = {}
    with open(args.hints) as f:
        for line in f:
            r = json.loads(line)
            hints[r["score_id"]] = r
    print(f"Loaded {len(hints)} hints from {args.hints}")

    print(f"Loading model from {args.checkpoint}")
    model, processor = FastVisionModel.from_pretrained(args.checkpoint, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    ds = load_dataset(args.dataset_repo, args.dataset_config, split=args.split)
    n = min(args.n_samples or len(ds), len(ds))
    print(f"Evaluating {n} samples\n")

    results = []
    no_hint, with_hint_fail = 0, 0
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i in range(n):
            sample = ds[i]
            sample_id = sample.get("score_id", f"sample_{i:04d}")
            hint_record = hints.get(sample_id, {})
            audiveris_mxc2 = hint_record.get("audiveris_mxc2", "")
            audiveris_failed = hint_record.get("audiveris_failed", True) or not audiveris_mxc2

            if audiveris_failed:
                no_hint += 1
                hint_text = "[transcription unavailable]"
            else:
                hint_text = audiveris_mxc2[:args.max_hint_chars]

            instruction = INSTRUCTION_WITH_HINT.format(hint=hint_text)
            inputs = prepare_inference_inputs(processor, sample["image"], instruction).to("cuda")

            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        do_sample=False,
                    )
                pred = processor.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                ref_xml = strip_musicxml_header(sample["musicxml"])
                ref_mxc2 = xml_to_mxc2(ref_xml, drop_beams=True)
                r = evaluate_pair(pred, ref_mxc2, score_id=sample_id)
                rec = {
                    "i": i,
                    "score_id": sample_id,
                    "audiveris_failed": audiveris_failed,
                    "pitched_only_similarity": r.pitched_only_similarity,
                    "rhythm_similarity": r.rhythm_similarity,
                    "combined_similarity": r.combined_similarity,
                    "note_coverage": r.note_coverage,
                    "pred_notes": r.pred_notes,
                    "ref_notes": r.ref_notes,
                }
                results.append(rec)
            except Exception as e:
                with_hint_fail += 1
                rec = {"i": i, "score_id": sample_id, "error": str(e)[:200]}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

            sim_str = f"{rec.get('pitched_only_similarity', 0):.1%}" if "error" not in rec else "FAIL"
            hint_flag = "no-hint" if audiveris_failed else "hinted"
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{n}] {sample_id} [{hint_flag}]: {sim_str}  "
                  f"({elapsed/(i+1):.0f}s/sample)")

    # Aggregate
    n_ok = len(results)
    print(f"\n{'='*60}")
    print(f"VLM + Audiveris hints, {args.dataset_config}/{args.split}, n={n}")
    print(f"{'='*60}")
    print(f"  Samples with Audiveris hint: {n - no_hint}/{n} ({(n-no_hint)/n:.0%})")
    print(f"  VLM eval failures: {with_hint_fail}")
    if n_ok:
        for key in ("pitched_only_similarity", "rhythm_similarity",
                    "combined_similarity", "note_coverage"):
            vals = [r[key] for r in results]
            print(f"  {key:30s}: {sum(vals)/n_ok:.1%}  (max {max(vals):.1%})")

        # Compare hinted vs no-hint
        hinted = [r for r in results if not r.get("audiveris_failed")]
        unhinted = [r for r in results if r.get("audiveris_failed")]
        if hinted:
            avg_h = sum(r["pitched_only_similarity"] for r in hinted) / len(hinted)
            print(f"\n  Hinted samples (n={len(hinted)}): pitched_only avg = {avg_h:.1%}")
        if unhinted:
            avg_u = sum(r["pitched_only_similarity"] for r in unhinted) / len(unhinted)
            print(f"  Unhinted samples (n={len(unhinted)}): pitched_only avg = {avg_u:.1%}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Per-sample saved to {args.output}")


if __name__ == "__main__":
    main()
