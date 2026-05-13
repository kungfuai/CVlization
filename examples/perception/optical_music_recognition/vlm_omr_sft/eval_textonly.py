#!/usr/bin/env python3
"""Text-only eval: feed Audiveris MXC2 as input, evaluate corrected output.

No image is shown to the model. Counterpart to eval_with_hints.py.

Usage:
    python eval_textonly.py --checkpoint outputs/1nh8jmjn/final_model \
        --hints audiveris_hints_level9_n50.jsonl -n 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


INSTRUCTION_TEXTONLY = (
    "A classical OMR system (Audiveris) produced this draft transcription of "
    "a sheet music page. It may have errors — especially in voice assignment, "
    "chord grouping, and pitch ordering. Output the corrected MXC2 "
    "transcription.\n\n"
    "Audiveris draft:\n{hint}\n\n"
    "Corrected MXC2 transcription:"
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hints", required=True)
    p.add_argument("-n", "--n-samples", type=int, default=None)
    p.add_argument("--dataset-repo", default="zzsi/synthetic-scores")
    p.add_argument("--dataset-config", default="level9")
    p.add_argument("--split", default="test")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-hint-chars", type=int, default=3500)
    p.add_argument("--output", default="eval_textonly_results.jsonl")
    args = p.parse_args()

    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import strip_musicxml_header
    from eval_mxc import evaluate_pair
    from mxc2 import xml_to_mxc2
    from hint_compress import compress_hint

    # Load hints
    hints = {}
    with open(args.hints) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("audiveris_failed") and r.get("audiveris_mxc2"):
                hints[r["score_id"]] = compress_hint(r["audiveris_mxc2"])
    print(f"Loaded {len(hints)} hints from {args.hints}")

    print(f"Loading model from {args.checkpoint}")
    model, processor = FastVisionModel.from_pretrained(args.checkpoint, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    ds = load_dataset(args.dataset_repo, args.dataset_config, split=args.split)
    n = min(args.n_samples or len(ds), len(ds))
    print(f"Evaluating {n} samples\n")

    results = []
    no_hint = 0
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i in range(n):
            sample = ds[i]
            sample_id = sample.get("score_id", f"sample_{i:04d}")
            hint = hints.get(sample_id, "")
            has_hint = bool(hint)

            if not has_hint:
                no_hint += 1
                hint_text = "[transcription unavailable]"
            else:
                hint_text = hint[:args.max_hint_chars]

            prompt = INSTRUCTION_TEXTONLY.format(hint=hint_text)

            # Text-only chat template (no image content)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")

            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        do_sample=False,
                    )
                pred = processor.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                ref_xml = strip_musicxml_header(sample["musicxml"])
                ref_mxc2 = xml_to_mxc2(ref_xml, drop_beams=True)
                r = evaluate_pair(pred, ref_mxc2, score_id=sample_id)

                rec = {
                    "i": i,
                    "score_id": sample_id,
                    "has_hint": has_hint,
                    "pitched_only_similarity": r.pitched_only_similarity,
                    "rhythm_similarity": r.rhythm_similarity,
                    "combined_similarity": r.combined_similarity,
                    "note_coverage": r.note_coverage,
                    "pred_notes": r.pred_notes,
                    "ref_notes": r.ref_notes,
                }
                results.append(rec)
            except Exception as e:
                rec = {"i": i, "score_id": sample_id, "error": str(e)[:200]}

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            elapsed = time.time() - t0
            sim_str = f"{rec.get('pitched_only_similarity', 0):.1%}" if "error" not in rec else "FAIL"
            hint_flag = "hint" if has_hint else "no-hint"
            print(f"  [{i:3d}/{n}] {sample_id} [{hint_flag}]: {sim_str}  "
                  f"({elapsed/(i+1):.0f}s/sample)")

    # Aggregate
    n_ok = len(results)
    print(f"\n{'='*60}")
    print(f"TEXT-ONLY eval: {args.dataset_config}/{args.split} n={n}")
    print(f"{'='*60}")
    print(f"  Samples with hint: {n - no_hint}/{n}")
    if n_ok:
        for key in ("pitched_only_similarity", "rhythm_similarity",
                    "combined_similarity", "note_coverage"):
            vals = [r[key] for r in results]
            print(f"  {key:30s}: {sum(vals)/n_ok:.1%}  (max {max(vals):.1%})")
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
