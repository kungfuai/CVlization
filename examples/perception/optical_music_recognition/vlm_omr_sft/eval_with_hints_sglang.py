#!/usr/bin/env python3
"""Hint-augmented eval using SGLang for faster inference.

Mirrors eval_with_hints.py but uses SGLang Engine instead of HF generate().
Expected speedup: 3-5× per sample, more with batching.

Requires a MERGED model (LoRA adapter baked into base weights). After
hint-SFT training, run merge_adapter.py first, then point this script
at the merged directory.

Usage:
    python eval_with_hints_sglang.py \
        --model outputs/hint_sft_merged \
        --hints audiveris_hints_level9_n50.jsonl \
        -n 50
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

INSTRUCTION_WITH_HINT = (
    "Transcribe this sheet music page to MXC2 (compact MusicXML). "
    "A classical OMR system (Audiveris) provides this draft transcription. "
    "It may have errors — especially in voice assignment, chord grouping, and "
    "pitch ordering — but it likely has correct measure boundaries, key/time "
    "signatures, clefs, and many correct pitch values. Use it as a hint and "
    "verify against the image.\n\n"
    "Audiveris draft:\n{hint}\n\n"
    "Corrected MXC2 transcription:"
)

INSTRUCTION_NO_HINT = "Transcribe this sheet music page to MXC2 (compact MusicXML)."


def encode_image(img: Image.Image) -> str:
    """Base64-encode a PIL image for SGLang's image_data field."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Path to merged model (LoRA baked into base weights)")
    p.add_argument("--hints", required=True,
                   help="JSONL from audiveris/extract_hints.py")
    p.add_argument("-n", "--n-samples", type=int, default=None)
    p.add_argument("--dataset-repo", default="zzsi/synthetic-scores")
    p.add_argument("--dataset-config", default="level9")
    p.add_argument("--split", default="test")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-hint-chars", type=int, default=3500)
    p.add_argument("--mem-fraction-static", type=float, default=0.5)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--output", default="eval_with_hints_sglang.jsonl")
    p.add_argument("--compress-hints", action="store_true", default=True,
                   help="Apply same compression as training (drop noise from Audiveris)")
    args = p.parse_args()

    from datasets import load_dataset
    from eval_mxc import evaluate_pair
    from mxc2 import xml_to_mxc2
    from train import strip_musicxml_header

    if args.compress_hints:
        from hint_compress import compress_hint
    else:
        compress_hint = lambda x: x  # noqa: E731

    # Load hints
    hints = {}
    with open(args.hints) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("audiveris_failed") and r.get("audiveris_mxc2"):
                hints[r["score_id"]] = compress_hint(r["audiveris_mxc2"])
    print(f"Loaded {len(hints)} hints from {args.hints}")

    # Load dataset
    ds = load_dataset(args.dataset_repo, args.dataset_config, split=args.split)
    n = min(args.n_samples or len(ds), len(ds))
    print(f"Evaluating {n} samples\n")

    # Load SGLang engine
    print(f"Loading SGLang engine: {args.model}")
    t0 = time.time()
    import sglang as sgl
    engine = sgl.Engine(
        model_path=args.model,
        mem_fraction_static=args.mem_fraction_static,
        tp_size=args.tp_size,
        log_level="warning",
    )
    print(f"  Engine loaded in {time.time() - t0:.1f}s\n")

    # Get the processor to use the chat template (matches how the model was trained)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    results = []
    t_start = time.time()
    with open(args.output, "w") as fout:
        for i in range(n):
            sample = ds[i]
            sample_id = sample.get("score_id", f"sample_{i:04d}")
            hint = hints.get(sample_id, "")
            has_hint = bool(hint)

            if has_hint:
                instruction = INSTRUCTION_WITH_HINT.format(hint=hint[:args.max_hint_chars])
            else:
                instruction = INSTRUCTION_NO_HINT

            # Build chat template input
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_data = encode_image(sample["image"])

            t1 = time.time()
            try:
                out = engine.generate(
                    prompt=prompt,
                    image_data=image_data,
                    sampling_params={
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": 0.0,
                    },
                )
                pred = out.get("text", "")

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
                    "gen_time_s": time.time() - t1,
                }
                results.append(rec)
            except Exception as e:
                rec = {"i": i, "score_id": sample_id, "error": str(e)[:300]}

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            elapsed = time.time() - t_start
            sim_str = f"{rec.get('pitched_only_similarity', 0):.1%}" if "error" not in rec else "FAIL"
            tag = "hint" if has_hint else "no-hint"
            print(f"  [{i:3d}/{n}] {sample_id} [{tag}]: {sim_str}  "
                  f"({elapsed/(i+1):.1f}s/sample)")

    engine.shutdown()

    # Aggregate
    n_ok = len(results)
    print(f"\n{'='*60}")
    print(f"SGLang hint-eval: {args.dataset_config}/{args.split} n={n}")
    print(f"{'='*60}")
    if n_ok:
        for key in ("pitched_only_similarity", "rhythm_similarity",
                    "combined_similarity", "note_coverage"):
            vals = [r[key] for r in results]
            print(f"  {key:30s}: {sum(vals)/n_ok:.1%}  (max {max(vals):.1%})")
        hinted = [r for r in results if r.get("has_hint")]
        unhinted = [r for r in results if not r.get("has_hint")]
        if hinted:
            avg = sum(r["pitched_only_similarity"] for r in hinted) / len(hinted)
            print(f"\n  Hinted (n={len(hinted)}): pitched_only = {avg:.1%}")
        if unhinted:
            avg = sum(r["pitched_only_similarity"] for r in unhinted) / len(unhinted)
            print(f"  Unhinted (n={len(unhinted)}): pitched_only = {avg:.1%}")
    print(f"\nTotal time: {time.time()-t_start:.0f}s  "
          f"({(time.time()-t_start)/n:.1f}s/sample)")


if __name__ == "__main__":
    main()
