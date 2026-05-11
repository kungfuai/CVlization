#!/usr/bin/env python3
"""Evaluate Audiveris on the zzsi/synthetic-scores dataset.

For each sample:
  1. Save the image to disk
  2. Run Audiveris in batch mode → MusicXML
  3. Read predicted MusicXML
  4. Compare to reference MusicXML via eval_mxc.evaluate_pair (auto-converts XML → MXC)
  5. Aggregate metrics

Usage:
    python eval_synthetic_scores.py -n 30                       # default: level9 test
    python eval_synthetic_scores.py -n 50 --split test
    python eval_synthetic_scores.py --dataset-config level7a -n 20

Outputs JSONL per-sample + a final aggregate to stdout.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

# Make eval_mxc importable from the SFT codebase (mounted at /cvlization_repo in container)
SFT_DIR = "/cvlization_repo/examples/perception/optical_music_recognition/vlm_omr_sft"
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from predict import find_audiveris, run_audiveris


def read_mxl(mxl_path: Path) -> str:
    """Audiveris outputs compressed .mxl; extract the inner MusicXML."""
    with zipfile.ZipFile(mxl_path) as z:
        # MXL container has META-INF/container.xml pointing to root score file
        names = z.namelist()
        # Find the score file — usually <basename>.xml, not META-INF
        score = next((n for n in names if n.endswith(".xml") and "META-INF" not in n), None)
        if not score:
            raise ValueError(f"No score XML in {mxl_path}: {names}")
        return z.read(score).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n-samples", type=int, default=30)
    parser.add_argument("--dataset-repo", default="zzsi/synthetic-scores")
    parser.add_argument("--dataset-config", default="level9")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="audiveris_level9_results.jsonl")
    parser.add_argument("--workdir", default=None,
                        help="Directory for intermediate files (default: tempdir)")
    parser.add_argument("--upscale", type=float, default=2.5,
                        help="Image upscale factor before Audiveris "
                             "(synthetic-scores images are ~150 DPI; Audiveris needs ~300)")
    args = parser.parse_args()

    from datasets import load_dataset
    from eval_mxc import evaluate_pair, _ensure_mxc

    print(f"Loading {args.dataset_repo} config={args.dataset_config} split={args.split}")
    ds = load_dataset(args.dataset_repo, args.dataset_config, split=args.split)
    n = min(args.n_samples, len(ds))
    print(f"Evaluating {n} samples")

    audiveris_bin = find_audiveris()
    print(f"Audiveris: {audiveris_bin}")

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="audiveris_eval_"))
    workdir.mkdir(parents=True, exist_ok=True)
    img_dir = workdir / "images"
    out_dir = workdir / "audiveris_out"
    img_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    results = []
    failures = 0
    out_path = Path(args.output)
    t0 = time.time()

    with open(out_path, "w") as fout:
        for i in range(n):
            sample = ds[i]
            sample_id = sample.get("score_id", f"sample_{i:04d}")
            img_path = img_dir / f"{i:04d}.png"
            img = sample["image"]
            if args.upscale and args.upscale != 1.0:
                from PIL import Image
                w, h = img.size
                img = img.resize(
                    (int(w * args.upscale), int(h * args.upscale)),
                    Image.LANCZOS,
                )
            img.save(img_path)
            sample_out_dir = out_dir / f"{i:04d}"

            try:
                mxl = run_audiveris(audiveris_bin, img_path, sample_out_dir, verbose=False)
                pred_xml = read_mxl(mxl)
                ref_xml = sample["musicxml"]
                # evaluate_pair auto-converts ref XML→MXC but assumes pred is already
                # MXC; convert pred XML explicitly so both go through the same parser.
                pred_mxc = _ensure_mxc(pred_xml)
                r = evaluate_pair(pred_mxc, ref_xml, score_id=sample_id)

                rec = {
                    "i": i,
                    "score_id": sample_id,
                    "pitched_only_similarity": r.pitched_only_similarity,
                    "pitched_only_positional": r.pitched_only_positional,
                    "rhythm_similarity": r.rhythm_similarity,
                    "combined_similarity": r.combined_similarity,
                    "note_coverage": r.note_coverage,
                    "pred_notes": r.pred_notes,
                    "ref_notes": r.ref_notes,
                }
                results.append(rec)
            except Exception as e:
                failures += 1
                rec = {"i": i, "score_id": sample_id, "error": str(e)[:200]}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

            elapsed = time.time() - t0
            ok = "error" not in rec
            sim_str = f"{rec.get('pitched_only_similarity', 0):.1%}" if ok else "FAILED"
            print(f"  [{i:3d}/{n}] {sample_id}: {sim_str}  ({elapsed/(i+1):.0f}s/sample)")

    # Aggregate
    n_ok = len(results)
    print(f"\n{'='*60}")
    print(f"Audiveris on {args.dataset_config} {args.split}, n={n} ({failures} failed)")
    print(f"{'='*60}")
    if n_ok:
        for key in ("pitched_only_similarity", "pitched_only_positional",
                    "rhythm_similarity", "combined_similarity", "note_coverage"):
            vals = [r[key] for r in results]
            print(f"  {key:30s}: {sum(vals)/n_ok:.1%}  (max {max(vals):.1%})")
        print(f"  Total time: {time.time() - t0:.0f}s")
    print(f"Per-sample results saved to {out_path}")


if __name__ == "__main__":
    main()
