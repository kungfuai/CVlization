#!/usr/bin/env python3
"""Pre-compute Audiveris MXC2 hints for a dataset slice.

Saves JSONL with per-sample: score_id, audiveris_mxc2, audiveris_failed.
This output is consumed by the VLM in-context evaluation, which prepends
each sample's hint to its prompt.

Usage:
    python extract_hints.py -n 50 --output audiveris_hints_level9_n50.jsonl
"""

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

SFT_DIR = "/cvlization_repo/examples/perception/optical_music_recognition/vlm_omr_sft"
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from predict import find_audiveris, run_audiveris
from eval_synthetic_scores import read_mxl


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--n-samples", type=int, default=50)
    p.add_argument("--start", type=int, default=0,
                   help="Start index (for sharding across parallel workers)")
    p.add_argument("--dataset-repo", default="zzsi/synthetic-scores")
    p.add_argument("--dataset-config", default="level9")
    p.add_argument("--split", default="test")
    p.add_argument("--output", default="audiveris_hints.jsonl")
    p.add_argument("--upscale", type=float, default=2.5)
    p.add_argument("--drop-beams", action="store_true", default=True)
    args = p.parse_args()

    from datasets import load_dataset
    from PIL import Image
    from mxc2 import xml_to_mxc2

    print(f"Loading {args.dataset_repo}/{args.dataset_config}/{args.split}")
    ds = load_dataset(args.dataset_repo, args.dataset_config, split=args.split)
    end = min(args.start + args.n_samples, len(ds))
    print(f"Processing samples [{args.start}, {end}) of {len(ds)}")

    audiveris_bin = find_audiveris()
    workdir = Path(tempfile.mkdtemp(prefix="audiveris_hints_"))
    img_dir = workdir / "images"
    out_dir = workdir / "out"
    img_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    t0 = time.time()
    n_ok, n_fail = 0, 0
    with open(args.output, "w") as fout:
        for j, i in enumerate(range(args.start, end)):
            sample = ds[i]
            sample_id = sample.get("score_id", f"sample_{i:04d}")
            img = sample["image"]
            if args.upscale != 1.0:
                w, h = img.size
                img = img.resize(
                    (int(w * args.upscale), int(h * args.upscale)),
                    Image.LANCZOS,
                )
            img_path = img_dir / f"{i:04d}.png"
            img.save(img_path)
            sample_out = out_dir / f"{i:04d}"

            record = {"i": i, "score_id": sample_id}
            try:
                mxl = run_audiveris(audiveris_bin, img_path, sample_out, verbose=False)
                pred_xml = read_mxl(mxl)
                hint_mxc2 = xml_to_mxc2(pred_xml, drop_beams=args.drop_beams)
                record["audiveris_mxc2"] = hint_mxc2
                record["audiveris_failed"] = False
                n_ok += 1
            except Exception as e:
                record["audiveris_mxc2"] = ""
                record["audiveris_failed"] = True
                record["error"] = str(e)[:200]
                n_fail += 1

            fout.write(json.dumps(record) + "\n")
            fout.flush()
            elapsed = time.time() - t0
            status = "OK" if not record.get("audiveris_failed") else "FAIL"
            hint_len = len(record.get("audiveris_mxc2", ""))
            print(f"  [{j+1:4d}/{end-args.start}] idx={i} {sample_id}: {status} "
                  f"hint={hint_len}  ({elapsed/(j+1):.0f}s/sample)")

    print(f"\nDone. {n_ok} ok, {n_fail} failed. Total: {time.time()-t0:.0f}s")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
