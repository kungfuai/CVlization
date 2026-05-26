#!/usr/bin/env python3
"""End-to-end eval of the per-measure pipeline on a det_v* dev set.

For each dev page:
  1. Run pipeline_per_measure -> predicted MXC2
  2. Convert the page's GT musicxml -> reference MXC2 via xml_to_mxc2
  3. Compute eval_mxc.evaluate_pair -> per-page metrics
  4. Aggregate across pages

Output: prints summary + writes per-page CSV + a representative side-by-side
JSON of (pred, ref) for the first N records.
"""
import argparse
import json
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent / "vlm_omr_sft"))
sys.path.insert(0, str(_THIS.parent.parent / "omr_detection"))
sys.path.insert(0, str(_THIS.parent))

from mxc2 import xml_to_mxc2  # noqa: E402
from eval_mxc import evaluate_pair  # noqa: E402
from pipeline_per_measure import run as pipeline_run  # noqa: E402


def _xml_to_mxc2_safe(xml: str) -> str:
    try:
        return xml_to_mxc2(xml)
    except Exception as e:
        return f"# xml_to_mxc2 failed: {e}"


def _gt_from_per_measure(per_measure_dir: Path, source: str, sid: str,
                         page: int) -> str | None:
    """Reconstruct a page-level GT MXC2 in the SAME format the pipeline's
    stitch_measures produces (per-part flat, not per-measure interleaved).
    Returns None if no matching measures exist.
    """
    from pipeline_per_measure import stitch_measures
    bodies_by_m: dict[int, str] = {}
    for split in ("dev", "train", "test"):
        f = per_measure_dir / f"labels_{split}.jsonl"
        if not f.exists():
            continue
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if (rec.get("source") == source
                        and rec.get("score_id") == sid
                        and rec.get("page") == page):
                    m = rec.get("measure")
                    mxc2 = rec.get("mxc2") or ""
                    if mxc2 and m is not None and m not in bodies_by_m:
                        bodies_by_m[m] = mxc2
    if not bodies_by_m:
        return None
    pairs = [(m, bodies_by_m[m]) for m in sorted(bodies_by_m)]
    return stitch_measures(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, type=Path,
                    help="det_v* dir with labels_dev.jsonl + images/")
    ap.add_argument("--per-measure-dir", required=True, type=Path,
                    help="per_measure_v4_* dir to source GT MXC2 from "
                         "(measure-level records keyed by source+score_id+page)")
    ap.add_argument("--source", required=True,
                    help="source tag for matching per-measure records "
                         "(l7a / l9 / openscore)")
    ap.add_argument("--det-ckpt", required=True, type=Path)
    ap.add_argument("--vlm-ckpt", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=10,
                    help="cap dev pages for quick smoke runs")
    ap.add_argument("--save-pairs", type=int, default=3,
                    help="dump (pred,ref) text for first N records")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src_jsonl = args.src_dir / "labels_dev.jsonl"
    if not src_jsonl.exists():
        print(f"missing {src_jsonl}", file=sys.stderr); sys.exit(1)
    recs = [json.loads(l) for l in src_jsonl.open() if l.strip()][:args.limit]
    print(f"Evaluating {len(recs)} pages from {src_jsonl}", flush=True)

    results = []
    pairs_dump = []
    for i, rec in enumerate(recs):
        sid = rec["score_id"]
        page = rec.get("page", 1)
        bar_start = rec.get("bar_start") or 1
        img_path = args.src_dir / rec["image"]
        t0 = time.time()
        try:
            pred = pipeline_run(str(img_path), str(args.det_ckpt),
                                 str(args.vlm_ckpt),
                                 imgsz=1280, conf=0.25,
                                 bar_start=bar_start, verbose=False)
        except Exception as e:
            print(f"  [{i+1}/{len(recs)}] {sid} p{page}: PIPELINE FAIL {e!r}",
                  flush=True)
            continue
        # GT from per-measure dataset (already converted via xml_to_mxc2)
        ref = _gt_from_per_measure(args.per_measure_dir, args.source,
                                   sid, page)
        if ref is None:
            print(f"    skipping {sid} p{page}: no GT per-measure records",
                  flush=True)
            continue
        r = evaluate_pair(pred, ref, score_id=sid)
        results.append(r)
        dt = time.time() - t0
        print(f"  [{i+1}/{len(recs)}] {sid} p{page}: pitched={r.pitched_only_similarity:.3f} "
              f"pitch_pos={r.position_only_similarity:.3f} rhythm={r.note_type_similarity:.3f} "
              f"key_ok={int(r.correct_key)} ({dt:.1f}s)", flush=True)
        if i < args.save_pairs:
            pairs_dump.append({
                "score_id": sid, "page": page, "bar_start": bar_start,
                "metrics": {
                    "pitched_only_similarity": r.pitched_only_similarity,
                    "position_only_similarity": r.position_only_similarity,
                    "note_type_similarity": r.note_type_similarity,
                    "correct_key": r.correct_key,
                    "correct_time": r.correct_time,
                    "pred_events": r.pred_events, "ref_events": r.ref_events,
                },
                "pred": pred[:4000], "ref": ref[:4000],
            })

    # Aggregate
    if not results:
        print("no results", flush=True); return
    import statistics
    def col(name): return [getattr(r, name) for r in results]
    print(f"\n=== aggregate over {len(results)} pages ===", flush=True)
    for m in ("pitched_only_similarity", "position_only_similarity",
              "note_type_similarity", "combined_similarity",
              "rhythm_similarity"):
        vals = [v for v in col(m) if v is not None]
        if vals:
            print(f"  {m}: mean={statistics.mean(vals):.3f} "
                  f"median={statistics.median(vals):.3f} "
                  f"min={min(vals):.3f} max={max(vals):.3f}", flush=True)
    print(f"  correct_key:  {sum(col('correct_key'))}/{len(results)}",
          flush=True)
    print(f"  correct_time: {sum(col('correct_time'))}/{len(results)}",
          flush=True)

    # Persist
    (args.out_dir / "metrics.json").write_text(json.dumps([
        {**{k: getattr(r, k) for k in r.__dataclass_fields__}, } for r in results
    ], indent=2, default=str))
    (args.out_dir / "pairs.json").write_text(json.dumps(pairs_dump, indent=2))
    print(f"\nWrote {args.out_dir / 'metrics.json'} and pairs.json",
          flush=True)


if __name__ == "__main__":
    main()
