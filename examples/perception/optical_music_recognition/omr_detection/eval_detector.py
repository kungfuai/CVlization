#!/usr/bin/env python3
"""Evaluate a trained layout detector.

Two checks:
  1. Detection quality -- per-class P / R / mAP50 / mAP50-95 via
     ultralytics YOLO.val on the dev split.
  2. Cell-derivation sanity -- run the detector on each dev page, derive
     cells, and compare measures-per-page against the ground-truth
     measure count (len(bar_numbers) in the dataset labels). A correct
     detector + derivation should match exactly.

Usage (inside Docker via eval.sh):
    python eval_detector.py --data /data \\
        --checkpoint outputs/detector_l7a_500_wide/run/weights/best.pt
"""

import argparse
import json
from pathlib import Path

from cells import derive_cells, measures_per_system
from pipeline import detect_layout, load_detector
from train_detector import prepare_yolo_layout


def detection_metrics(checkpoint: str, ds_yaml: Path, imgsz: int) -> None:
    from ultralytics import YOLO
    model = YOLO(checkpoint)
    res = model.val(data=str(ds_yaml), imgsz=imgsz, verbose=False,
                    plots=False)
    names = ["system", "staff", "barline_single", "barline_heavy"]
    print("\n=== Detection metrics (dev) ===")
    print(f"{'class':18s} {'P':>7s} {'R':>7s} {'mAP50':>8s} {'mAP50-95':>9s}")
    for i, name in enumerate(names):
        try:
            print(f"{name:18s} {float(res.box.p[i]):7.3f} "
                  f"{float(res.box.r[i]):7.3f} {float(res.box.ap50[i]):8.3f} "
                  f"{float(res.box.ap[i]):9.3f}")
        except (IndexError, AttributeError):
            print(f"{name:18s} {'--':>7s} {'--':>7s} {'--':>8s} "
                  f"(no instances)")
    print(f"{'all':18s} {res.box.mp:7.3f} {res.box.mr:7.3f} "
          f"{res.box.map50:8.3f} {res.box.map:9.3f}")


def _gt_layout(rec: dict) -> dict:
    """Reconstruct a detect_layout-shaped dict from a dataset record.

    Dataset bbox lists are nested: systems [x,y,w,h], staves
    [sys,idx,x,y,w,h], barlines [sys,x,y,w,h,heavy].
    """
    bb = rec["bboxes"]
    return {
        "systems": [tuple(b) for b in bb["systems"]],
        "staves": [tuple(b[2:6]) for b in bb["staves"]],
        "barlines": [tuple(b[1:5]) for b in bb["barlines"]],
    }


def _measure_count(layout: dict) -> int:
    cells = derive_cells(layout["systems"], layout["staves"],
                         layout["barlines"])
    return sum(measures_per_system(cells).values())


def cell_sanity(checkpoint: str, data_dir: Path, imgsz: int,
                conf: float) -> None:
    """Two checks per dev page:

    A. predicted-derived measure count vs GT-box-derived count
       -- isolates detector + derivation consistency.
    B. GT-box-derived count vs n_measures (true count from MusicXML)
       -- validates the derivation logic itself. Single-page scores only.
    """
    labels = data_dir / "labels_dev.jsonl"
    if not labels.exists():
        print(f"\n[cell sanity] {labels} missing -- skipped")
        return
    model = load_detector(checkpoint)
    n_pages = pred_ok = deriv_ok = deriv_checked = 0
    pred_abs_err = 0
    pred_bad: list[tuple] = []
    deriv_bad: list[tuple] = []
    with labels.open() as f:
        for line in f:
            rec = json.loads(line)
            img = data_dir / rec["image"]
            if not img.exists():
                continue
            n_pages += 1
            gt_layout = _gt_layout(rec)
            gt_count = _measure_count(gt_layout)

            pred_layout = detect_layout(model, str(img), imgsz=imgsz,
                                        conf=conf)
            pred_count = _measure_count(pred_layout)
            err = pred_count - gt_count
            pred_abs_err += abs(err)
            if err == 0:
                pred_ok += 1
            else:
                pred_bad.append((rec["score_id"], gt_count, pred_count))

            # Check B only makes sense when the page holds the whole score.
            if rec.get("n_measures") is not None and rec.get("n_pages", 1) == 1:
                deriv_checked += 1
                if gt_count == rec["n_measures"]:
                    deriv_ok += 1
                else:
                    deriv_bad.append((rec["score_id"], rec["n_measures"],
                                      gt_count))

    print("\n=== Cell-derivation sanity (dev) ===")
    if n_pages == 0:
        print("no pages evaluated")
        return
    print(f"pages: {n_pages}")
    print(f"A. predicted vs GT-derived:  {pred_ok}/{n_pages} exact "
          f"({100*pred_ok/n_pages:.1f}%), mean abs err "
          f"{pred_abs_err/n_pages:.3f}")
    for sid, gt, pred in pred_bad[:8]:
        print(f"   MISMATCH {sid}: gt-derived={gt} pred-derived={pred}")
    if deriv_checked:
        print(f"B. GT-derived vs n_measures: {deriv_ok}/{deriv_checked} "
              f"exact ({100*deriv_ok/deriv_checked:.1f}%)")
        for sid, true_n, gt in deriv_bad[:8]:
            print(f"   MISMATCH {sid}: n_measures={true_n} gt-derived={gt}")
    else:
        print("B. skipped -- no n_measures in labels "
              "(re-render with the updated make_dataset.py)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="dataset dir from labels/make_dataset.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--skip-sanity", action="store_true")
    args = p.parse_args()

    if not args.skip_metrics:
        yolo_root = Path("outputs/_eval_yolo_data")
        ds_yaml = prepare_yolo_layout(args.data, yolo_root)
        detection_metrics(args.checkpoint, ds_yaml, args.imgsz)
    if not args.skip_sanity:
        cell_sanity(args.checkpoint, args.data, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
