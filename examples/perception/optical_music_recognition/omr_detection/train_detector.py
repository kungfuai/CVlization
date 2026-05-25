#!/usr/bin/env python3
"""Train YOLOv8n on the OMR layout-detection dataset.

Takes a directory produced by labels/make_dataset.py:

    <data_dir>/
        images/<score_id>_p<page>.png
        labels_train.jsonl
        labels_dev.jsonl

Converts each JSONL line into ultralytics YOLO label format (one .txt
per image, normalized cx/cy/w/h in [0,1]), writes a temporary
ultralytics dataset.yaml, then calls `YOLO("yolov8n.pt").train(...)`.

Classes:
    0 - system
    1 - staff
    2 - barline_single
    3 - barline_heavy
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

CLASS_NAMES = ["system", "staff", "barline_single", "barline_heavy",
               "key_signature"]
NUM_CLASSES = len(CLASS_NAMES)

# Multi-task variant: replace the single key_signature class with 15
# sub-classes, one per fifths value -7..+7. The detector simultaneously
# localises and classifies the keysig in one forward pass -- no
# downstream CNN needed.
MULTITASK_BASE_CLASSES = ["system", "staff", "barline_single", "barline_heavy"]
KEY_MIN_MT, KEY_MAX_MT = -7, 7
CLASS_NAMES_MT = MULTITASK_BASE_CLASSES + [
    f"key_{i:+d}" for i in range(KEY_MIN_MT, KEY_MAX_MT + 1)
]
NUM_CLASSES_MT = len(CLASS_NAMES_MT)   # 4 + 15 = 19


def _keysig_class_for_fifths(fifths: int) -> int | None:
    """In multi-task mode, the YOLO class index for a keysig of N fifths."""
    if fifths < KEY_MIN_MT or fifths > KEY_MAX_MT:
        return None
    return len(MULTITASK_BASE_CLASSES) + (fifths - KEY_MIN_MT)


def fifths_from_multitask_class(cls: int) -> int | None:
    """Inverse of _keysig_class_for_fifths. Returns None for non-keysig classes."""
    base = len(MULTITASK_BASE_CLASSES)
    if cls < base or cls >= NUM_CLASSES_MT:
        return None
    return cls - base + KEY_MIN_MT

# YOLOv8 struggles with sub-pixel-wide boxes. Real barlines are ~1 px
# wide at our 1280px imgsz, which the detector can't localize. We widen
# them in the YOLO label space; downstream we only care about x-center,
# so the inflated box doesn't hurt cell derivation.
#
# Width is expressed as a multiple of staff-space (one of the 4 gaps
# between a staff's 5 lines), so the same recipe works across image
# resolutions and engraving styles. ~0.7 staff-space gives an 8-9 px
# wide bbox at L7a's HF rendering and scales naturally elsewhere.
MIN_BARLINE_W_STAFF_FRAC = 0.7
# Absolute floor for sources where staff height is unknown.
MIN_BARLINE_W_PX_FLOOR = 6


def _xywh_to_yolo(x: float, y: float, w: float, h: float,
                  img_w: int, img_h: int) -> tuple[float, float, float, float]:
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0,1] -- a tiny barline can rarely extend past the page
    # crop after viewbox rescaling.
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh


def _staff_space_px(rec: dict) -> float | None:
    """Average staff-space (gap between two staff lines) in this record's
    pixel space, derived from the staff bboxes (5 lines = 4 gaps).
    Returns None if no staves were detected."""
    staves = rec["bboxes"]["staves"]
    if not staves:
        return None
    heights = [s[5] for s in staves]  # (sys, idx, x, y, w, h) -> h
    return (sum(heights) / len(heights)) / 4.0


def _record_to_yolo_lines(rec: dict, multitask: bool = False) -> list[str]:
    """Convert one JSONL detection record into YOLO label lines.

    multitask=True: keysig boxes are emitted as one of 15 keysig
    sub-classes (key_-7 ... key_+7) based on rec['key_first']. Records
    with missing or out-of-range key_first contribute no keysig labels
    (their structural labels still go through)."""
    img_w, img_h = rec["width"], rec["height"]
    out: list[str] = []
    ss = _staff_space_px(rec)
    min_bar_w = max(MIN_BARLINE_W_PX_FLOOR,
                    (MIN_BARLINE_W_STAFF_FRAC * ss) if ss else 0)
    for x, y, w, h in rec["bboxes"]["systems"]:
        cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
        out.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    for _sys, _idx, x, y, w, h in rec["bboxes"]["staves"]:
        cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
        out.append(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    # Barline tuple is either 6 fields (old schema) or 7 (new with kind_id).
    for item in rec["bboxes"]["barlines"]:
        _sys, x, y, w, h, heavy = item[:6]
        cls = 3 if heavy else 2
        if w < min_bar_w:
            x = x + w / 2 - min_bar_w / 2
            w = min_bar_w
        cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
        out.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # Key signatures.
    #   Old schema: [sys, x, y, w, h]  (5 fields, fifths from rec['key_first'])
    #   New schema: [sys, x, y, w, h, fifths, kind_id]  (7 fields, per-item)
    for item in rec["bboxes"].get("key_sigs", []):
        _sys, x, y, w, h = item[:5]
        fifths = item[5] if len(item) >= 6 else rec.get("key_first")
        if multitask:
            cls_keysig = (_keysig_class_for_fifths(fifths)
                          if fifths is not None else None)
            if cls_keysig is None:
                continue
            cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
            out.append(f"{cls_keysig} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        else:
            cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
            out.append(f"4 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return out


def _build_yolo_split(src_data: Path, jsonl_name: str, dst_dir: Path,
                       multitask: bool = False) -> int:
    """Symlink images + write YOLO .txt labels for one split. Returns count."""
    images_out = dst_dir / "images"
    labels_out = dst_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    src_jsonl = src_data / jsonl_name
    if not src_jsonl.exists():
        return 0
    n = 0
    with src_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            src_img = src_data / rec["image"]
            if not src_img.exists():
                continue
            dst_img = images_out / src_img.name
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())
            dst_lbl = labels_out / (src_img.stem + ".txt")
            dst_lbl.write_text("\n".join(_record_to_yolo_lines(rec, multitask=multitask)))
            n += 1
    return n


def prepare_yolo_layout(src_data: Path, dst_root: Path,
                        multitask: bool = False) -> Path:
    """Produce <dst_root>/{train,val}/{images,labels} + dataset.yaml."""
    if dst_root.exists():
        shutil.rmtree(dst_root)
    train_dir = dst_root / "train"
    val_dir = dst_root / "val"
    n_train = _build_yolo_split(src_data, "labels_train.jsonl", train_dir, multitask=multitask)
    n_val = _build_yolo_split(src_data, "labels_dev.jsonl", val_dir, multitask=multitask)
    # If only one split exists, mirror it (single-source smoke runs)
    if n_train and not n_val:
        n_val = _build_yolo_split(src_data, "labels_train.jsonl", val_dir)
    if n_val and not n_train:
        n_train = _build_yolo_split(src_data, "labels_dev.jsonl", train_dir)

    print(f"YOLO layout: train={n_train} val={n_val}", flush=True)

    nc = NUM_CLASSES_MT if multitask else NUM_CLASSES
    names = CLASS_NAMES_MT if multitask else CLASS_NAMES
    ds_yaml = dst_root / "dataset.yaml"
    ds_yaml.write_text(yaml.safe_dump({
        "path": str(dst_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": nc,
        "names": names,
    }))
    return ds_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="dir produced by labels/make_dataset.py")
    p.add_argument("--config", type=Path,
                   default=Path("configs/detector_l7a.yaml"))
    p.add_argument("--output", type=Path, default=Path("outputs/detector_l7a"))
    p.add_argument("--epochs", type=int, default=None,
                   help="override cfg.training.epochs")
    p.add_argument("--limit-epochs-for-smoke", type=int, default=None,
                   help="cap epochs for fast smoke testing")
    p.add_argument("--multitask", action="store_true",
                   help="replace single key_signature class with 15 "
                        "keysig sub-classes (key_-7 ... key_+7). 19 total.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tr = cfg.get("training", {})
    epochs = args.epochs or tr.get("epochs", 50)
    if args.limit_epochs_for_smoke is not None:
        epochs = args.limit_epochs_for_smoke
    img_size = tr.get("image_size", 1280)
    batch = tr.get("batch_size", 16)
    lr = tr.get("lr", 1e-3)
    backbone = cfg.get("model", {}).get("backbone", "yolov8n")

    yolo_root = args.output / "yolo_data"
    ds_yaml = prepare_yolo_layout(args.data, yolo_root, multitask=args.multitask)
    if args.multitask:
        print(f"Multi-task mode: 19 classes "
              f"(4 structural + 15 keysig sub-classes)", flush=True)

    # Import ultralytics lazily so --help works without it.
    from ultralytics import YOLO  # type: ignore
    weights = f"{backbone}.pt"
    print(f"Training {backbone} epochs={epochs} imgsz={img_size} "
          f"batch={batch} lr={lr}", flush=True)
    model = YOLO(weights)
    model.train(
        data=str(ds_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        lr0=lr,
        project=str(args.output),
        name="run",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
