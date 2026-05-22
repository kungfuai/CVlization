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

CLASS_NAMES = ["system", "staff", "barline_single", "barline_heavy"]
NUM_CLASSES = len(CLASS_NAMES)


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


def _record_to_yolo_lines(rec: dict) -> list[str]:
    """Convert one JSONL detection record into YOLO label lines."""
    img_w, img_h = rec["width"], rec["height"]
    out: list[str] = []
    for x, y, w, h in rec["bboxes"]["systems"]:
        cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
        out.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    for _sys, _idx, x, y, w, h in rec["bboxes"]["staves"]:
        cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
        out.append(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    for _sys, x, y, w, h, heavy in rec["bboxes"]["barlines"]:
        cls = 3 if heavy else 2
        cx, cy, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
        out.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return out


def _build_yolo_split(src_data: Path, jsonl_name: str, dst_dir: Path) -> int:
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
                # symlink so we don't duplicate bytes; ultralytics is fine with it
                dst_img.symlink_to(src_img.resolve())
            dst_lbl = labels_out / (src_img.stem + ".txt")
            dst_lbl.write_text("\n".join(_record_to_yolo_lines(rec)))
            n += 1
    return n


def prepare_yolo_layout(src_data: Path, dst_root: Path) -> Path:
    """Produce <dst_root>/{train,val}/{images,labels} + dataset.yaml."""
    if dst_root.exists():
        shutil.rmtree(dst_root)
    train_dir = dst_root / "train"
    val_dir = dst_root / "val"
    n_train = _build_yolo_split(src_data, "labels_train.jsonl", train_dir)
    n_val = _build_yolo_split(src_data, "labels_dev.jsonl", val_dir)
    # If only one split exists, mirror it (single-source smoke runs)
    if n_train and not n_val:
        n_val = _build_yolo_split(src_data, "labels_train.jsonl", val_dir)
    if n_val and not n_train:
        n_train = _build_yolo_split(src_data, "labels_dev.jsonl", train_dir)

    print(f"YOLO layout: train={n_train} val={n_val}", flush=True)

    ds_yaml = dst_root / "dataset.yaml"
    ds_yaml.write_text(yaml.safe_dump({
        "path": str(dst_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
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
    ds_yaml = prepare_yolo_layout(args.data, yolo_root)

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
