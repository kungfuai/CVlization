#!/usr/bin/env python3
"""Train SmallKeyCNN on YOLO-detected keysig crops from HF images.

Self-consistent: the training crops come from running YOLO on each L7a
HF image and taking the predicted key_signature boxes. The inference
crops at deploy time come from the same YOLO call on the same HF image
distribution. No coordinate scaling between two renderings, no
re-rendered-vs-HF mismatch.

Usage (inside Docker via train_keysig_yolocrop.sh):
    python train_keysig_yolocrop.py --det-ckpt outputs/.../best.pt \\
        --output outputs/keysig_cnn_yolocrop --epochs 12
"""

import argparse
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_THIS = Path(__file__).resolve().parent
_VLM_DIR = _THIS.parent / "vlm_omr_sft"
sys.path.insert(0, str(_VLM_DIR))
sys.path.insert(0, str(_THIS))

from train_keyclf_cnn import (  # type: ignore
    SmallKeyCNN, _EVAL_TFM, _TRAIN_TFM, fifths_to_label,
)
from pipeline import load_detector, detect_layout, _crop  # noqa: E402


def _gt_key(mxl_text: str) -> int:
    m = re.search(r"<fifths>(-?\d+)</fifths>", mxl_text)
    return int(m.group(1)) if m else 0


def _inflate_down(box, target_aspect: float = 3.0):
    x, y, w, h = box
    target_h = w / target_aspect
    return x, y, w, max(h, target_h)


def collect_crops(det, split: str) -> list[tuple]:
    """Run the detector on every HF image in this split; return one
    sample per detected key_signature box."""
    import os
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    from datasets import load_dataset
    ds = load_dataset("zzsi/synthetic-scores", "level7a", split=split)
    items: list[tuple] = []
    tmp = Path("/tmp/_keysig_train")
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, row in enumerate(ds):
        sid = row["score_id"]
        img = row["image"].convert("RGB")
        path = tmp / f"{sid}.png"
        img.save(path)
        layout = detect_layout(det, str(path))
        path.unlink(missing_ok=True)
        key_label = fifths_to_label(_gt_key(row["musicxml"]))
        for box in layout["key_signatures"]:
            x, y, w, h = _inflate_down(box)
            crop = _crop(img, (x, y, w, h), 4)
            items.append((crop, key_label))
        if (i + 1) % 50 == 0:
            print(f"  [{split}] {i+1}/{len(ds)} ({time.time()-t0:.1f}s, "
                  f"crops={len(items)})", flush=True)
    return items


class _CropDataset(Dataset):
    def __init__(self, items: list[tuple], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        crop, label = self.items[i]
        return self.tfm(crop), label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det-ckpt", required=True,
                   help="YOLO with key_signature class")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading detector ...", flush=True)
    det = load_detector(args.det_ckpt)

    print("Collecting train crops via YOLO ...", flush=True)
    train_items = collect_crops(det, "train")
    print("Collecting dev crops via YOLO ...", flush=True)
    dev_items = collect_crops(det, "dev")
    print(f"train crops: {len(train_items)}, dev crops: {len(dev_items)}",
          flush=True)

    train_dl = DataLoader(_CropDataset(train_items, _TRAIN_TFM),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=2)
    dev_dl = DataLoader(_CropDataset(dev_items, _EVAL_TFM),
                        batch_size=args.batch_size, num_workers=2)

    model = SmallKeyCNN().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        ok = total = 0
        with torch.no_grad():
            for x, y in dev_dl:
                x, y = x.cuda(), y.cuda()
                pred = model(x).argmax(1)
                ok += int((pred == y).sum().item())
                total += y.numel()
        acc = ok / max(total, 1)
        print(f"epoch {ep+1}/{args.epochs}  dev_acc={acc:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict()},
                       args.output / "best.pt")
    print(f"best dev_acc={best:.4f} -> {args.output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
