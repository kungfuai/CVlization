#!/usr/bin/env python3
"""Train SmallKeyCNN on YOLO-detected keysig crops from multiple sources.

Pulls L7a, L9, and openscore HF images, runs the YOLO keysig detector
on each, and trains a single CNN on the union of detected-and-cropped
key signatures. Labels are the GT `<fifths>` from each MusicXML.

Leak prevention: we use HF's own `train` / `dev` splits per config and
ASSERT zero `score_id` overlap between train and dev across every
config before training.

Usage (inside Docker via train_keysig_multisource.sh):
    python train_keysig_multisource.py \\
        --det-ckpt outputs/detector_l7a_500_v3/run/weights/best.pt \\
        --output outputs/keysig_cnn_multisource \\
        --epochs 12 \\
        --per-source-train 500 --per-source-dev 100
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
    SmallKeyCNN, _EVAL_TFM, _TRAIN_TFM,
)
from pipeline import load_detector, detect_layout, _crop  # noqa: E402

# Sources to mix. Streaming for openscore (it's large; the others fit
# in memory comfortably). Each entry: (repo, config, streaming?).
SOURCES = [
    ("zzsi/synthetic-scores", "level7a", False),
    ("zzsi/synthetic-scores", "level9",  False),
    ("zzsi/openscore",        "pages_transcribed", True),
]

# Wider key-signature range than the legacy 9-class (-4..+4) CNN.
# Openscore has ~16% of scores with |fifths| in {5, 6}; -7..+7 covers
# every theoretical major/minor key. We retrain SmallKeyCNN at 15 classes
# from scratch.
KEY_MIN, KEY_MAX = -7, 7
N_CLASSES = KEY_MAX - KEY_MIN + 1   # 15


def fifths_to_label(f: int) -> int | None:
    if f < KEY_MIN or f > KEY_MAX:
        return None
    return f - KEY_MIN


def label_to_fifths(l: int) -> int:
    return l + KEY_MIN


def _gt_key(mxl_text: str) -> int:
    m = re.search(r"<fifths>(-?\d+)</fifths>", mxl_text)
    return int(m.group(1)) if m else 0


def _inflate_down(box, target_aspect: float = 3.0):
    x, y, w, h = box
    target_h = w / target_aspect
    return x, y, w, max(h, target_h)


def _iter_rows(repo: str, cfg: str, split: str, streaming: bool, limit: int):
    """Yield up to `limit` rows from one (repo, cfg, split)."""
    import os
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    from datasets import load_dataset
    ds = load_dataset(repo, cfg, split=split, streaming=streaming)
    if streaming:
        it = iter(ds)
        for _ in range(limit):
            try:
                yield next(it)
            except StopIteration:
                return
    else:
        n = min(limit, len(ds))
        for i in range(n):
            yield ds[i]


def collect_crops(det, split: str, limit: int) -> tuple[list[tuple], set[str]]:
    """For each source, run YOLO on `limit` pages of `split`; return
    (crops list, set of score_ids consumed).

    Each crop carries its source tag so we can break out per-source
    metrics during eval.
    """
    items: list[tuple] = []   # (crop_pil, label, source_tag)
    seen_ids: set[str] = set()
    tmp = Path("/tmp/_keysig_multi")
    tmp.mkdir(parents=True, exist_ok=True)

    for repo, cfg, stream in SOURCES:
        tag = cfg if cfg != "pages_transcribed" else "openscore"
        t0 = time.time()
        rows = list(_iter_rows(repo, cfg, split, stream, limit))
        print(f"  [{split}/{tag}] loaded {len(rows)} rows "
              f"({time.time()-t0:.1f}s)", flush=True)
        oor = 0  # out-of-range key signatures skipped
        for j, row in enumerate(rows):
            sid = row.get("score_id") or f"{tag}_{split}_{j:06d}"
            seen_ids.add(f"{tag}::{sid}")
            mxl = row.get("musicxml")
            if not mxl:
                continue
            fifths = _gt_key(mxl)
            key_label = fifths_to_label(fifths)
            if key_label is None:
                oor += 1
                continue
            img = row["image"].convert("RGB")
            path = tmp / "page.png"
            img.save(path)
            layout = detect_layout(det, str(path))
            path.unlink(missing_ok=True)
            for box in layout["key_signatures"]:
                x, y, w, h = _inflate_down(box)
                crop = _crop(img, (x, y, w, h), 4)
                items.append((crop, key_label, tag))
            if (j + 1) % 50 == 0:
                print(f"    {tag} {j+1}/{len(rows)}  crops={len(items)}  "
                      f"oor={oor}", flush=True)
        if oor:
            print(f"    {tag} skipped {oor} pages with |fifths|>{KEY_MAX}",
                  flush=True)
    return items, seen_ids


class _CropDataset(Dataset):
    def __init__(self, items: list[tuple], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        crop, label, _tag = self.items[i]
        return self.tfm(crop), label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det-ckpt", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--per-source-train", type=int, default=500)
    p.add_argument("--per-source-dev",   type=int, default=100)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading detector ...", flush=True)
    det = load_detector(args.det_ckpt)

    print(f"\nCollecting TRAIN crops "
          f"(<= {args.per_source_train} pages per source) ...", flush=True)
    train_items, train_ids = collect_crops(det, "train", args.per_source_train)
    print(f"\nCollecting DEV crops "
          f"(<= {args.per_source_dev} pages per source) ...", flush=True)
    dev_items, dev_ids = collect_crops(det, "dev", args.per_source_dev)

    # LEAKAGE CHECK: train and dev score_id sets must not overlap.
    overlap = train_ids & dev_ids
    assert not overlap, (
        f"train/dev score_id leakage detected ({len(overlap)} overlapping ids); "
        f"example: {next(iter(overlap))}")
    print(f"\n  train: {len(train_items)} crops from "
          f"{len(train_ids)} unique scores", flush=True)
    print(f"  dev:   {len(dev_items)} crops from "
          f"{len(dev_ids)} unique scores", flush=True)
    print(f"  leak-check: 0 overlapping score_ids (PASSED)\n", flush=True)

    train_dl = DataLoader(_CropDataset(train_items, _TRAIN_TFM),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=2)
    dev_dl = DataLoader(_CropDataset(dev_items, _EVAL_TFM),
                        batch_size=args.batch_size, num_workers=2)

    # Per-source eval indices
    src_indices: dict[str, list[int]] = {}
    for i, (_, _, tag) in enumerate(dev_items):
        src_indices.setdefault(tag, []).append(i)

    model = SmallKeyCNN(n_classes=N_CLASSES).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    def eval_acc(loader):
        model.eval()
        ok = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.cuda(), y.cuda()
                pred = model(x).argmax(1)
                ok += int((pred == y).sum().item())
                total += y.numel()
        return ok / max(total, 1)

    def eval_per_source():
        model.eval()
        per: dict[str, tuple[int, int]] = {}
        with torch.no_grad():
            for tag, idxs in src_indices.items():
                ok = total = 0
                for batch_start in range(0, len(idxs), 64):
                    batch = idxs[batch_start:batch_start + 64]
                    xs = torch.stack([_EVAL_TFM(dev_items[i][0])
                                       for i in batch]).cuda()
                    ys = torch.tensor([dev_items[i][1]
                                       for i in batch]).cuda()
                    pred = model(xs).argmax(1)
                    ok += int((pred == ys).sum().item())
                    total += ys.numel()
                per[tag] = (ok, total)
        return per

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
        acc = eval_acc(dev_dl)
        per = eval_per_source()
        per_str = "  ".join(f"{t}={ok}/{n}" for t, (ok, n) in sorted(per.items()))
        print(f"epoch {ep+1}/{args.epochs}  dev_acc={acc:.4f}  "
              f"({time.time()-t0:.1f}s)  per-source: {per_str}", flush=True)
        if acc > best:
            best = acc
            torch.save({
                "state_dict": model.state_dict(),
                "n_classes": N_CLASSES,
                "key_min": KEY_MIN, "key_max": KEY_MAX,
            }, args.output / "best.pt")
    print(f"\nbest dev_acc={best:.4f} -> {args.output / 'best.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
