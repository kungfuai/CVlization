#!/usr/bin/env python3
"""Train a small CNN for image → key=N classification on a fixed crop of
the key-signature region.

By cropping ONLY the key-sig region we make the model physically incapable
of learning shortcuts from note content. The classifier can only see the
clef + accidentals at the start of the first staff.

Output: 9-class softmax over fifths ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4}.

Usage:
    python train_keyclf_cnn.py \\
        --configs level7a level5 level6 level8 level9 \\
        --output outputs/keyclf_cnn --epochs 10
"""
import argparse
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image


# 9-class: fifths in [-4, +4]
N_CLASSES = 9


def fifths_to_label(f: int) -> int:
    """fifths in [-4, +4] -> label in [0, 8]."""
    return f + 4


def label_to_fifths(lab: int) -> int:
    return lab - 4


# Crop fractions: top-left region capturing the first staff's clef + key sig
CROP_LEFT_FRAC = 0.05
CROP_TOP_FRAC = 0.04
CROP_RIGHT_FRAC = 0.30
CROP_BOTTOM_FRAC = 0.10


def crop_keysig(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((int(w * CROP_LEFT_FRAC), int(h * CROP_TOP_FRAC),
                     int(w * CROP_RIGHT_FRAC), int(h * CROP_BOTTOM_FRAC)))


# Resize crops to a fixed size for the CNN
INPUT_W, INPUT_H = 256, 96


_NORMALIZE = transforms.Normalize(mean=[0.5], std=[0.5])

_TRAIN_TFM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((INPUT_H, INPUT_W)),
    transforms.ToTensor(),
    _NORMALIZE,
])
_EVAL_TFM = _TRAIN_TFM


class KeySigDataset(Dataset):
    def __init__(self, hf_dataset, tfm, use_crop=True):
        self.ds = hf_dataset
        self.tfm = tfm
        self.use_crop = use_crop

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        sample = self.ds[i]
        img = sample["image"].convert("RGB")
        if self.use_crop:
            img = crop_keysig(img)
        x = self.tfm(img)
        m = re.search(r"<fifths>(-?\d+)</fifths>", sample["musicxml"] or "")
        fifths = int(m.group(1)) if m else 0
        return x, fifths_to_label(fifths)


class SmallKeyCNN(nn.Module):
    """4-conv-block CNN, ~70K params. Plenty for a 9-class glyph counting task."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="zzsi/synthetic-scores")
    ap.add_argument("--configs", nargs="+",
                    default=["level7a", "level5", "level6", "level8", "level9"])
    ap.add_argument("--output", default="outputs/keyclf_cnn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-crop", action="store_true",
                    help="Use the full image (probes spurious-correlation risk)")
    ap.add_argument("--input-w", type=int, default=None)
    ap.add_argument("--input-h", type=int, default=None)
    args = ap.parse_args()

    if args.no_crop:
        # For full-image, use a larger resize that preserves layout but
        # fits in memory. 384×512 ≈ 200K pixels — small CNN can still process.
        global INPUT_W, INPUT_H, _TRAIN_TFM, _EVAL_TFM
        INPUT_W = args.input_w or 384
        INPUT_H = args.input_h or 512
        _TRAIN_TFM = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((INPUT_H, INPUT_W)),
            transforms.ToTensor(),
            _NORMALIZE,
        ])
        _EVAL_TFM = _TRAIN_TFM
        print(f"  no-crop mode: resizing full image to {INPUT_H}×{INPUT_W}")

    os.makedirs(args.output, exist_ok=True)

    from datasets import load_dataset
    print(f"Loading {args.configs} ...")
    train_parts = []
    for cfg in args.configs:
        ds = load_dataset(args.repo, cfg, split="train")
        keep = ["image", "musicxml", "score_id"]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        print(f"  {cfg}: {len(ds)} samples")
        train_parts.append(KeySigDataset(ds, _TRAIN_TFM, use_crop=not args.no_crop))
    train_ds = ConcatDataset(train_parts)
    print(f"  total: {len(train_ds)} samples")

    # Dev: just level7a for now (consistent with our other evals)
    dev_hf = load_dataset(args.repo, "level7a", split="dev")
    dev_ds = KeySigDataset(dev_hf, _EVAL_TFM, use_crop=not args.no_crop)
    print(f"  dev: {len(dev_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SmallKeyCNN(N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        n = 0
        loss_sum = 0.0
        correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n += y.size(0)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        train_loss = loss_sum / n
        train_acc = correct / n
        sched.step()

        # Dev eval
        model.eval()
        correct = 0
        n_dev = 0
        per_key_correct = {}
        per_key_total = {}
        with torch.no_grad():
            for x, y in dev_loader:
                x = x.to(device)
                logits = model(x)
                pred = logits.argmax(1).cpu()
                for i in range(y.size(0)):
                    true_f = label_to_fifths(int(y[i]))
                    pred_f = label_to_fifths(int(pred[i]))
                    n_dev += 1
                    per_key_total[true_f] = per_key_total.get(true_f, 0) + 1
                    if true_f == pred_f:
                        correct += 1
                        per_key_correct[true_f] = (
                            per_key_correct.get(true_f, 0) + 1
                        )
        dev_acc = correct / n_dev
        elapsed = time.time() - t0
        print(f"epoch {epoch+1:>2}: train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.1%}  dev_acc={dev_acc:.1%}  "
              f"({elapsed:.0f}s)")

        if dev_acc > best_acc:
            best_acc = dev_acc
            ckpt_path = os.path.join(args.output, "best.pt")
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "dev_acc": dev_acc,
                "per_key": {k: per_key_correct.get(k, 0) / v
                            for k, v in per_key_total.items()},
            }, ckpt_path)
            print(f"  saved best ({best_acc:.1%}) to {ckpt_path}")
            print(f"  per-key:")
            for k in sorted(per_key_total):
                acc = per_key_correct.get(k, 0) / per_key_total[k]
                print(f"    {k:>+3d}  n={per_key_total[k]:>3}  acc={acc:.1%}")

    # Save final too
    torch.save({"state_dict": model.state_dict()},
               os.path.join(args.output, "final.pt"))
    print(f"\nbest dev_acc: {best_acc:.1%}")


if __name__ == "__main__":
    main()
