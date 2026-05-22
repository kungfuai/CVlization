# YOLOv8n on L7a layout detection — baseline

## Result

Detector hits mAP50 = 0.995, recall = 1.000 across all three present classes
after 50 epochs on 500 train / 100 val L7a pages at imgsz=1280.

```
all      P=0.999  R=1.000  mAP50=0.995  mAP50-95=0.992
system   P=0.998  R=1.000  mAP50=0.995
staff    P=1.000  R=1.000  mAP50=0.995
barline_single  P=1.000  R=1.000  mAP50=0.995
barline_heavy   (no instances on L7a)
```

Training wall clock: ~3.5 min on RTX PRO 6000 Blackwell.

## What initially didn't work

First run hit P=0.999 / R=0.667 / mAP50=0.663. Per-class breakdown:
- system, staff: near-perfect
- barline_single: **R=0.000, mAP50=0.000**

Root cause: raw barlines are ~1.3 px wide × ~28 px tall at 1280 imgsz.
After YOLO's 8× stride at the first detection head, that is ~0.2 px wide
in feature space — sub-pixel. The detector cannot localize what its
feature map cannot resolve.

## Fix

Inflate barline boxes to a minimum width of 8 px (around the x-center)
during YOLO label conversion. Downstream cell derivation only needs the
x-center of each barline, so widening the GT box is lossless.

See `train_detector.MIN_BARLINE_W_PX` and the conditional in
`_record_to_yolo_lines`.

## Open

- L7a has no heavy/final barlines, so class 3 was never tested. Mid-piece
  or end-of-piece barlines in real corpora will need this verified.
- Per-system staff index (top/middle/bottom) is encoded in the JSONL but
  collapsed into a single `staff` class for YOLO. That's fine for cell
  derivation (we order detected staves top-to-bottom anyway) but worth
  noting if we later need treble vs bass detection directly.
- Next step: build cell derivation (staff ∩ barline gaps) and wire into
  pipeline.py.
