# Experiments on improving system + barline detection

## Background

The multi-task `detector_mt` (yolov8n, 1280) shipped these per-source
recalls on the val set:

|         | L7a   | L9    | openscore |
|---------|-------|-------|-----------|
| system  | 1.00  | 1.00  | 0.92      |
| staff   | 1.00  | 0.86  | 0.88      |
| barline | 0.93  | **0.12** | **0.32** |
| keysig  | 1.00  | 0.97  | 0.90      |

Two specific failure modes on HF inference (vs the cleaner val):
1. `system` class at conf=0.25 only finds 1 of 3 systems on L7a HF
   pages (val showed R=1.0). Conf sweep showed no good threshold.
2. Barline recall is catastrophic on L9 (12%) and weak on openscore
   (32%). On val L7a it's fine (93%).

## Experiment 1: geometric system grouping

Replace the YOLO `system` class with `cells.group_staves_into_systems(
staves)` -- pure y-clustering with an adaptive threshold of
`max(1.0 * avg_staff_height, median_gap + 0.5 * avg_staff_height)`.

Measure derivation on 9 HF dev pages, detected-system vs geometric:

| page             | sys_det | sys_geom | meas_det | meas_geom |
|------------------|---------|----------|----------|-----------|
| L7a_04000        |   1     |   3 ✓    |   5      |  14       |
| L7a_04001        |   1     |   3 ✓    |   6      |  16       |
| L9_03116         |   2     |   4 ✓    |   3      |   8       |
| openscore lc6211535 | 0    |   3 ✓    |   0      |  15       |

Verdict: clear win on L7a + L9 (where staves are detected well). On
openscore where many staves are themselves missed, geometric grouping
can only do so much.

## Experiment 2: yolov8s + imgsz 1920

Bigger backbone + higher input resolution. Train recipe identical to
`detector_mt` otherwise (multi-task, 19 classes, same data, 50 epochs).

Per-class recall change (vs yolov8n@1280):

| metric              | n@1280 | s@1920 |  delta |
|---------------------|--------|--------|--------|
| L7a barline R       | 0.934  | 1.000  | +0.066 |
| L9 barline R        | 0.118  | 0.117  |  ~0    |
| openscore barline R | 0.315  | 0.304  | -0.011 |
| L9 staff R          | 0.855  | 0.809  | -0.046 |
| openscore staff R   | 0.882  | 0.825  | -0.057 |

Verdict: more capacity doesn't fix the hard cases. L7a got a small
lift (already near-perfect). L9 + openscore barline recall is
unchanged; staff recall slightly worse (more capacity over-fits to
common patterns at the expense of rare ones).

Inference visualisations show v8s over-detects staves on some
openscore pages: lc6211535 went from 7 → 37 staff predictions
(page actually has ~18). Net measure derivation got worse from
over-segmentation.

## Diagnosis

The bottleneck on L9 + openscore is **training data**, not model
capacity:
- L9 has 302 training pages with denser, more rhythmic content. The
  detector hasn't seen enough L9-style pages to recognise barlines
  in dense clusters.
- Openscore has only 113 unique training pages (after dedup); the
  remaining ~thousands of openscore pages in the HF dataset are
  unused.
- Our re-rendered openscore training images don't perfectly match
  HF openscore originals (different paper size / staff size /
  font), creating a train/inference distribution mismatch.

## Recommended next steps (data-side)

1. **More openscore training data**. Stream further into the
   `zzsi/openscore` dataset; ~500+ unique scores instead of 150.
2. **Match openscore rendering style more closely**. Our match-HF
   render uses 150 DPI; openscore HF appears to use ~101 DPI
   (smaller pages). Try rendering at 101 DPI for openscore-source
   pages.
3. **Use HF originals as detector training input directly**,
   bypassing re-render. Either derive bboxes by running detector
   in self-supervised fashion (and human-correct) or render only
   the SVG and use the bboxes there + HF original as image.

The single biggest leverage point right now: **more openscore data**.
Bigger model on under-sampled data doesn't help.

## Decisions

- Adopt geometric system grouping in `cells.group_staves_into_systems`;
  drop reliance on the YOLO `system` class for downstream
  measure derivation. (Already merged.)
- Keep yolov8n@1280 as the production checkpoint. The yolov8s@1920
  experiment is not pulled forward.
