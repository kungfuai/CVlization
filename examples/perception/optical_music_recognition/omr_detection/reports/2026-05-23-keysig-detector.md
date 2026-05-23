# Option B: keysig detector + per-crop CNN — end-to-end on L7a

## Result

Wired Option B (detector adds `key_signature` class; existing-arch CNN
classifies each detected crop). On 10 L7a dev pages, mode=page:

```
pitched base     mean=47.4%  median=36.9%
pitched respell  mean=95.4%  median=99.6%
rhythm           mean=75.8%  median=100.0%
key accuracy:    10/10 (100%)
```

Parity with the legacy fixed-crop CNN on L7a (96.9% mean / 99.6% median
on 5 pages); now generalises to mid-piece key changes because the keysig
location is detected per system instead of assumed at the page's
top-left fraction.

## Two things bit us along the way

### 1. Cross-rendering scaling is invalid

The detector trains on our re-rendered PNGs (835×1181, bar-numbers
visible) and inference happens on HF PNGs (1240×1754, no bar numbers).
At first I scaled detected boxes from re-rendered px to HF px with a
single (sx, sy) factor. That landed boxes on the wrong staves —
the two renderings have **different layouts** (bar-number text shifts
content down), not the same image at different scales.

Fix: run YOLO directly on the HF image. The detector trained on the
re-rendered distribution generalises to HF well enough that the
predicted boxes are correctly placed on the HF image. Verified
visually.

### 2. CNN crops must be self-consistent with detection

Re-training the small CNN on GT-keysig-from-re-rendered crops scaled
to HF px gave only **58% dev accuracy** — the scaled GT crops were
themselves misplaced. Re-training on crops produced by YOLO running
on HF images directly (label = GT key from MusicXML) gives **100%**
dev accuracy.

Lesson: in a detect→classify pipeline, train the classifier on the
detector's outputs, not on independently-derived "ground truth"
boxes from a different image distribution.

## Pipeline

```
HF page image
  └── YOLO (cvlization/omr-detection:latest, 5 classes)
         system, staff, barline_single, barline_heavy, key_signature
  └── crop each key_signature box (inflate downward to ~3:1 aspect
       so staff sits in the top half, matching the legacy CNN's
       expected layout)
  └── SmallKeyCNN (98K params, trained per train_keysig_yolocrop.py)
         → fifths in {-4..+4}
  └── majority vote → page key
  └── safckylj VLM → raw MXC2
  └── respell_mxc2(pred, key) → corrected MXC2
```

## What changed

| File | Change |
|---|---|
| `cells.py` | `derive_keysig_areas()` — geometric definition for GT |
| `labels/extract_bboxes.py` | populate `key_sigs` |
| `labels/make_dataset.py` | store `key_sigs`, store `n_measures` (the SVG-text `bar_numbers` is unreliable) |
| `train_detector.py` | 5 classes, including `key_signature` |
| `pipeline.py` | `predict_keys_from_detections()` + downward-only crop inflate |
| `eval_pipeline.py` | `--detector-key` path, detect on HF directly |
| `train_keysig_yolocrop.py` | new: train CNN on YOLO-detected crops |

## Open

- Detector still trained on re-rendered PNGs. Generalising to HF works,
  but reduces our headroom. Worth re-rendering training PNGs without the
  bar-number injection so they match HF style exactly.
- L7a has no mid-piece key changes; the detector path doesn't gain over
  the legacy fixed crop here. The real test is data with mid-piece
  changes.
