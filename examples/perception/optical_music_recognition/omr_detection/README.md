# OMR Detection — decomposed pipeline (2026-05-22)

Sibling to `vlm_omr_sft/`. This folder hosts the **detection-first** approach to
OMR: detect layout (systems / staves / measures / key-sigs) → crop each cell
→ run a focused per-cell transcription model → stitch and reconcile.

Motivation: `vlm_omr_sft/` proved a monolithic whole-page VLM can be made to
work on L7a (via the [respell pipeline](../vlm_omr_sft/reports/2026-05-21-respell-pipeline.md)),
but the architecture is brittle to mid-piece key changes, varied real-data
layouts, and long pages. Classical OMR systems (Audiveris, Oemer) use a
decomposed pipeline because each stage is small, verifiable, and composable.

## Pipeline overview

```
page image
   ▼
DETECTOR ──> { system_bbox, staff_bbox, barline_x, key_sig_bbox?, clef_bbox? }
   ▼
derive cells: (staff, measure) crops + the key-sig active at that measure
   ▼
per-cell transcriber: small VLM or seq2seq on a measure × staff crop
   ▼
stitch + reconcile (key per cell → respell key-implied accidentals)
   ▼
full-page MXC2
```

## Hierarchy (conventions)

```
page
└── system           horizontal "line of music"; ordered top → bottom
    ├── staff        5-line staff for one instrument/hand; ordered top → bottom within system
    │   ├── clef        (read once; determines treble/bass/etc.)
    │   ├── key_sig     (at start of staff; re-engraved every system)
    │   └── measures
    │       └── notes   (the transcription unit = staff × measure = "cell")
    └── ...
```

### Treble vs bass — not a detection problem

The detector emits **staff** bboxes only. Which is treble vs bass is determined
by the **clef glyph** at the start of each staff. Either:
- classify clef separately (small CNN), or
- let the per-cell transcription model see the clef and emit it.

### Voice (polyphony within one staff) — not a detection problem

A staff with two voices looks like one staff visually. Voices are distinguished
by **stem direction** and beaming. Detection finds the staff; per-cell
transcription handles voice separation.

### Part identity (Voice vs Piano-RH vs Piano-LH) — order-based

The top staff in each system is part 1, next is part 2, etc. Works for synthetic
L7a and most lieder where staff count per system is constant. For mixed-staff-
count scores (orchestra with `\RemoveEmptyStaves`), use the source MusicXML's
part-staff mapping.

## Detection classes (start small, extend later)

| class | priority | notes |
|---|---|---|
| `system` | ★ required | y-band; derived from staff groupings |
| `staff` | ★ required | the 5-line region; one per part per system |
| `barline` | ★ required | vertical line; defines measure boundaries within a system |
| `key_signature` | optional | crop classifier already exists in `models/keysig_cnn.py` |
| `clef` | optional | small, but trivial to add |
| `time_signature` | optional | rarely needed at this layer |

Cells are **derived**, not detected: `cell = staff_bbox ∩ (between two consecutive barlines)`.

## Training data — free from synthetic SVGs

For every synthetic page, LilyPond emits SVG with full glyph coordinates.
`datasets/omr/pipeline.py` already extracts bar-number text positions from
these SVGs for the page-slicing pipeline.

Extending the same parser to emit per-element bboxes (staff lines via grouped
horizontal paths, barlines as vertical paths, key-sig glyphs as recognisable
path patterns) gives us labelled detection data at the scale of the whole
synthetic corpus.

For openscore, the same SVG → bbox pipeline applies. The existing pipeline
already proves this works on real-data SVGs.

## File layout

```
omr_detection/
├── README.md                    this file
├── labels/
│   ├── extract_bboxes.py        SVG → bboxes (per-page)
│   └── make_dataset.py          builds HF detection dataset
├── models/
│   ├── layout_detector.py       multi-class detector (YOLO/DETR-style)
│   └── keysig_cnn.py            small CNN, ported from vlm_omr_sft
├── configs/
│   └── detector_l7a.yaml        starter training config
├── train_detector.py            single-page detection training
├── eval_detector.py             mAP, per-class IoU on dev
├── pipeline.py                  orchestrator: detect → crop → transcribe → stitch
└── reports/                     experiment logs
```

## Build order

1. **`labels/extract_bboxes.py`** — get bboxes out of LilyPond SVGs.
   Concrete goal: for one L7a sample, output `{systems, staves, barlines}`
   bboxes that visually match the rendered page.
2. **`labels/make_dataset.py`** — build a Hugging Face detection dataset from
   step 1. Coverage: synthetic L7a + L9 first, openscore lieder later.
3. **`models/layout_detector.py`** — start with a small YOLO-style model
   (Ultralytics or our own minimal head). Train on synthetic.
4. **`eval_detector.py`** — mAP and per-class IoU on a held-out dev set.
5. **`pipeline.py`** — orchestrate detector + (existing) safckylj transcription
   + per-cell respell. Validate on L7a dev first, then extend.

Optional stages:
- Per-measure-per-staff focused transcription model (instead of safckylj).
- Detection of key-sig + clef glyphs directly.

## Reused from `vlm_omr_sft/`

- `respell.py` — deterministic accidental application (post-process).
- `mxc2_slice.py` — stateless MXC2 measure slicer.
- `mxc2_normalize.py` — musical-equivalence normalizer.
- `train_keyclf_cnn.py` — small CNN architecture for key classification;
  ported to `models/keysig_cnn.py`.

## Out of scope (for now)

- Real-time inference / deployment.
- Generation of MusicXML from scratch (we use MXC2 as the target).
- Pixel-perfect re-rendering.
- Mid-system key changes — handled later via per-measure key-sig detection
  (after the basic per-system case works).
