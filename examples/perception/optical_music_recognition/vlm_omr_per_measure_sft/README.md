# vlm_omr_per_measure_sft

Per-measure SFT for OMR transcription. Sibling of `vlm_omr_sft` (whole-page)
and `omr_detection` (layout detection).

## Why this exists

`safckylj` (Qwen3-VL-8B SFT on L7a-style whole pages) achieves ~95% pitched
on L7a but ~2% pitched on openscore (measured 2026-05-23 -- the model
hallucinates note-by-note on real-world engraving). The whole-page approach
is also expensive at inference (~75 s/page) and the model has to learn
layout + transcription jointly.

The `omr_detection` workstream now reliably detects:
- systems, staves, barlines, key signatures (multi-task YOLOv8n)
- cells (staff × measure derived from staves + barlines)

This unlocks the per-measure decomposition: crop each measure cell, feed it
to a transcriber that's been trained on measure-sized inputs only.

## What this trains

A VLM (initially extending `safckylj`, optionally a smaller backbone later)
that maps **one cell image → that measure's MXC2 slice**. Each training
sample is `(cell_crop, measure_mxc2)` where `measure_mxc2` comes from
`mxc2_slice.iter_measures(full_page_mxc2)` -- already a self-contained
slice with restated key/clef/time and explicit voice/staff tags.

Cross-measure elements (ties, slurs, beams crossing barlines, wedges)
are handled by MXC2's per-note endpoint tokens: a measure ending with a
tied note emits `tie=start`; the next measure's first note emits
`tie=stop`. Stitching at inference time is just concatenation. See
`reports/2026-05-23-design.md` for the cross-measure analysis.

## Pipeline at inference

```
HF page image
  ▼
multi-task YOLO (from omr_detection)
  ▼  systems / staves / barlines / keysig boxes
cells.derive_cells  ─→ list[Cell]
  ▼
for each cell: crop page image, run per-measure VLM ─→ measure MXC2
  ▼
stitch (concat per part, per measure, in reading order) ─→ page MXC2
  ▼
respell with detector-predicted key ─→ corrected MXC2
```

## Folder layout

```
vlm_omr_per_measure_sft/
├── README.md                      this file
├── Dockerfile                     extends cvlization/vlm-omr-sft + ultralytics
├── build.sh / train.sh
├── labels/
│   └── build_per_measure_dataset.py   detection cells + MXC2 slices -> HF dataset
├── train_per_measure.py           SFT off safckylj at the per-measure level
├── eval_per_measure.py            per-measure + stitched-page metrics
├── stitch.py                      naive concat of per-measure outputs
└── reports/
    └── 2026-05-23-design.md       cross-measure handling write-up
```

## Status

Scaffolding only. Dataset builder + training next.

## What's reused

From `vlm_omr_sft/`:
- `mxc2.py` -- whole-page → MXC2 encoder
- `mxc2_slice.py` -- per-measure slicer (the key utility for this work)
- `eval_mxc.py` -- evaluation harness
- `respell.py` -- post-hoc key correction
- `train.py` -- training loop patterns (prepare_inference_inputs, INSTRUCTION_MXC2)

From `omr_detection/`:
- `detector_mt` (multi-task YOLO) -- gives us cells at inference
- `cells.derive_cells` / `derive_measures` -- cell geometry

## Decisions to make as we go

- **Initial base model**: start with safckylj (Qwen3-VL-8B). Per-measure
  SFT extends what's already trained, fastest to validate viability.
  Smaller backbones (Qwen2.5-VL 3B, or a non-VLM seq2seq) can swap in
  later if inference latency matters.
- **Per-cell vs per-measure**: detection emits both. Per-cell (staff ×
  measure) has the cleanest MXC2 slice via `mxc2_slice`. Per-measure
  (whole system × measure) gives the VLM cross-staff context (chord
  voicings spanning treble + bass) at the cost of being multi-staff.
  Start with per-measure (whole-system slab); the slicer handles
  multi-staff correctly.
- **Cross-measure tokens**: emit partial endpoints per measure (the
  natural MXC2 behaviour); stitcher concatenates.
- **MXC2 extensions** (ottava, etc.): defer until we measure whether
  it matters on openscore.
