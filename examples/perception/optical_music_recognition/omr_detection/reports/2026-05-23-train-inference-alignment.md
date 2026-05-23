# Detection-first OMR: train/inference distribution alignment

Two days of pipeline plumbing -- detect → transcribe → respell. The
quality number stayed the same as the legacy fixed-crop baseline, but
along the way three things tripped me up. Worth writing down so we
don't re-hit them on the next dataset.

## Status

End-to-end on 10 L7a dev pages, mode=page:

| Pipeline | Key acc | Pitch mean | Pitch median | Rhythm median |
|---|---|---|---|---|
| Raw VLM (safckylj, no respell) | — | 47.4% | 36.9% | 100% |
| v2 detector (re-rendered) + yolocrop CNN + detect-on-HF-via-tmp | 10/10 | 95.4% | 99.6% | 100% |
| v3 detector (HF-matched) + legacy fixed-crop CNN | 10/10 | 95.4% | 99.6% | 100% |
| v3 detector (HF-matched) + yolocrop CNN | 10/10 | 95.4% | 99.6% | 100% |

All three pipelines converge to the same number on L7a -- which means the
detection workstream on L7a is at parity with the legacy approach.
The win is structural, not numeric:

- **Mid-piece key changes** are now handled. Detected keysig boxes are
  per system; legacy fixed-crop only ever sees the top-left page.
- **Layout-robust**: detector finds the keysig wherever it is, not at
  a hard-coded page fraction.
- **No cross-distribution scaling** needed after v3.

## Architecture, end-to-end

```
HF page image  (or equivalent: any page rendered like HF)
   │
   ▼
YOLOv8n (5 classes: system, staff, barline_single, barline_heavy,
                    key_signature)
   │   ─→ systems, staves, barlines  ─→ cells.derive_cells / measures
   │   ─→ key_signature boxes
   ▼
SmallKeyCNN (98K params, trained on YOLO-detected crops)
   │
   ─→ per-detection key prediction ─→ majority vote
   │
   ▼
safckylj VLM (whole-page MXC2)
   │
   ▼
respell.respell_mxc2(pred, key)  ─→ corrected MXC2
```

## Three things that bit us

### 1. SVG `bar_numbers` over-counts by 1

For the cell-derivation sanity check I was using
`len(rec["bboxes"]["bar_numbers"])` as ground truth. Every page came back
"off by exactly one". The real measure count from MusicXML was 16; the
SVG-extracted bar numbers said 17.

LilyPond prints a bar number after the final barline that doesn't
correspond to a real measure. The SVG text-extractor caught it as part
of the largest consecutive cluster. **Use `n_measures` from MusicXML**;
SVG bar-number text is unreliable as ground truth.

### 2. Cross-rendering scaling is invalid

I trained the detector on our re-rendered PNGs and ran inference by
scaling boxes from re-rendered px to HF px with a single (sx, sy)
factor. Boxes landed on the wrong staves. Cause: the two renderings
are **separate runs of LilyPond with different settings** (bar-number
visibility, paper size, DPI), not the same image at different
resolutions. A pure scale won't reconcile them.

Two ways to fix it:

- **Apply YOLO directly on the HF image.** The detector trained on
  re-rendered images generalises to HF well enough that boxes land
  correctly. Verified visually. This is the temporary fix we shipped
  first (`eval_pipeline --detector-key` path).
- **Render training data to match HF exactly.** This is the durable
  fix: drop the bar-number injection, use `-dresolution=150`. After
  this change, re-rendered images are pixel-identical to HF, and
  YOLO has no distribution shift at inference. This is the v3 path.

### 3. The classifier must train on what the detector emits

Re-training the small CNN on **GT** keysig boxes scaled from
re-rendered to HF px capped at 58% dev accuracy. The scaled GT boxes
were themselves mis-positioned (consequence of #2), so the CNN saw
crops where the keysig glyph was at a randomly-shifted spatial position
relative to the staff.

Re-training the same CNN on crops produced by **running YOLO on HF
images** at training time (label = GT key from the MusicXML) hit 100%
dev accuracy in 12 epochs (98K params, 1.6s per epoch). Self-consistent.

Generalisation: in a detect→classify pipeline, **the classifier's
training crops should come from the same detector's outputs that it
will see at inference**, not from an independently-derived "ground
truth" that lives in a different image distribution.

### Generalised: principles

- Build training data through the **same code path** as inference. If
  inference is `detect_layout(image) → crop → cnn`, training crops
  should come from `detect_layout(image) → crop`. Don't substitute a
  geometric GT, because it can have a hidden alignment error.
- If two image distributions differ along ANY axis the model is
  sensitive to (DPI, render settings, text overlays, paper size), do
  not linearly scale annotations between them. Render once, match the
  target distribution, annotate in the target frame.
- A perfect mAP@0.5 on the *training* distribution says nothing about
  the *deployed* distribution.

## What changed (today's commits)

- `cells.derive_keysig_areas()` -- geometric definition of keysig area
  (top staff, first measure)
- `labels/extract_bboxes.py` -- populates `key_sigs`
- `labels/make_dataset.py` -- stores `key_sigs`, stores `n_measures`,
  defaults to `match_hf=True` rendering (no bar-number injection,
  PNG at -dresolution=150 matches HF synthetic-scores exactly)
- `train_detector.py` -- 5 classes; barline-width minimum widening
  preserved
- `pipeline.predict_keys_from_detections()` -- downward-only crop
  inflate so the staff sits in the top half of the crop, matching the
  legacy CNN's layout
- `eval_pipeline.py` -- `--detector-key` path, with a temp-file detour
  for the v2 detector that needed to see the HF image (no longer
  needed on v3)
- `train_keysig_yolocrop.py` -- new trainer that produces a CNN whose
  inputs match what the YOLO will emit at inference

## Generalization probe: L9 and openscore

Quick check on what the L7a-trained pipeline does on other levels.

```
sample size = 10 pages per config; mode=page; key-src = YOLO+CNN

L7a (in-distribution)                            10/10 keys correct
L9  (same engraving, harder content)             10/10 keys correct
openscore (real-world lieder, different engraving)  0/10 keys correct
```

Decomposed:

- **Detector itself generalises well to all three.** Visualising
  predicted boxes on L9 and openscore samples shows correctly-placed
  `key_signature` boxes on the top staff of every system, including a
  real hymn arrangement ("Just for Today", "Think of Today") with
  4 flats. The "find the leading area of the top staff" pattern is
  geometric and the YOLO learned it robustly.

- **CNN classifier does not generalise to openscore.** It hits 10/10
  on L9 (same engraving — generated by the same `synthetic_scores/
  generate.py` machinery, same staff size, same LilyPond version) but
  zero on openscore (different paper size, different staff size, real
  lyrics + title blocks, possibly different LilyPond version). Most
  openscore predictions cluster around `key=0`, suggesting the CNN
  doesn't recognise the sharp/flat glyphs at openscore's scale.

- **L10 isn't available** on the HF hub yet -- the
  `synthetic_scores/DESIGN.md` lists it as planned. We tested everything
  available (level1 … level9).

The fix is the same recipe as the keysig CNN itself: train the CNN on
crops produced by running YOLO on a **mix of L7a + L9 + openscore HF
images**, with keys from each source MusicXML. The detector already
gives the right crops; we just need a classifier trained on the union
distribution.

## Next

L7a doesn't exercise the new architecture's full value (no mid-piece
key changes, fixed layouts). The natural follow-ups:

1. **A dataset with mid-piece key changes** -- the only way to
   actually verify the per-system keysig path beats the legacy fixed
   crop.
2. **OpenScore lieder / quartets** -- real-world layouts with varied
   indents, instrument names, title blocks. The legacy fixed-crop CNN
   is brittle here; the detector should generalise.
3. **Per-cell transcription with a model retrained on cell crops** --
   `pipeline.transcribe_measure` is wired but uses safckylj, which is
   whole-page-trained. A retrained per-cell transcriber would let
   detection actually contribute to transcription quality, not just
   key correction.
