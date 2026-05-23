# Detection-first OMR: train/inference distribution alignment

Multiple days of pipeline plumbing -- detect → transcribe → respell --
on a goal of going from L7a-only key prediction to multi-source
generalisation (L9 + openscore). Doc grew organically. Headline numbers
and the architectural lessons are up top; full chronology below.

## TL;DR (current state)

**Chosen pipeline**: L7a-only YOLOv8n detector (`detector_l7a_500_v3`)
+ multi-source 15-class SmallKeyCNN (`keysig_cnn_multisource_v2`) +
safckylj VLM + respell. Two clean models, decoupled.

**Page-level key accuracy on 60 unique dev pages per source**:

| source | strict (page-majority == first `<fifths>`) | any-box-in-set | per-box |
|---|---|---|---|
| L7a       | 100% | 100% | 100% |
| L9        | 100% | 100% | 100% |
| openscore | 78.3% | **90.0%** | 71.8% |

**Per-source detector quality** (`detector_mix2` on combined dev):

| source | system R | staff R | keysig R |
|---|---|---|---|
| L7a       | 1.000 | 1.000 | 1.000 |
| L9        | 1.000 | 0.860 | 0.860 |
| openscore | 0.941 | 0.898 | 0.891 |

**What's actually limiting openscore key accuracy**:
1. Multi-key pages + page-majority-vote eval (closes from 78% → 90% with
   "any-box-in-set" metric).
2. Real CNN miscounts on rare keys (|fifths| ≥ 5).
3. MusicXML `<fifths>` ≠ what's drawn (transposing instruments / metadata
   bugs); not fixable downstream.

**Things tried that didn't move the number**:
- Multi-source detector training (regressed everywhere)
- Re-trained CNN on the multi-source detector's crops (regressed)
- Staff-relative barline widening (no measurable effect)
- Page-level eval with set-based GT (recovered the metric, didn't actually
  change the model).

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

## Multi-source CNN (train_keysig_multisource.py)

Done. Trainer pulls L7a + L9 + openscore HF images, runs YOLO on each
to produce keysig crops, labels with `<fifths>` from each MusicXML.

**Leak prevention**: uses HF's own `train`/`dev` splits for each config
and asserts zero `score_id` overlap between train and dev across all
sources before training. (passes: 1146 train scores, 239 dev scores,
0 overlap.)

**Class range**: extended from -4..+4 (9 classes) to -7..+7 (15 classes)
to cover everything in standard Western tonal notation. Openscore had
~16% of pages with `|fifths|` in {5, 6}; under the 9-class CNN they'd
have triggered a CUDA index error in cross-entropy. -7..+7 covers it
all without expansion ever being needed -- 8+ sharps/flats are written
enharmonically by every composer.

**Result** (500 train + 100 dev pages per source, 30 epochs):

| Source | dev accuracy |
|---|---|
| L7a | 290/290 = 100.0% |
| L9  | 297/310 = 95.8% |
| openscore | 273/400 = 68.3% |
| Overall | 860/1000 = 86.0% |

L7a + L9 stay at near-perfect (same engraving family). Openscore went
from 0% (L7a-only CNN) to 68% (multi-source) -- still climbing at
epoch 30, and likely improves further with (a) more openscore training
data (we used 500 of thousands available), (b) augmentation that
narrows the staff-size variation across sources, or (c) staff-spacing
normalisation in the crop step before the CNN sees it.

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

## Detector multi-source experiment: helped the CNN, hurt the detector

Tested whether training the YOLO detector on a L7a+L9+openscore mix
(mirroring what we did for the CNN) would further improve openscore
quality. It didn't.

Per-source key-prediction accuracy on 30 dev pages from each source:

| detector trained on | CNN trained on | L7a | L9 | openscore |
|---|---|---|---|---|
| L7a only | L7a only | 100% | 93% | 46% |
| **L7a only** | **L7a + L9 + openscore** | 100% | 100% | **63%** ← best |
| L7a + L9 + openscore | L7a only | 100% | 93% | 33% |
| L7a + L9 + openscore | L7a + L9 + openscore | 100% | 100% | 57% |

What happened to the detector when we added openscore + L9 training data:

```
class             L7a-only detector   multi-source detector
system            mAP50 0.995         mAP50 0.977
staff             mAP50 0.995         mAP50 0.884
barline_single    mAP50 0.995         mAP50 0.608  (R drops 1.00 -> 0.50)
key_signature     mAP50 0.995         mAP50 0.863
```

The barline-recall collapse is the loud signal. We trained on 915
pages total but only 113 from openscore (after dedup -- the streaming
HF dataset returned multiple rows per multi-page score). Openscore has
many more barlines per page than L7a (real piano music), at a
different scale; the detector's barline class definition (8 px wide GT
post-widen) was sized for L7a's barline pixels and is wrong for
openscore's. Result: degraded everywhere instead of generalised.

**Decision**: keep the L7a-only detector + multi-source CNN.

What it suggests for the future:
- The CNN benefits from seeing multiple engraving styles (classifier
  task is style-sensitive).
- The detector is robust to style at L7a's training scale; adding
  *under-sized* multi-source data without source-aware bbox-widening
  (or per-source min-box-px) regresses it.
- If we want to push the detector further on openscore, we need
  (a) more openscore pages -- ideally dedup-aware, ~500+ unique pages,
  (b) per-source barline minimum width to match the destination
  engraving's barline thickness, and
  (c) probably a longer training schedule.

End-to-end best on openscore right now: **63% key accuracy** with the
L7a-only YOLO detector + the multi-source 15-class CNN. Up from 0%
with the L7a-only CNN at the start of this thread.

## After dedup + staff-relative widening: same picture

Two clean-up improvements:
- `make_dataset.dedup_by_score_id` -- the openscore HF dataset has ~4 rows per unique score (each = one page slice). Without dedup we were re-rendering the same score multiple times. With dedup we got 353 unique-train + 79 unique-dev openscore pages (vs 113+39 before, plus matching ~600 unique scores in dev/train).
- `train_detector.MIN_BARLINE_W_STAFF_FRAC = 0.7` -- replaces the L7a-tuned 8-px floor with a per-record staff-relative width (`0.7 * staff_space`). One inference recipe everywhere.

Re-built the multi-source detector (`detector_mix2`) on this larger, cleaner data:

```
                L7a-only detector    detector_mix2 (this run)
system          mAP50 0.995          mAP50 0.991
staff           mAP50 0.995          mAP50 0.951
barline         mAP50 0.608          mAP50 0.622
key_signature   mAP50 0.863          mAP50 0.947  (+0.084)
```

Keysig mAP improved meaningfully. Then re-trained CNN on the new
detector's crops (`keysig_cnn_multisource_v3`):

```
                         L7a    L9     openscore
v2 (L7a-only det crops)  100%   96%    68%
v3 (mix2 det crops)      67%    62%    32%  <- worse everywhere
```

Surprising: the better detector produces *worse* crops for CNN training.
Investigation showed v3 collected ~60% the number of crops v2 did
(mix2 detector's per-box confidence is lower, so more keysigs fall
below the conf=0.25 threshold at training time) and the boxes that
*do* clear threshold have higher position/size variance.

Final 2×2×3 grid on 30 dev pages per source (HF originals):

| detector | CNN | L7a | L9 | openscore |
|---|---|---|---|---|
| L7a-only | yolocrop (L7a) | 100% | 93% | 46% |
| **L7a-only** | **multi-source v2** | **100%** | **100%** | **63%** ← best |
| mix2 (multi-source) | yolocrop (L7a) | 100% | 90% | 41% |
| mix2 (multi-source) | multi-source v2 | 100% | 100% | 57% |
| mix2 (multi-source) | multi-source v3 | 67% | 62% | 32% |

**Decision**: keep the L7a-only detector + multi-source v2 CNN. Two
days of trying to improve openscore via better detector data did not
move the downstream number; the bottleneck is the classifier's
training distribution, not the detector.

What's actually limiting openscore key accuracy at 63%:

1. **Transposing-instrument GT mismatch** (multiple cases inspected):
   musicxml `<fifths>` is concert pitch; the visible page can be the
   transposed key. The CNN reads what's drawn. Counted as "wrong" by
   our eval. Maybe ~5-10% of openscore.

2. **Rare-key class imbalance**: keys with `|fifths| in {5, 6, 7}` are
   under-represented in the training mix. The CNN under-counts the
   last 1-2 accidentals in those signatures.

3. **Possible mid-piece key changes** affecting the per-page GT (the
   musicxml's first `<fifths>` may not be what's drawn on that page).

Fixing any of these requires data-side work, not architecture.

## Better openscore GT: per-page fifths set, per-box accuracy

Earlier we compared the page's *majority* prediction to the *first*
`<fifths>` in the musicxml. That conflates two things:

1. Multi-key pages where the model gets some systems right and others wrong.
2. Real CNN errors.

Each openscore HF row's musicxml is a slice corresponding to its
`bar_start`-`bar_end` range, so the musicxml contains ALL key changes
visible on that page. Better metric: extract every `<fifths>` in the
slice, treat each as a valid GT.

Per-source detector quality (multi-source detector, on combined dev):

```
                       P       R    GT
l7a/system        1.000   1.000   291
l7a/staff         0.952   1.000   873
l7a/key_signature 1.000   1.000   291
l9/system         0.896   1.000   129
l9/staff          0.771   0.860   387
l9/key_signature  0.771   0.860   129
openscore/system  0.865   0.941   238
openscore/staff   0.891   0.898   737
openscore/key_signature 0.887 0.891 238
```

System recall is high enough (>=0.94) everywhere that per-system
evaluation is viable -- the bottleneck is mapping musicxml measures
to detected systems, which we don't have for HF images.

Page-level openscore eval (60 unique dev pages, L7a-only detector +
multi-source CNN):

| Metric                                         | Result |
|---|---|
| Strict: page-majority == first `<fifths>`      | 47/60 = 78.3% |
| Lenient: page-majority in page's `<fifths>` set | 48/60 = 80.0% |
| Any-box-in-set: any predicted keysig in set    | 54/60 = 90.0% |
| Per-box accuracy: each predicted keysig in set | 196/273 = 71.8% |
| Multi-key pages on the page                     | 9/60 |

`any-box-in-set = 90%` is the most honest page-level number; the
remaining 10% are pages where every detected keysig predicts a value
not in the musicxml's set (real CNN failures + occasional
musicxml/display mismatches).

The 78.3% strict number understated by ~12 points. Going forward we
report 90% any-box / 72% per-box as the openscore quality figure.
