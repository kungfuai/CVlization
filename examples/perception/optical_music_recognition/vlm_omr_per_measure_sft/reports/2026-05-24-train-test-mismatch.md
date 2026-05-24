# Train/test mismatch on per-measure crops

## The bug

The per-measure dataset uses **GT bboxes** (extract_bboxes on rendered
SVG = perfect barline coverage). Inference uses **YOLO detector
bboxes** (~50% barline recall measured earlier). `cells.derive_measures`
clusters cells around barlines; with sparse barlines it produces
either giant boxes (covering many measures) or no boxes at all.

Inference-time measure derivation on 9 dev HF pages:

| source | page | detected measures (expected) |
|---|---|---|
| L7a | L7a_04000 | **5** (expected ~16) -- only top system found |
| L7a | L7a_04001 | 6 |
| L7a | L7a_04002 | 22 (over-segmented) |
| L9  | L9_03115  | **1** measure on a full page |
| L9  | L9_03116  | 3 |
| L9  | L9_03117  | 2 |
| openscore | lc6211535 | **0** measures detected |
| openscore | lc6486038 | 1 |
| openscore | lc5079512 | 15 |

Visual inspection (L7a_04000): the page has 3 systems × ~5 measures
each. Detector finds the 5 measures on the TOP system; systems 2-3 are
empty of red bboxes. The visible barlines on systems 2-3 just weren't
picked up by YOLO at conf=0.25.

A few measure crop sizes from L9_03116: `251x50`, `832x50`, `1011x149`.
The 1011x50 box is most of a staff line, not a measure.

## What this means

The "100% L7a per-measure" eval result was on **GT-cropped cells from
the rendered training data**, not on detector-derived cells from HF
images. The end-to-end pipeline (HF image → YOLO → measure crops →
per-measure VLM → stitch → respell) would transcribe ~30% of measures
at best on L7a, and ~6% on openscore.

This is the same "classifier must train on what the detector emits"
lesson we learned in the detection workstream's keysig CNN. We
violated it here.

## Fixes (ranked by leverage)

1. **Train per-measure VLM on detector-derived crops, not GT crops.**
   Build the dataset by running multi-task YOLO on each train image
   and using its measure outputs as crops; label with the matching
   MXC2 slice (figured out via x-overlap with the GT measure boxes).
   This mirrors what we did to make the keysig CNN work.

2. **Fix barline detection.** Train the YOLO with a heavier barline
   weight or richer barline data. We know the staff-relative widening
   we already added doesn't fully solve it -- barline recall is ~50%.

3. **Train a direct `measure` YOLO class.** Instead of deriving
   measures from staves x barlines, detect measure rectangles
   directly. Eliminates the derivation step's fragility. Costs
   training a new class.

4. **Detection-free fallback at inference.** If detector returns < N
   measures for a page, fall back to whole-page VLM. Pragmatic
   safety net.

(1) is the right structural fix. (3) is the cleanest but requires
substantial new data.

## Addendum: the real per-page bottleneck is system detection, not barlines

User pushed back on my "barline recall ~50% so measures don't derive"
story. Actual numbers from L7a_04000 at varying confidence thresholds:

```
L7a_04000 (expected 3 systems, ~17 measures):
  conf=0.50  system=1   staff=9   barline=45   measures=5
  conf=0.25  system=1   staff=9   barline=45   measures=5
  conf=0.10  system=4   staff=9   barline=45   measures=32
  conf=0.05  system=6   staff=9   barline=46   measures=51
  conf=0.01  system=9   staff=9   barline=51   measures=72
```

Barline recall on this page is 45/50 = 90% (not 50%). Staff count is
9/9 = perfect. **The bottleneck on this page is system detection**:
at conf=0.25 only the most confident system is found, so
`derive_measures` only assigns barlines to that one system. Lower
confidence over-detects (multiple boxes per real system).

So there's **no good system-class threshold for L7a HF originals**.
The detector's aggregate system R=1.0 on val was on rendered val
images, not HF originals at conf=0.25.

The aggregate barline R=0.50 also holds -- per-class mAP across all
sources averaged 50% recall. On L7a specifically barlines are fine
(90%); it's openscore + L9 that drag the aggregate down.

### Implications for the fix

The right next move is no longer just "rebuild data on detector crops".
We need a measure crop that's robust to detector failures. Options:

1. **Drop the `system` class entirely; group staves into systems
   geometrically** (cluster by y). Pure geometry, no train/test gap.
2. **Detect measure boxes directly** as a YOLO class (not derive from
   system x barlines). Reduces the brittle multi-class dependency to
   a single one.
3. **Per-class conf thresholds**: keep keysig at conf=0.25 (where
   it's already strong), drop system to conf=0.10 with explicit NMS
   to dedup overlapping system predictions.

(1) is the simplest. Staves cluster into systems trivially: sort staves
by y, split at any gap > N staff-spaces. No model needed.
