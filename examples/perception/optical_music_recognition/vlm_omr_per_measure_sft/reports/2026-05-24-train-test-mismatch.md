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
