# 2x2 ablation: training-time padding × inference-time upsampling

Goal: isolate causal effects of two ideas for closing the per-measure
pitch gap.
- (#1) Train on cell crops padded with vertical context (+50% measure
  height above + below). New model = v3.
- (#2) Upsample crops at inference so the shorter side is >= 384 px.

Holding everything else fixed (LoRA on safckylj, 2 epochs, same LR/
batch/seed, same eval sample seed=0, 12 dev cells per source).

## Grid

| cond | model    | inf upsample | L7a P / R     | L9 P / R     | openscore P / R |
|------|----------|--------------|---------------|--------------|------------------|
| A    | v2 tight | none         | **68.7/100.0**| **26.8/62.2**| 16.9 / 58.5     |
| B    | v2 tight | 384          | 55.2 / 89.6   | 15.2 / 48.4  | 19.3 / 49.4     |
| C    | v3 pad   | none         | 61.8 / 99.8   | 19.1 / 58.4  | **20.3 / 55.7** |
| D    | v3 pad   | 384          | 65.8 / 98.3   | 18.3 / 58.2  | 16.6 / 50.8     |

## Isolated effects

| effect                                  | comparison | L7a    | L9     | openscore |
|-----------------------------------------|------------|--------|--------|-----------|
| Inference upsample on v2 (native-trained) | A → B    | -13.5  | -11.6  | +2.4      |
| Inference upsample on v3 (pad-trained)    | C → D    | +4.0   | -0.8   | -3.7      |
| Training-time padding                     | A → C    | -6.9   | -7.7   | **+3.4**  |
| Both stacked                              | A → D    | -2.9   | -8.5   | -0.3      |

## Findings

1. **Inference upsampling on a native-trained model is harmful** on
   synthetic data (-13 / -12 pts pitch). The vision tower learned glyph
   appearances at training resolution; 2-3x resampling artifacts look
   like a distribution shift.

2. **Upsampling helps when training was padded.** C → D recovers L7a
   by +4. Padding apparently makes the model more scale-tolerant, but
   not enough to beat baseline A.

3. **Training-time padding splits the win/loss across sources.** Helps
   openscore (+3.4) -- real-world music has lyrics and directions in
   the inter-staff space that padding captures. Hurts L7a/L9 (-7 to
   -8) -- the added context is noise for clean synthetic layouts.

4. **Best per-source**:
   - L7a:       A (baseline 68.7) wins
   - L9:        A (baseline 26.8) wins
   - openscore: C (20.3) wins by +3.4 over A

5. **Caveat on A vs C**: dev sets differ (v2 has 3631 cells, v3 has
   2397 -- padding pushed some cells past the aspect-filter cutoff).
   Some of C's openscore lift could be sample drift toward easier
   cells. A controlled re-run would evaluate both models on a fixed
   intersection of cells.

## What the data says about next moves

- Neither lever is a clear win at the 36-cell scale we tested.
- The per-measure VLM's persistent ~70% L7a pitch cap (vs whole-page's
  ~95%) is not closed by changing crop resolution or context.
- The pattern "rhythm 100%, pitch 60-70%" suggests the model needs
  more pixel detail at the notehead level than either intervention
  delivers, or fundamentally different architecture (smaller model
  with input-faithful preprocessing, or non-VLM seq2seq trained from
  scratch on per-measure pairs).
- More openscore *training* data is probably the bigger lever for
  openscore (we have ~150 unique scores; the dataset has thousands).
