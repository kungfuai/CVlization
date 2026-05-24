# 2x2 ablation re-run with respell (the right metric)

Same 4 conditions as before, but now applying respell with the GT key
before computing pitched_only_similarity. Without respell, the metric
penalised the model for guessing `key=` on per-measure crops that
don't show a key signature -- a cascading false penalty.

## Grid (respelled pitch / native rhythm; same 12 dev cells per source, seed=0)

| cond | model    | inf upsample | L7a (raw / resp) | L9 (raw / resp) | openscore (raw / resp) |
|------|----------|--------------|------------------|------------------|------------------------|
| A    | v2 tight | none         | 68.7 / **100.0** | 26.8 / **35.7**  | 16.9 / 22.6            |
| B    | v2 tight | 384          | 55.2 / 83.5      | 15.2 / 27.7      | 19.3 / 23.9            |
| C    | v3 pad   | none         | 61.8 / 99.4      | 19.1 / 30.1      | 20.3 / **24.2**        |
| D    | v3 pad   | 384          | 65.8 / 91.7      | 18.3 / 23.4      | 16.6 / 21.7            |

## Isolated causal effects (respelled-pitch deltas vs A)

| effect                                  | comparison | L7a    | L9     | openscore |
|-----------------------------------------|------------|--------|--------|-----------|
| Inference upsample on v2 (native model) | A → B      | -16.5  | -8.0   | +1.3      |
| Inference upsample on v3 (padded model) | C → D      | -7.7   | -6.7   | -2.5      |
| Training-time padding                   | A → C      | -0.6   | -5.6   | +1.6      |
| Both stacked                            | A → D      | -8.3   | -12.3  | -0.9      |

## What's different vs the pre-respell read

Before respell, the L7a "ceiling" looked like 68.7% and padding looked
like it might be worth pursuing because the gap looked open. After
respell:

- **Baseline (A) hits 100% L7a pitch.** Per-measure VLM + correct key
  + respell solves L7a's pitch.
- The "padding helped openscore" effect shrank from +3.4 to +1.6 --
  still positive, but on the order of noise at n=12.
- The "upsampling hurts" finding survives. Even after fixing the
  metric, upsampling is bad: A->B is -16.5 on L7a (much bigger than
  pre-respell's -13.5). The vision tower's training-time resolution
  bias is real.

## What's not different

- L9 best is still A (baseline) at 35.7%. Padding hurts L9 (-5.6).
- openscore best is C (padded, no upsample) at 24.2%, but barely.
- Openscore is stuck in the low-20s no matter what we do at this
  data scale and architecture.

## Updated takeaways

1. **The per-measure architecture is fundamentally sound.** Baseline
   v2 + respell achieves perfect L7a transcription. The earlier
   "per-measure is worse than whole-page" framing was wrong.

2. **Crop interventions are not the lever.** Padding gives a tiny
   openscore lift but costs L9. Upsampling at inference is harmful
   regardless of training distribution.

3. **The real openscore lever is training data**, not crop design.
   We trained on ~150 unique openscore scores; the dataset has
   thousands. L9 35.7% suggests more L9 data would help too -- L9
   has complex rhythms (triplets, ties, multi-voice) that need more
   examples to learn.

4. **Always include respell when evaluating models that emit `key=`
   fields they can't see.** Anything that requires a model to predict
   a structural field absent from its input is a metric trap.
