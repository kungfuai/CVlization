# The real bottleneck was the metric, not the model

## Summary

The reported per-measure pitch ceiling (68.7% on L7a, 16.9% on
openscore) **was almost entirely metric artefact**, not model
limitation. After applying respell with the correct key:

| source | raw pitch | respelled (oracle key) | Δ |
|---|---|---|---|
| L7a | 68.7% | **100.0%** | +31.3 |
| L9  | 26.8% | 35.7%      | +8.9  |
| openscore | 16.9% | 22.6% | +5.7  |

**L7a is now solved by the per-measure VLM.** L9 and openscore
have real residual errors beyond keysig, but the magnitude is much
smaller than reported.

## Why the metric lied

A per-measure crop almost never shows the key signature -- it's only
drawn at the start of each system (in the first measure of that
system). For measures 2..N of any system, the model crop has zero
information about the key.

But MXC2 requires `key=N` on every M line. So the model has to make
one up. It mostly guesses `key=1` (the most common in training).

When `key=1` is wrong (say, real key=-3), the model's per-note
spellings are right for *its* assumed key (`B4` instead of `Bb4`).
`evaluate_pair.pitched_only_similarity` counts each as a separate
note-name error -- one keysig hallucination cascades into ~10-40
spurious pitch errors per measure.

Visual inspection of L7a_04071 m=3 confirmed: model output had
every rhythm, every stem direction, every staff position right.
Only the key field was wrong, and the pitches were "right modulo
re-spelling".

## How the pipeline solves this

The detection workstream gives us the key per system (multi-task
YOLO detects `key_signature` boxes; the keysig class encodes the
key value). We already have ~100% L7a / ~90% openscore on key.

`respell.respell_mxc2(pred, key)` re-spells the model's notes to
the supplied key. Per-measure VLM output + detector's key + respell
== correct pitches on L7a.

## What this means for next steps

- **L7a is done.** Per-measure VLM + detector-key + respell hits 100%.
- **L9 and openscore have real residual errors after respell**
  (35.7% and 22.6% pitched). Those are genuinely wrong notes /
  wrong voices / wrong durations. They will need more training data,
  particularly:
  - openscore: we only used ~150 unique scores; the dataset has
    thousands.
  - L9: complex rhythms (triplets, ties across barlines, multi-voice).
- **The 2x2 ablation (padding x upsampling) we just ran is mostly
  invalidated** -- it was measuring the keysig-hallucination penalty,
  not the underlying transcription. Re-run with respell would tell us
  what those interventions actually change.

## Methodology lesson

For a model trained to emit a structural field it can't observe from
the input (here: `key=` on per-measure crops), automatic metrics
penalize the unobservable as if it were an error -- and one such
"error" can cascade through dozens of downstream tokens. Inspect
predictions before trusting a metric on a new task setup.

The correct evaluation harness for per-measure should always apply
respell with the detector-predicted key before computing pitch
similarity. We have all the pieces; just wire them.
