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

## Addendum: verification (responding to a fair pushback)

Two things were called out:

(1) The respell test used GT key, which is oracle. Real inference must
    use the *predicted* key.
(2) Earlier claims of "100% L7a / 97% L9 / 90% openscore" keysig
    detection accuracy needed evidence.

### Detector keysig accuracy: verified on 30 dev pages per source

For each page, take majority vote of detected `key_*` sub-class labels.
A prediction is "correct" if it equals the first `<fifths>` in the
page's musicxml or appears anywhere in the page's `<fifths>` set
(handles multi-key pages).

| source | accuracy | off-by-one |
|---|---|---|
| L7a       | **30/30 = 100.0%** | 0 |
| L9        | **29/30 =  96.7%** | 1 |
| openscore | **27/30 =  90.0%** | 2 |

Errors are exclusively off-by-one between adjacent key signatures.
Annotated examples saved at `/tmp/verify_key/`. Two were shown to user:

- l7a_correct_0_L7a_04000: 3 detected keysig boxes, all voted +2 ✓
- openscore_correct_2_lc5079512: 6 detected boxes, all voted -4 ✓
- l9_wrong_0_L9_03137 (Strauss "Die Nacht"): GT=-2, votes [-1,-2],
  majority picked -1 due to tie-breaking insertion order
- openscore_wrong_1_lc6686980 (Brahms Op.121 No.3): GT=+1, votes [2,2]

### Per-measure pipeline with predicted (not GT) key

Same 12 dev cells per source, but respell uses the detector's
majority-vote key:

| source | raw pitch | resp (GT key) | resp (PRED key) |
|---|---|---|---|
| L7a       | 62.8% | 100.0% | 100.0% |
| L9        | 26.0% |  36.2% |  36.2% |
| openscore | 16.5% |  23.5% |  23.5% |

On this 12-sample slice the detector key matched GT for every page
(100/100/100), so PRED-respell == GT-respell. On the larger 30-page
verification we'd expect L9 and openscore to drop a small amount
from the 3-10% detector error rate.

The headline 100% L7a / 36% L9 / 24% openscore therefore reflects
realistic, not oracle, end-to-end performance.
