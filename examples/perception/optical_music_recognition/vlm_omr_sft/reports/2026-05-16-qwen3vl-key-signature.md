# Qwen3-VL Migration & the Key-Signature Bottleneck

**Previous**: [`2026-05-13-audiveris-hint-augmentation.md`](2026-05-13-audiveris-hint-augmentation.md)
— closed the Audiveris-hint direction; concluded the 84% Level 9 plateau
needs a different break.

## Why we migrated to Qwen3-VL

RL was the natural next lever for the plateau, but Qwen3.5-9B-VLM blocks RL
on every framework (Miles/slime bug #1894, TRL+vLLM #5269, unsloth multimodal
SFT token-mismatch). Qwen3-VL has proven RL infrastructure (Miles geo3k_vlm,
verl, ms-swift, unsloth all ship working Qwen3-VL GRPO recipes).

Plan: migrate SFT to Qwen3-VL-8B, then run RL with confidence.

## Phase 1-2: SFT on Qwen3-VL-8B

| Run | Config | Level 7a pitched-only |
|---|---|---|
| Smoke (100 steps) | — | undertrained, ~5% |
| Full SFT, vision=False | 3 epochs, r=32 | **44% mean** |
| Full SFT, vision=True | 3 epochs, r=32 | **70% mean / 96.6% median** |
| Full SFT, vision=True | 3 epochs, r=128 | **72% mean / 96.x median** |

Key facts:
- No unsloth multimodal bug on Qwen3-VL — pipeline runs clean.
- Vision LoRA matters here (44% → 70%) — unlike Qwen3.5 where vision=False
  was fine. Different vision encoder architecture.
- **The mean (70%) badly misrepresents the model.** Distribution is bimodal:
  27/50 samples ≥95%, 17/50 samples <50%. Median is 96.6%.

## The bimodal distribution → key-signature root cause

Accuracy broken down by key signature (r=32 vision=True SFT, n=50 test):

| Key (fifths) | n | Accuracy |
|---|---|---|
| −4 | 10 | 97.9% |
| −3 | 4 | 32.5% |
| −2 | 7 | 84.4% |
| −1 | 2 | 13.8% |
| 0 | 9 | 96.5% |
| +1 | 6 | 15.1% |
| +2 | 2 | 26.9% |
| +3 | 3 | 26.0% |
| +4 | 7 | 99.1% |

**The model is near-perfect on {−4, 0, +4} (and partially −2), broken on
everything else.** When it misreads the key, every accidental-affected note
in the page is wrong → pitched-only similarity collapses.

## Root-cause investigation — four hypotheses, three eliminated

| Hypothesis | Test | Verdict |
|---|---|---|
| Training-data imbalance | Counted key dist in train split | ❌ Uniform (~440/key of 4000) |
| Undertraining | r=32 eval_loss curve | ❌ Flat by epoch 2 (0.0046→0.0025→0.0022) |
| LoRA capacity | Retrained r=128 (4× params) | ❌ Broken keys unchanged (15-32%) |
| **Loss insensitivity** | (remaining) | ✅ **Confirmed** |

### Why loss insensitivity is the cause

The key signature is ~1-3 tokens out of ~3000 per training sample. Getting
it wrong costs almost nothing in token cross-entropy. The model can — and
did — drive eval_loss to 0.0022 *without* learning key signatures. The loss
objective is essentially blind to this error.

So the model took the lazy path: learn key signature as holistic
**image→bucket classification**, memorize the 3-4 most visually distinctive
buckets ({−4, 0, +4}), and snap everything else to the nearest learned bucket.

### Direct-key diagnostic

Asked the model directly "how many sharps/flats?" instead of full
transcription (n=6, mixed keys):
- For 2 broken key=+1 samples: model answered "1" **correctly** — yet
  produces key=3 in full transcription. The visual info IS perceivable.
- For a key=+4 sample it transcribes correctly (99%): model answered "12"
  (garbage) to the direct question — it has **no explicit counting ability**,
  only holistic pattern-matching.

Confound: the model was never trained on the direct-question format, so
answers are noisy. But the signal is consistent with loss-insensitivity:
the model perceives the key signature but has no incentive to encode it
correctly during transcription.

## RL on Level 7a — and why it can't fix this

We ran GRPO on Level 7a (the bimodal model is, in theory, an ideal RL
testbed — high reward variance, clear failure mode).

Result: 70.2% → 70.6%. **No change.** The 17 broken samples stayed broken.

Verified the mechanism directly: sampled a broken sample 8× at temp 0.8,
then 6× at temp 1.2. **Every rollout byte-identical** — the model predicts
the wrong key with total confidence; even aggressive sampling produces zero
variance.

GRPO improves by reinforcing higher-scoring rollouts within a group. With
zero rollout variance, GRPO's group-relative advantage is `(r−mean)/std`
with std=0 → zero gradient. **RL is provably powerless on the deterministic
failures**, regardless of framework (Miles included — same mechanism).

Note: the RL *reward* IS aligned with the goal (wrong key → wrong pitches →
low reward), unlike the SFT token-loss. But an aligned reward is necessary,
not sufficient — RL also needs exploration, and SFT's loss-insensitivity had
already collapsed the model into a confident-wrong state with no exploration.

PPO/REINFORCE (non-group-relative baseline) would not be *instantly* stuck
the way GRPO is, but they'd have to wander out of an extremely peaked
distribution — not a clean fix either.

## The fix: auxiliary key-signature task via data mixing

The diagnostic showed the model lacks the "read the key signature" skill.
Directly teach that skill by mixing dedicated key-signature examples into
the SFT data.

### Why data mixing, not loss-weighting

Loss-weighting (upweight the `key=` token in the collator) addresses the
symptom. Data mixing teaches the skill and is the more standard technique:
- "Mix in sub-task examples, same next-token objective" is the dominant
  data-centric way to fix specific LLM weaknesses (cf. instruction-tuning
  mixtures, capability-preservation data blends).
- On a key-only example, the key signature IS the entire output — so it's
  a large fraction of that example's loss. Data mixing achieves the gradient
  pressure of loss-weighting through dataset design, without a custom loss.

### Implementation plan

Build-time augmentation (not per-step collator):
- Keep all 4000 transcription examples
- **Add** ~1000 key-only examples (25% mix) derived from the same images
- Each key-only example:
  - `user = [image] + "What is the key signature of this score?"`
  - `assistant = "key=N"` — using the **exact `key=N` token** that fails in
    transcription, so the auxiliary task is targeted practice of that token
    conditioned on the image
- Result: ~5000 examples, +25% training time
- Generated from existing dataset at load time — nothing new stored

### Open questions for the run
1. Does the auxiliary task transfer to the transcription task (does the
   model get `key=N` right in full transcription, not just when asked)?
2. Mixing ratio — 25% is a starting guess; may need tuning.
3. Does it generalize across all 9 key signatures or just the trained set?

## Status

- Qwen3-VL-8B SFT pipeline: working (no unsloth bug)
- Best SFT checkpoint: r=32 vision=True (`outputs/safckylj`), 70% mean / 96.6% median
- Root cause: loss-insensitivity to key-signature tokens — confirmed
- RL on Level 7a: no improvement — confirmed and mechanistically explained
- Next: auxiliary key-signature task via data mixing

## Cross-experiment thread

This is the same lesson as the Audiveris and RL arcs: the OMR plateau is not
an information problem (the image has the signal; the model can perceive it)
— it's a *training-objective* problem. The token-loss doesn't value the
fine-grained details (key signatures, accidentals) that dominate the eval
metric. Fixes that align the objective with the goal (data mixing, loss
weighting) are the productive direction; fixes that add information
(Audiveris hints) or assume the model is merely inconsistent (RL) are not.
