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

## Auxiliary key-signature task — RESULT: failed (2026-05-17)

Ran Level 7a SFT with `key_aux_ratio=0.25` (4000 transcription + 1012
key-only examples). Result: **made things worse.**

| Key | r=32 baseline | key-aux | Δ |
|---|---|---|---|
| −4 | 97.9% | 96.8% | −1.1 |
| −3 | 32.5% | 32.4% | 0 |
| −2 | 84.4% | **25.4%** | **−59** |
| −1 | 13.8% | 4.7% | −9 |
| 0 | 96.5% | 96.9% | +0.4 |
| +1 | 15.1% | 15.3% | 0 |
| +2 | 26.9% | 30.4% | +3.5 |
| +3 | 26.0% | 26.1% | 0 |
| +4 | 99.1% | 95.4% | −3.7 |
| **Overall** | **70.2%** | **61.1%** | **−9** |

Broken keys did not improve; key=−2 collapsed (84→25%); overall −9pp from
interference. **The loss-insensitivity hypothesis is refuted** — if it were
purely a gradient-pressure problem, the auxiliary task (where `key=N` is the
whole loss) would have fixed it. It didn't.

## Revised root cause: input-resolution bottleneck (2026-05-17)

The model does NOT downsample our images — Qwen3-VL's processor cap is
`longest_edge=16777216` px; our 1240×1754 images (2.2M px) are far below it,
processed at full resolution → 2145 image tokens.

But the **tokenization granularity is coarse**: `patch_size=16, merge_size=2`
→ each image token covers a **32×32 pixel block**.

In our 150-DPI renderings:
- A single sharp/flat symbol ≈ 12×30 px
- A 3-sharp key signature ≈ 45 px wide
- At 32 px/token, the **entire key signature spans only ~2 tokens**

To distinguish "2 sharps" from "3 sharps," the model must count symbols that
are individually *sub-token-sized*, packed into ~2 tokens. It can't. Instead
it pattern-matches the *gross shape* of the key signature into buckets:
- key=0 (empty): trivially distinguishable
- key=±4 (widest): distinguishable by gross width
- key=±1,2,3 (intermediate): require sub-token counting → fails

This is a genuine **perception** bottleneck, not a training-objective one —
which is why neither RL, larger LoRA, nor auxiliary data mixing helped, and
why one of them actively hurt. It matches the frontier doc-AI consensus:
fine-detail perception is fixed by resolution / better encoder / region
processing, never by training-objective tricks.

## Status

- Qwen3-VL-8B SFT pipeline: working
- Best SFT checkpoint: r=32 vision=True (`outputs/safckylj`), 70% mean / 96.6% median
- key-aux checkpoint (`sf4lrr47`) is *worse* — do not use
- Root cause: **input-resolution bottleneck** — key signature spans only
  ~2 image tokens at 150 DPI / 32-px-per-token granularity

### Ruled out experimentally
- ❌ RL (zero rollout variance on broken samples)
- ❌ LoRA capacity (r=128 no help)
- ❌ Auxiliary data mixing (no help, −9pp interference)
- ❌ Loss-insensitivity as the *sole* cause (refuted by the above)

### Remaining levers — all perception-side
1. Higher render DPI (300+) so the key signature spans more tokens
2. Region crop-and-zoom on the key signature (Reducto-style)
3. Stronger / higher-resolution base VLM

## Visual confirmation of the resolution bottleneck (2026-05-17)

Rendered images of a broken vs good sample, side by side:

- **L7a_04500 (key=+4, model correct):** after each clef, a chunky
  `####` cluster — wide, dense, visually distinctive.
- **L7a_04517 (key=+1, model misreads as key=3):** after each clef, a
  single tiny `#` — one ~12 px mark.

The model reads key signatures by gross shape, not by counting. A 4-sharp
block is a distinctive wide shape; a 1-sharp mark is a single symbol near
the token-resolution floor. This *is* the diagnosis, visible to the eye.

Note: the key signature is repeated on every staff of every system —
3 systems × 3 staves = **9 copies on one page**. The model has 9 chances
and still fails. Redundancy does not rescue a per-instance resolution
problem; more copies at the same coarse granularity don't help. A
multi-page score with 30 copies would fail identically.

## SEPARATE BUG: openscore/PDMX page-slicing drops global key/clef (2026-05-17)

While investigating multi-page handling, found a real data bug in the
*real-data* pipeline (does NOT affect synthetic Level 7a/9).

`datasets/omr/pipeline.py::slice_musicxml` uses music21's
`score.measures(start, end)` to slice a multi-page score into per-page
MusicXML. Verified empirically (key=+4 score, music21 9.9.1):

| Slice | Key in sliced MusicXML | Clefs |
|---|---|---|
| measures 1-10 (incl. m1) | fifths=[4,4,4] ✅ | [G,G,F] ✅ |
| measures 20-30 (mid-score) | **fifths=[] ✗** | [G,G,G] ✗ (bass clef lost) |
| measures 40-48 (end) | **fifths=[] ✗** | [G,G,G] ✗ |

**music21's `.measures()` does not carry the global key signature or
non-default clefs into mid-score slices** — only the slice containing
measure 1 keeps them.

Consequence: for openscore/PDMX, every page after page 1 gets a
**corrupted reference label** — the page image (rendered from the full
score) visually shows key=4 on every system, but the sliced reference
MXC2 says key=0 (default, `<fifths>` absent). The model would be trained
"image with 4 sharps → output key=0" on every non-first page.

This is a serious data-integrity issue for the multi-page real-data
corpora. It is independent of the resolution bottleneck. It must be
fixed before any further openscore/PDMX training — likely explains part
of the historically poor openscore accuracy.

**Fix:** after slicing, explicitly inject the active key signature and
clef into the first measure's `<attributes>` of each sliced page (query
music21 for the active KeySignature/Clef context at the slice start, or
post-process the MusicXML).

## Cross-experiment thread

The OMR plateau is a **perception** problem, not an information or
training-objective problem. The image technically *contains* the key
signature, but at our render resolution × the model's token granularity,
the fine accidental marks are not resolvable. Every downstream fix that
assumed the model could see the detail (RL to improve consistency,
Audiveris hints to add information, data mixing to add gradient pressure)
was attacking the wrong layer. The productive direction is to make the
detail physically resolvable in the model's input.

Separately, the page-slicing bug shows the real-data pipeline has a
label-integrity problem orthogonal to the model — also worth fixing
since corrupted labels cap achievable accuracy regardless of the model.

## DPI hypothesis — cheap probe result: leans NEGATIVE (2026-05-17)

Re-rendered 7 Level 7a samples at 150 and 300 DPI, fed both to the
existing `safckylj` model, asked the key signature directly.

| Sample | Ref key | 150 DPI | 300 DPI |
|---|---|---|---|
| L7a_04517 | 1 | 2 ✗ | 2 ✗ |
| L7a_04521 | 1 | 2 ✗ | 2 ✗ |
| L7a_04526 | −1 | −2 ✗ | −2 ✗ |
| L7a_04519 | 3 | 4 ✗ | 4 ✗ |
| L7a_04532 | 2 | 2 ✓ | 2 ✓ |
| L7a_04500 | 4 | 4 ✓ | 4 ✓ |
| L7a_04508 | 0 | 0 ✓ | 0 ✓ |

**Every sample gave a byte-identical answer at both resolutions.**
Doubling input resolution shifted nothing.

Caveat: confounded — the model was trained at 150 DPI, so it may not
*use* 300-DPI detail. The probe cannot fully rule out that a model
*trained* at 300 DPI would improve.

But two signals lean against the resolution hypothesis:
1. Not one answer shifted — if 2× pixels gave any extra signal, at
   least one borderline sample should flip. None did.
2. Errors are **systematic off-by-one away from zero** (1→2, 3→4,
   −1→−2), not noisy. A resolution problem produces blurry errors that
   change with resolution. A consistent off-by-one looks like a learned
   **counting / representation** miscalibration, not a "can't see the
   pixels" problem.

Updated read: the bottleneck looks less like raw pixel resolution and
more like the model's learned image→count mapping. Holding off on the
expensive 300-DPI full retrain (lower expected value now).

## Next: crop-and-zoom probe

Crop just the key-signature region, feed it large and isolated. This
separates three hypotheses:
- Model still miscounts a big isolated key signature → counting /
  representation deficit (not resolution, not distraction)
- Model gets it right when isolated → attention-spread over the busy
  full page is the issue
- (Resolution per se already argued against by the DPI probe)
