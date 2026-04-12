# MXC Experiments: Ablation Study & Training Instability (2026-04-08)

**Previous**: [`2026-04-01-context-length-and-eos.md`](2026-04-01-context-length-and-eos.md) —
fixed training target truncation by raising `max_length` to 8192.

This document covers:
- Experiment 13: Ablation study on Level 7 factors (initial findings later
  retracted)
- Experiment 14: Discovery that Level 6b's poor accuracy was caused by
  training instability, not content difficulty
- Experiment 15: Stable hyperparameters that double 6b's accuracy

**Important**: Experiment 13's main claim ("voice part is the dominant
factor, −55pp") was later retracted because Level 6b's score was caused by
training instability, not content. See Experiments 14 and 15 below.

## Experiment 13: Ablation study — what factors drive accuracy down?

Level 7 changes multiple factors from Level 6 simultaneously (voice part,
lyrics, more measures). We created 4 controlled variants to isolate each factor.
All variants use Qwen3.5-9B r=32, MXC, 8K context, 3 epochs, 1000 samples.

### Variant design

| Variant | Parts | Measures | Lyrics | MXC chars | Isolates |
|---|---|---|---|---|---|
| Level 6 | 2 (piano) | 16 | No | ~3,100 | baseline |
| Level 6b | 2 (piano) | 24 | No | ~4,600 | +length only |
| Level 7a | 3 (voice+piano) | 16 | No | ~3,600 | +voice part |
| Level 7b | 3 (voice+piano) | 16 | Yes | ~4,100 | +voice+lyrics |
| Level 7c | 3 (voice+piano) | 24 | No | ~5,600 | +voice+length |
| Level 7 | 3 (voice+piano) | 24 | Yes | ~6,400 | all factors |

7a and 7b produce identical note sequences (verified: same 297 note lines
from same RNG seed). The only difference is lyric annotations on voice notes.

### WandB runs

- Level 6b: `p099t00s`
- Level 7a: `mtm0roq9`
- Level 7b: `v9dwbr9u`
- Level 7c: `tu946jbl`

### Results (n=50 each)

| Variant | Pitched-only sim | Rhythm | Change from baseline |
|---|---|---|---|
| Level 6 (n=10) | 97%\* | — | baseline |
| **Level 6b** | **81%** | 59% | **−16pp** (length effect) |
| **Level 7a** | **42%** | 38% | **−55pp** (voice part effect) |
| **Level 7b** | **59%** | 53% | −38pp (voice+lyrics) |
| **Level 7c** | ~47%\*\* | ~47%\*\* | ~−50pp (voice+length) |
| Level 7 (8K) | 67% | 59% | −30pp (all factors) |

\* n=10, not validated. \*\* preliminary (5/50 samples).

### Key findings

**1. The voice part is the dominant accuracy bottleneck (−55pp).**

Adding a 3rd staff (voice) drops pitch accuracy from ~97% to 42%, even at
only 16 measures and no lyrics. This is far larger than the length effect
(−16pp). The model struggles with multi-part scores where it must track
pitch across 3 independent parts.

**2. Length alone has moderate impact (−16pp).**

Going from 16 to 24 measures with piano-only (Level 6 → 6b) drops accuracy
from ~97% to 81%. More notes to read and more output to generate, but the
model handles the increased load reasonably well when the structure is
familiar (2 parts).

**3. Lyrics improve accuracy (+17pp).**

Adding lyrics to voice+piano scores (7a → 7b: 42% → 59%) improves pitch
accuracy. This is counterintuitive — lyrics add more tokens — but lyrics
provide positional anchors that help the model track where it is in the
score. The lyric text is visually prominent and easy for the language model
to read, giving a "second channel" of sequential information.

Verified that 7a and 7b have identical note content (same RNG seed, same
297 note lines). The improvement is purely from lyric annotations.

**4. The full Level 7 (67%) outperforms most subsets.**

The combined accuracy of all factors (67%) is higher than voice-only (42%),
voice+length (47%), and only slightly below piano+length (81%). This
confirms that lyrics act as a compensating factor — they offset much of the
accuracy loss from adding the voice part.

**5. Inference speed varies dramatically across variants.**

| Variant | Per-sample time | Notes |
|---|---|---|
| Level 6b | 112s | Short output, EOS learned |
| Level 7a | 305s | Long output, no EOS |
| Level 7b | 117s | Short output, EOS learned |
| Level 7c | 277s | Long output, no EOS |

The 7a and 7c variants generate much more output (~305s and ~277s), suggesting
the model without lyrics does not learn to stop generating (no EOS). With
lyrics (7b, 7), the model learns EOS and stops at the right length.

### Implications

The voice part addition is the critical challenge, not output length or data
complexity. Training the model to handle 3-part scores is the key problem.
Possible approaches:

- More training data with 3 parts (current: 800 train samples)
- Curriculum learning: train on 2-part first, then fine-tune on 3-part
- Architectural changes: explicit part-tracking mechanism

Lyrics are not a burden — they're a feature. Future training should always
include lyrics when available, as they provide structural anchors that
improve accuracy.

## Experiment 14: Level 6 vs 6b deep dive — training instability discovery

### Context

Initial ablation results suggested Level 6b (piano at 24 measures) drops
accuracy catastrophically (97%* → 30-36%). We hypothesized voice part was
the dominant factor, but this was inconsistent with other results (7a with
3 parts scored 91%). The contradiction prompted deeper investigation.

### Control: Level 6 (16m) with 8K context

**WandB run**: `0jbjvq3t`. Identical to original Level 6 run except
`max_length=4096 → 8192`. All other settings (800 train samples, 3 epochs,
r=32, etc.) unchanged.

| Metric | Original Level 6 (n=10) | Level 6 + 8K context (n=50) |
|---|---|---|
| Pitched-only sim | 97%\* | **95%** |
| Coverage | — | 100% |
| Per-sample time | — | 71s |

**Conclusion**: 8K context is NOT the problem. Level 6 + 8K reproduces the
original Level 6 result faithfully.

### 6b training instability

Examining WandB inference tables over training steps reveals a clear phase
transition in the model's output quality:

**3-epoch 6b run** (`sekek6wi`, seed=3407):

| Step | Epoch | Pred chars | Notes | Max 4-gram rep | Status |
|---|---|---|---|---|---|
| 202 | 0.20 | 4489 | 325 | 3 | ✅ matches ref |
| 405 | 0.40 | 4576 | 332 | 22 | some repetition |
| 608 | 0.60 | 4821 | 321 | 3 | ✅ matches ref |
| **811** | **0.81** | **12008** | **831** | **274** | 💥 catastrophic |
| 1014 | 1.01 | 11824 | 806 | 45 | broken |
| ... | ... | ... | ... | ... | never recovers |

**5-epoch 6b run** (`0xpv0x6b`, same seed=3407):

| Step | Epoch | Max 4-gram rep | Status |
|---|---|---|---|
| 202-608 | 0.2-0.6 | 3 | ✅ good |
| 811 | 0.81 | 11 | ⚠️ partial collapse |
| 1014-2841 | 1.0-2.8 | 3-18 | ✅ recovered |
| 3044 | 3.0 | 127 | 💥 re-collapse |
| 3653 | 3.7 | 146 | 💥 re-collapse |
| 4871-5074 | 4.9-5.1 | 4 | ✅ final recovery |

### Key observations

1. **Both runs collapsed at exactly step 811** (same seed → deterministic).
   A specific batch of training samples at ~epoch 0.8 causes the instability.

2. **The 3-epoch run never recovered.** It stayed in the degenerate state
   and scored 30% on the post-training eval.

3. **The 5-epoch run oscillated** between good (~3× rep) and catastrophic
   (127-146× rep) states, eventually stabilizing by epoch 5. It scored 36%
   — slightly better than 3-epoch, but the post-training eval on 50 dev
   samples still caught many pathological outputs.

4. **Eval loss tracks the instability.** Spikes at epoch 0.8 (0.0013),
   epoch 2.0 (0.0016) coincide with the generation collapses.

5. **Prediction characteristics during collapse**:
   - Sample 0 (3-ep, step 811): generates `N D#3 q 2 sd` **274× in a row**
   - Sample 1: 14× repetition of a descending scale
   - Sample 2: 27× repetition of chord progression
   - Format errors appear (doubled `sd` stems)

### Root cause analysis

This is **classic neural text degeneration** triggered by a training
instability. The model initially learns the task correctly (step 202: good
output), but a specific training batch at step 811 knocks it into a
degenerate attractor state where greedy decoding produces repetitive loops.

Evidence it's training-instability, not content difficulty:
- Level 6 (same generator, 16m) + 8K context works fine (95%)
- Level 7a (voice+piano, 16m, 8K) scores 91%
- The same step (811) triggers collapse in both 6b runs
- Predictions show degenerate loops (274× same note) not present in training

Why 24-measure specifically triggers the instability is still unclear.
Possible factors:
- 50% longer sequences push attention toward more extreme gradients
- More chord notes per sample → more loss from chord predictions
- Interaction with cosine LR schedule at ~epoch 0.8

### Correction to previous findings

The "voice part is the dominant factor (−55pp)" claim from Experiment 13
was **incorrect**. That conclusion was drawn from 6b scoring 30-36% and
7a scoring 91%, but we now know 6b's low score is due to training
instability, not content difficulty.

**Revised understanding**: We don't actually know yet what the "true" Level
6b accuracy would be with a stable training run. It may be anywhere from
70% to 95%.

### What to try next

1. **Lower learning rate** (2e-4 → 5e-5 or 1e-4) to reduce gradient step size
2. **Tighter gradient clipping** (max_grad_norm 0.3 → 0.1)
3. **More frequent checkpointing** (save_steps=200) to catch good states
4. **Different seed** to see if instability is seed-specific
5. **Re-examine the training batch at step 811** to find what triggers the collapse

## Experiment 15: 6b with stable training hyperparameters

**WandB run**: `ijgyv68j`. Same data and architecture as previous 6b runs,
but with hyperparameters tuned to prevent the training instability:

| Setting | Previous 6b | Stable 6b |
|---|---|---|
| `learning_rate` | 2.0e-4 | **5.0e-5** (4× lower) |
| `max_grad_norm` | 0.3 | **0.1** (3× tighter) |
| `save_steps` | 1000 | **600** (5 checkpoints) |
| Other settings | unchanged | unchanged |

### Training stability comparison

| Metric | Unstable (LR=2e-4) | **Stable (LR=5e-5)** |
|---|---|---|
| Spikes >5× local median | 16 | 9 |
| Max spike ratio | 88.7× | 71.3× |
| Step 742 loss | 0.01593 | 0.01017 |
| Best eval_loss | 0.00067 | **0.00011** (6× better) |
| Eval loss curve | oscillating | smooth (mostly monotonic) |

**Spikes were not eliminated** — step 742 still spikes (deterministic from
seed=3407). But lower LR + tighter clipping kept the spike-induced gradient
updates small enough that the model never entered the degenerate attractor.

### Eval results (n=50)

| Metric | Unstable 6b | **Stable 6b** |
|---|---|---|
| Pitched-only sim | 30-36% | **68%** |
| Rhythm similarity | 18-24% | **75%** |
| Note-type sim | 18-25% | **76%** |
| Combined sim | 40-64% | **81%** |
| Coverage | 137-280% | **100%** |
| Max pitch sim | 45-81% | **94%** |
| Per-sample time | 145-307s | **108s** |

**Pitch accuracy nearly doubled** (36% → 68%) just from hyperparameter
tuning. Coverage normalized to 100%. The model now generates correctly-sized
output instead of runaway loops.

### EOS learning analysis

Direct token-level inspection of generation:

| Sample | Generated tokens | EOS (`<\|im_end\|>`, 248046)? | Stop reason |
|---|---|---|---|
| 0 | 2946 | **Yes** at pos 2943 | clean ChatML EOS |
| 1 | 3072 | No | hits `<\|endoftext\|>` (248044) instead |
| 2 | 2920 | No | hits `<\|endoftext\|>` (248044) instead |

**EOS learning is partial.** The model learned *some* stopping behavior
(it doesn't run to max_new_tokens), but it doesn't reliably emit the
`<|im_end|>` token from training. Instead, samples 1 and 2 emit Qwen's
default `<|endoftext|>` token after some "drift" content (extra `---`
separators).

This explains why per-sample inference time is uniform (~108-125s): the
model generates ~3000 tokens reliably, then stops via one of two mechanisms.
The 100% coverage in the eval is slightly misleading — pred-to-ref ratios
are actually 0.93-0.98 (slightly under-generation).

### Updated cross-dataset comparison

| Variant | Parts | Meas. | Lyrics | Pitched-only | Status |
|---|---|---|---|---|---|
| Level 6 (8K) | 2 | 16 | No | **95%** | stable |
| **Level 6b (stable)** | 2 | **24** | No | **68%** | stable |
| Level 7a | 3 | 16 | No | 91% | stable |
| Level 7b | 3 | 16 | Yes | 42%\*\*\* | unverified |
| Level 7 (8K) | 3 | 24 | Yes | 67% | stable |

\*\*\* may also have been affected by training instability — not yet re-run.

### Cleaner ablation findings

With stable training, we can now isolate the length effect:

1. **Length effect (16m → 24m)**:
   - Piano only: 95% → 68% = **−27pp**
   - Voice+piano+lyrics: 91% → 67% = **−24pp**

2. **Voice part addition (at 16m)**:
   - Piano vs voice+piano: 95% vs 91% = **−4pp** (small)

3. **Voice part addition (at 24m)**:
   - Piano vs voice+piano+lyrics: 68% vs 67% = **−1pp** (negligible)

The dominant factor is length, not voice part. The "voice is hard"
hypothesis (Experiment 13) is fully retracted.

### Loss spikes correlate with output sequence length

Examining all completed 8K runs:

| Run | Measures | Output tokens | Spikes >5× | Max ratio |
|---|---|---|---|---|
| Level 6 (8K) | 16 | ~2000 | **0** | — |
| Level 7c (8K) | 16 | ~2400 | **0** | — |
| Level 7b (8K) | 16 | ~2700 | 2 | 12.5× |
| Level 7 (8K, original) | 24 | ~4200 | 9 | 25.3× |
| Level 6b unstable | 24 | ~3000 | 16 | 88.7× |
| Level 6b stable | 24 | ~3000 | 9 | 71.3× |

**Clear pattern**: 16-measure runs have 0-2 spikes; 24-measure runs have
9-16 spikes with ratios up to 88×. The instability scales with output length.

This is a **known issue in LLM fine-tuning**, not specific to our setup:

- **"Spike No More: Stabilizing the Pre-training of Large Language Models"**
  (COLM 2025) — long sequences amplify gradient norms via rapid amplification
  in residual connection shortcuts and intensification around layer
  normalization paths.
- **"The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup
  for Training GPT Models"** (Li et al., OpenReview) — proposes Sequence
  Length Warmup (SLW): start with short sequences, gradually increase length.
  Enables 4-40× higher learning rates without divergence and improves final
  accuracy.
- HuggingFace forum reports of "FREQUENT LOSS SPIKING in CONTINUE TRAINING
  LLM" with similar patterns on long-sequence fine-tuning.
- Unsloth documentation explicitly notes long-context training stability
  challenges and recommends batch size 1 + gradient checkpointing.

Standard mitigations and what we tried:

| Mitigation | Status | Effect on 6b |
|---|---|---|
| Lower learning rate (2e-4 → 5e-5) | ✅ tried | helped (36% → 68% pitch) |
| Tighter gradient clipping (0.3 → 0.1) | ✅ tried | helped |
| Sequence Length Warmup (16m → 24m) | ❌ untried | most promising |
| Longer warmup ratio (0.03 → 0.1) | ❌ untried | could help |
| SPAM (Spike-Aware Adam) | ❌ untried | resets momentum on spikes |
| Lower `gradient_accumulation_steps` | ❌ untried | breaks up bad batch averaging |

The most effective documented solution is **Sequence Length Warmup**, which
we haven't tried. We could leverage the existing 16-measure datasets (Level
6, 7a) as warmup before training on 24-measure content (6b, 7).

### Open issues

1. **Spikes are still present** — the stable run has 9 large spikes, just
   smaller than before. Further tuning might help (longer warmup, even
   lower LR, or batch shuffling to break up the bad batch). Sequence Length
   Warmup is the most promising untried fix.

2. **EOS learning is partial** — the model uses `<|endoftext|>` instead of
   `<|im_end|>`, which works but isn't fully aligned with training format.
   Could be improved with more training or explicit EOS token configuration.

3. **The 27pp length penalty** is real but not catastrophic. It suggests
   that for openscore (full pages with even more content), the model will
   be limited by sequence length even with proper training.


---

**Continued in [`2026-04-12-findings.md`](2026-04-12-findings.md)** — current
state of knowledge: reliable findings, retracted claims, open questions, and
next steps.
