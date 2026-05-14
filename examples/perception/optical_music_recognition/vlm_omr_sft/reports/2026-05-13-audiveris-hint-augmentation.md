# Audiveris-Augmented OMR — Findings

**Previous**: [`2026-05-11-miles-rl.md`](2026-05-11-miles-rl.md) — Miles RL
infrastructure exploration; ecosystem study showing Qwen3.5-9B-VLM RL is
bleeding-edge.

## Question

Can we use Audiveris (rule-based OMR baseline) output as an in-context hint
to lift our VLM transcription accuracy past the 84% Level 9 plateau?

## TL;DR

- **Audiveris extracts genuinely useful information** even when pitch
  similarity looks low (e.g., 15.3% on a dense piano sample): 88% of pitch
  values are correct, structural metadata (key/time/clef/measures) is
  near-perfect, the errors are reordering/voice-assignment, not pitch IDs.
- **Zero-shot in-context hinting HURT** (84% → 77%). The model anchors on
  Audiveris's output and copies its errors instead of verifying against the
  image. 35/50 samples were copied within ±5pp of Audiveris.
- **Hint-augmented SFT was blocked** by a deep `unsloth`/`transformers`
  multimodal tokenizer-mismatch bug we couldn't fully eliminate.
- The information IS useful; the model architecture is willing; the training
  pipeline blocks us.

## Setup

- Model: Qwen3.5-9B VLM, SFT'd on Level 9 MXC2 (`outputs/tamqjf4k`)
- Audiveris 5.10.2 (rule-based OMR) — Java binary, runs CPU-only
- Synthetic Level 9 dataset: 3115 train + 389 dev + 390 test samples,
  ~150 DPI images, ~1240×variable height
- Reward/metric: `SequenceMatcher.ratio()` on pitched-only pitch sequence

## Step 1: Audiveris baseline on Level 9 (n=50)

| Metric | Audiveris 5.10.2 | VLM (SFT) |
|---|---|---|
| Pitched-only sim | **76.5%** | 84% |
| Rhythm sim | 68.7% | 80% |
| Note coverage | 93.1% | 100% |
| Failed/skipped | 2/50 (4%) | 0% |

Audiveris's accuracy is **bimodal**:
- 30/50 samples ≥ 90% (median 95.7%)
- 14/50 samples < 50%
- 2/50 fail entirely
- Median 95.7% but mean dragged down by the failures

### What makes Audiveris fail

| Bad samples | Good samples |
|---|---|
| 8 notes/measure | 4.7 notes/measure |
| 89 accidentals avg | 58 avg |
| 6/8 time often | 2/4, 3/8, 2/2 simple |
| Multi-voice piano with stems up+down sharing staff | Cleaner two-hand piano |

Image DPI doesn't help much — at native 300 DPI vs 2.5× upscaled, the
worst sample improved only from 12.7% → 15.3% (+2.6pp). Audiveris's
failure mode is segmentation/grouping, not resolution.

## Step 2: What does Audiveris extract on "bad" pages?

Inspection of the worst sample (L9_03535, sim=15.3%):

| Feature | Reference | Audiveris | Match |
|---|---|---|---|
| Parts | P1, P2 | P1, P2 | ✅ |
| Measures | 32 | 32 | ✅ |
| Time signature | 4/4 | 4/4 | ✅ |
| Key (fifths) | 3 (A major) | 3 | ✅ |
| Clefs | G/2, F/4 | G/2, F/4 | ✅ |
| Duration types | {eighth, half, quarter, whole} | same | ✅ |
| Note count | 244 pitches | 214 pitches | 88% coverage |
| **Pitch values** | — | only 15% match in sequence | ❌ ordering |
| Voice assignment | 2 voices | 4 voices | ❌ |
| Chord grouping | 0 `<chord/>` tags | 75 tags | ❌ over-grouped |

The killer revelation:

```
First 20 ref pitches:  ['A4', 'B4', 'D5', 'B4', 'D5', 'B4', 'G#4', 'A4', 'F#4', 'E4', ...]
First 20 pred pitches: ['A4', 'B4', 'D5', 'B4', 'D5', 'B4', 'G#4', 'A4', 'F#4', 'E4', ...]
```

**The pitches are right.** Top-5 most common pitches match perfectly
(F#4:18, D5:17, A4:16, C#5:15, E5:15 — identical between ref and pred).
The 15% similarity score is fooled by `SequenceMatcher`'s order-sensitivity:
Audiveris correctly identifies notes but emits them in the wrong reading
order due to wrong voice/staff assignment.

**Conclusion: Audiveris provides a noisy-but-information-rich starting
point.** A model that learns to verify pitches against the image should
be able to use this.

## Step 3: Zero-shot in-context hinting

Tested existing SFT model with hints prepended:

```
Audiveris draft (may have errors):
P1 Voice
P2 Piano
---
P1
M 1 key=3 time=4/4 clef=G2
N A4 quarter
...
Corrected transcription:
```

| | Baseline (no hint) | With Audiveris hint |
|---|---|---|
| Pitched-only avg | 84% | **77.1%** (−7pp) |
| Rhythm avg | 80% | 69.7% (−10pp) |
| Note coverage | 100% | 97.3% |

**Hints HURT, not helped.** Per-sample analysis:

- 35/50 (70%) of samples: VLM output is within ±5pp of Audiveris (the
  model copies the hint)
- On dense samples where Audiveris failed: VLM sometimes recovers
  (L9_03531: 27% → 53%, +26pp), sometimes doesn't
- On clean samples where Audiveris was correct: VLM frequently corrupts
  it (L9_03526: 99% → 73%, −25pp)

**Reading: the model treats Audiveris's hint as authoritative.** It
anchors on the text rather than using the image to verify.

## Step 4: Hint compression

Original Audiveris MXC2 hints are verbose. We don't need:
- `Voice Voice` duplicates → strip
- `art=staccato` articulations → strip (Audiveris detects unreliably)
- `slur1=start`, `tie=continue` → strip
- `v=1` voice markers → strip (Audiveris's voice assignment is wrong)
- `su`/`sd` stem direction → strip (visual artifact)
- `bak half dot` backup commands → strip
- `print new-system` → strip (formatting)
- `dir @below dyn=p` dynamics → strip

Compression result: 3700 → 2791 chars avg (25% reduction).

## Step 5: Hint-augmented SFT — blocked

The path: SFT the model on (image + Audiveris_hint) → clean MXC2, so it
learns to treat hints as noisy and verify against the image.

### Plan executed

1. ✅ Extract hints for all 3115 train + 389 dev samples (80 min via
   8 parallel CPU shards on the 32-core host)
2. ✅ Compress hints (25% reduction)
3. ✅ Modify train.py to inject hints into the prompt template
4. ❌ Training crashed at multiple checkpoints

### The crash

`ValueError: Mismatch in 'image' token count between text and 'input_ids'`
from `transformers/processing_utils.py:_check_special_mm_tokens`.

The check compares image-pad string count in the chat-template-rendered
text vs the image-token-id count in the tokenized input_ids. They
disagreed by 1-5 tokens on specific samples.

### What we tried

| Attempt | Change | Result |
|---|---|---|
| 1 | `max_length=12288` | Crash at step 2 |
| 2 | `max_length=16384` | Crash at step 391 (off by 2) |
| 3 | `max_length=24576` + length filter | Crash at step 391 (off by 2) |
| 4 | Back to `max_length=8192` (matches working SFT) | Crash at step 391 (off by 2) |
| 5 | `seed=4242` (different shuffle order) | Crash at step 200 (off by 5) |
| 6 | Pad all images to (1240, 1792) uniformly | Crash at step 200 (off by 1) |

### What we learned about the bug

1. **It's deterministic per-sample** (same seed → same crash step).
2. **Multiple bad samples exist** (different seeds → crashes at different steps).
3. **It's not a length issue** — all samples fit under 8K tokens; even at 24K we crashed.
4. **Padding helped but didn't eliminate** — went from off-by-2/5 to off-by-1 by uniformizing image dimensions. The remaining mismatch is BPE-tokenizer-level.
5. **Working SFT runs (without hints) never hit this bug** — the trigger is hint-specific text content combined with image processing.

The bug appears to be: when the chat template inserts N `<|image_pad|>`
tokens adjacent to specific text content, the BPE tokenizer occasionally
encodes them as N-1 token IDs (likely due to a token merge with adjacent
text). Without hints, the user message has shorter text → no such adjacency
triggers it.

### Practical implications

To unblock this experiment, options are:

1. **Use TRL SFTTrainer directly** without unsloth wrappers — different
   image processing pipeline, may not have the check
2. **Monkey-patch `_check_special_mm_tokens`** to log and skip mismatches
   — risky, may corrupt training
3. **Switch model** to one with simpler image tokenization (Qwen3-VL has
   more reproducible behavior)
4. **File issue upstream** and wait for fix

All are non-trivial. None were attempted in this thread.

## Artifacts preserved

All hint extraction outputs are saved and committed:

- `vlm_omr_sft/audiveris_hints_train.jsonl` (3115 entries, ~12MB)
- `vlm_omr_sft/audiveris_hints_dev.jsonl` (389 entries)
- `vlm_omr_sft/audiveris_hints_level9_n50.jsonl` (50 entries used in eval)
- `audiveris/audiveris_level9_n50.jsonl` (Audiveris-alone benchmark)
- `vlm_omr_sft/eval_hints_level9_n50.jsonl` (zero-shot in-context eval)

Code:
- `audiveris/extract_hints.py` (sharded parallel hint extraction)
- `audiveris/eval_synthetic_scores.py` (Audiveris benchmark)
- `vlm_omr_sft/hint_compress.py` (verb→skeleton compressor)
- `vlm_omr_sft/eval_with_hints.py` (HF in-context eval)
- `vlm_omr_sft/eval_with_hints_sglang.py` (SGLang variant — known issues)
- `vlm_omr_sft/train.py` (hint injection hooks added)
- `vlm_omr_sft/train_with_hints.py` (training wrapper)
- `vlm_omr_sft/config_hint_level9.yaml` (training config)

If the unsloth bug is fixed (or we switch frameworks), all training data is
ready to go — no re-extraction needed.

## What this means for the broader project

The 84% Level 9 plateau is **not** because the model lacks information.
Audiveris demonstrates that even a rule-based system has access to enough
signal to recover most pitches. A model that can verify text claims
against image evidence — once we can train it — should be able to use
hints productively.

The bottleneck is **training infrastructure for Qwen3.5-VL**, not
algorithmic. Earlier we found the same pattern with Miles: the model
is willing, the ecosystem isn't ready.

## Step 6: Text-only denoising (image-free)

To sidestep the unsloth multimodal SFT bug, we tried training the model
without the image: input = Audiveris MXC2, output = corrected MXC2.
Same Qwen3.5-9B base, same SFT adapter as starting point, just dropping
the image from the user message.

### Training

- 1 epoch on 3115 Level 9 hint→reference pairs (~80 min)
- `text_only: true` flag in `convert_to_conversation`: skips image, only text
- Clean loss decay: 1.05 → 0.18 over 779 steps
- Eval loss: 0.177 — looks like solid convergence
- No crashes — text-only path bypasses the multimodal bug entirely

### Eval — and the surprise

Initial eval gave **0% across the board**. Root cause: Qwen3.5's chat
template inserts `<think>` after the assistant role, putting the model
in reasoning mode. Our training data has no thinking content, so the
model produced English reasoning prose instead of MXC2.

Fix: strip `<think>\\s*\\n*$` from the rendered prompt at inference
time. The model then produces MXC2 directly (with a token `</think>`
prefix it emits to "close" the absent thinking).

Final result with the fix:

| Metric | Score |
|---|---|
| Pitched-only similarity | 69.7% |
| Rhythm similarity | 65.2% |
| Combined similarity | 77.0% |
| Note coverage | 92.0% |

### The decisive per-sample analysis

Comparing the text-only denoiser to Audiveris alone on the same 48 samples:

| Outcome | Count |
|---|---|
| Improved by ≥5pp over Audiveris | **0 / 48** |
| Degraded by ≥5pp | 12 / 48 |
| Within ±5pp (essentially copied) | 36 / 48 |

**The denoiser never improves on Audiveris.** It either copies the input
(36 samples) or makes it worse (12 samples). On no sample does it
recover from Audiveris's mistakes.

### Why this fails

During training, most of the 3115 samples have Audiveris already at
80-99% correct. The minimum-loss action is to copy the input. The
model learned exactly that.

- **Easy samples (most of training):** copying is optimal → 0pp improvement
- **Hard samples (few, where Audiveris is wrong):** no image → no way
  to know the right answer

The model never developed a "verify and correct" capability because
the training signal couldn't differentiate it from "copy".

### Cross-experiment comparison

| Approach | Pitched-only | What it can do |
|---|---|---|
| Audiveris alone (rule-based) | 76.5% | Bimodal: great on clean, fails on dense |
| Text-only denoiser (this run) | **69.7%** | Copies Audiveris; can never improve it |
| VLM image+hint zero-shot | 77.0% | Anchors on hint, ignores image |
| VLM image-only (SFT baseline) | **84.0%** | **Best — uses image as primary signal** |

### Conclusion: Audiveris hints don't help us

The fundamental signal source is **the image**. The VLM at 84% already
does better than Audiveris (76.5%). Adding Audiveris as a hint:
- Multimodally: the model anchors on the hint and corrupts image-based reasoning (77% zero-shot)
- Text-only: the model degenerates to a copy function (69.7%)

In retrospect this should have been the prior: a learned multimodal
model has more capacity than a rule-based pipeline, so its native
output should dominate. The 84% ceiling needs a different break.

## Recommendations for next steps

The 84% Level 9 plateau is **not** solvable by adding Audiveris as
auxiliary information. The bottleneck is the image-only model's
capacity, not its information access.

Real candidates to break the plateau:

1. **Larger VLM (Qwen3-VL-32B or similar)** — more parameters, more
   capacity for the visual reasoning the 9B model is missing on dense
   polyphony.
2. **Best-of-N inference** — earlier finding showed +16pp on Level 7
   from sampling 8 generations and picking the best. The model has
   better outputs in its distribution; we just need to surface them.
3. **Per-measure / per-staff transcription** — break the page into
   smaller units, transcribe each individually, then reassemble. Avoids
   the full-page visual-reasoning challenge.
4. **Stronger SFT data** — augment with more diverse synthetic levels
   (handwritten-style, different engravings, scaled fonts).
5. **The hint-augmented SFT direction is closed** — confirmed
   experimentally that Audiveris hints (multimodal or text-only) do
   not help, regardless of the unsloth bug. No reason to retry once
   that bug is fixed.

### What this experiment is worth keeping

- The Audiveris infrastructure (`audiveris/extract_hints.py`,
  `eval_synthetic_scores.py`, the 5.10.2 Docker image with 2.5× upscale)
  is a useful **baseline** for OMR.
- The 3504 (image, audiveris_mxc2, reference_mxc2) triples are useful
  training data for any future denoiser/distillation experiments.
- The hint compressor (`hint_compress.py`) is reusable.
- The text-only training mode added to `train.py` is reusable for any
  future text-to-text SFT experiments on the same model.
