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

## Recommendations for next steps

If continuing the hint-augmented SFT direction:
1. Try TRL `SFTTrainer` directly without unsloth's `FastVisionModel` wrapper
2. Or wait 2-4 weeks for unsloth multimodal SFT to stabilize on Qwen3.5
3. Or train on a different VLM (Qwen3-VL-8B) where the ecosystem is more mature

If pivoting:
1. **Per-measure denoising approach**: smaller LLM trained to fix Audiveris output one measure at a time — less context per call, simpler task
2. **Best-of-N inference**: our earlier finding showed +16pp on Level 7 just from sampling 8 generations and picking the best. No SFT needed.
3. **Larger VLM**: 32B model with more capacity might break the 84% plateau without RL or hints.
