# Model Comparison & Data Cleaning (2026-03-15 to 2026-03-18)

## Overview

Ran full 2-epoch training on 4 models with XML targets, compared LoRA ranks,
and iteratively cleaned the MusicXML training data. Key finding: **no model
produced actual pitched notes with XML targets** — the token budget was always
consumed by verbose XML preamble before reaching `<note>` elements.

## Model runs

### Ministral-3 3B (r=32)

**WandB run**: `x4clj3nz`

| Setting | Value |
|---|---|
| Model | `unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit` |
| LoRA r/alpha | 32/32 |
| Trainable params | ~67M (1.7%) |
| train_loss | 0.120 |

**Inference:** Degenerate repetition — instrument IDs like `I132222222222222...`
(hundreds of repeated digits) that consumed all tokens. Model never reached `<note>`.

### Qwen3.5-9B

**WandB run**: `gyyynoxw`

| Setting | Value |
|---|---|
| Model | `unsloth/Qwen3.5-9B-unsloth-bnb-4bit` |
| LoRA r/alpha | 16/16 |
| train_loss | 0.187 |

**Inference:** Best sample had 93 notes — but ALL were `<rest/>` with no pitches.
The model learned MusicXML structure but only generated rests, never actual pitched notes.

### DeepSeek-OCR-2 3B

**WandB run**: `4jplh7ef` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/4jplh7ef

| Setting | Value |
|---|---|
| Model | `unsloth/DeepSeek-OCR-2` (custom architecture) |
| LoRA r/alpha | 16/16 |
| train_loss / eval_loss | 0.312 / 0.197 |
| Runtime | 1h 44m |

Required a **separate training script** (`train_deepseek_ocr2.py`) due to:
- Custom `<|User|>`/`<|Assistant|>` conversation format
- `DeepSeekOCR2DataCollator` (not `UnslothVisionDataCollator`)
- `snapshot_download` + trust_remote_code required
- Patches needed: `DeepseekV2MoE` → `DeepseekV2Moe` (transformers 5.x rename),
  `logits.float()` → keep bfloat16 (prevents 50 GB OOM)
- `model.generate()` instead of `model.chat()` for inference
- Images must be explicitly moved to CUDA

**Inference:** Best sample had 29 notes — all rests, no pitches. Instrument IDs
also showed degenerate repetition (`I1e0e5e0e0e0e0...`).

### Ministral-3 3B (r=8)

**WandB run**: `fsnysrn7` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/fsnysrn7

| Setting | Value |
|---|---|
| Model | Ministral-3 3B |
| LoRA r/alpha | 8/8 |
| Trainable params | 16.9M (0.44%) |
| train_loss / eval_loss | 0.173 / 0.158 |

**Inference:** Instrument IDs still repeated but shorter and terminated properly.
Model continued past IDs to valid XML structure. Best sample: 35 notes, 35 with
`<pitch>` — the **only XML-target model to produce any pitched notes**.

## LoRA rank comparison: r=32 vs r=8

| | r=32 | r=8 |
|---|---|---|
| Trainable params | ~67M (1.7%) | 16.9M (0.44%) |
| Degenerate repetition | Infinite loops | Shorter, terminates |
| Pitched notes | 0 | 35 (in WandB examples) |

Reducing rank eliminated catastrophic repetition. However, both ranks failed to
generate notes during post-training inference (2048 token budget consumed by preamble).

## Data cleaning iterations

### Iteration 1: Strip non-musical metadata

Removed from training targets:
- `<score-instrument>` blocks (redundant with `<part-name>`)
- `<midi-instrument>` blocks (playback only)
- `<midi-device>` elements (playback only)

### Iteration 2: Strip invisible elements

- XML comments (`<!--=== Measure 1 ===-->`) — ~60 per page, pure token waste
- `<sound tempo="92"/>` — invisible numeric metadata
- Empty `<direction>` blocks (only `<sound>` or empty `<words/>`)
- `implicit="no"` attribute (always "no")

**Kept:** stems, beams, lyrics, accidentals, barlines, directions with visible text.

### Iteration 3: Keep composer/lyricist

Changed `<identification>` stripping from blanket removal to selective:
- **Keep:** `<creator type="composer">`, `<creator type="lyricist">` (visible on page)
- **Strip:** `<encoding>`, `<rights>`, `<creator type="arranger">` (IMSLP metadata)

### Ministral-3 r=8, cleaned data

**WandB run**: `3g5cjx1q` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/3g5cjx1q

| Metric | Before cleaning | After cleaning |
|---|---|---|
| train_loss | 0.173 | **0.065** |
| eval_loss | 0.158 | **0.048** |
| Reaches `<note>` in inference | No | No |

Loss dropped 3x, but inference still consumed all tokens on XML preamble. The model
now hallucinated `<print><system-layout>` blocks instead of instrument IDs.

## Token budget analysis

Measured on 200 lieder pages:

| | Cleaned XML | MXC (later) |
|---|---|---|
| Median ~tokens | ~14,363 | ~1,192 |
| Fit in 4096 | 10% | ~95% |

**Only ~10% of pages fit in 4096 tokens even after aggressive cleaning.** The typical
page is ~14K tokens — models only ever see the first 4096 tokens during training,
which is predominantly preamble.

## Conclusions

1. **XML format is the bottleneck** — no amount of model tuning or data cleaning
   enables note generation when 90% of pages are truncated before the first note.

2. **Data cleaning reduces loss** but doesn't solve the structural problem.
   Cleaned XML: eval_loss 0.048 (best), but still no notes in inference.

3. **Smaller LoRA rank is better** for this task — r=8 avoids degenerate repetition
   that r=32 suffers from, while producing equivalent or better results.

4. **DeepSeek-OCR-2** required significant engineering (separate script, multiple
   patches) but did not outperform simpler models on this task.

5. **Qwen3.5-9B** (largest model tested) generated more structured output but still
   only rests — model size alone doesn't solve the token budget problem.

These findings directly motivated the development of MXC format (see
`2026-03-19-mxc-experiments.md`).
