# MXC Format Experiments (2026-03-19)

## Background

Previous XML-target runs (Gemma-3 4B, Ministral-3 3B, Qwen3.5-9B, DeepSeek-OCR-2)
all failed to generate actual `<note>` elements within the 2048 inference token budget.
The root cause: MusicXML is ~14K tokens per page — the model spends all inference tokens
on preamble (part-list, instrument IDs, MIDI metadata, comments) before reaching notes.

We developed **MXC (MusicXML Compact)**, a line-based encoding achieving 12x compression,
to fit full pages within the context window. See `MXC.md` for format spec.

## Experiment 1: Ministral-3 r=8, MXC, 2 epochs

**WandB run**: `r48244y4` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/r48244y4

| Setting | Value |
|---|---|
| Model | Ministral-3 3B (QLoRA 4-bit, r=8) |
| Target format | MXC |
| Train rows | 2,986 lieder |
| Epochs | 2 |
| train_loss / eval_loss | 0.142 / 0.109 |
| Runtime | 1h 52m |

### Key findings

**First time any model generated actual pitched notes.**

| Step | Samples w/ pitched notes | Max pitched | Avg pitched |
|------|-------------------------|-------------|-------------|
| 0 | 0/4 | 0 | 0.0 |
| 100 | 1/4 | 120 | 30.0 |
| 500 | 1/4 | 106 | 26.5 |
| 1000 | 3/4 | 84 | 43.0 |
| 1400 | 3/4 | 79 | 44.0 |

What the model learned well:
- MXC format syntax (headers, parts, measures, note lines)
- Lyrics with bilingual syllabic encoding (`L1:s:Mein L2:s:My`)
- Rhythm variety (quarters, eighths, dotted, beams)
- Voice/staff assignments (`v=2 st=2`)
- Page metadata (title, composer, lyricist, tempo text)

What it did NOT learn:
- **Pitch variety** — all predictions use 1 unique pitch per sample (all A4, all Eb4, etc.)
- **Piano part** — P2 mostly rests; only voice part gets notes

### Eval loss trend (still dropping)

| Epoch | eval_loss |
|-------|-----------|
| 0.33 | 0.136 |
| 0.67 | 0.120 |
| 1.00 | 0.114 |
| 1.34 | 0.112 |
| 1.67 | 0.109 |

Not yet plateaued — motivated a 4-epoch follow-up.

## Experiment 2: Ministral-3 r=8, MXC, 4 epochs

**WandB run**: `jkvlkq96` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/jkvlkq96

| Setting | Value |
|---|---|
| Model | Ministral-3 3B (QLoRA 4-bit, r=8) |
| Target format | MXC |
| Train rows | 2,986 lieder |
| Epochs | 4 |
| train_loss / eval_loss | 0.106 / 0.117 (overfit) |
| Runtime | 2h 51m |

### Eval loss — overfitting after epoch 2

| Epoch | eval_loss |
|-------|-----------|
| 0.67 | 0.123 |
| 1.34 | 0.114 |
| **2.01** | **0.111** ← best |
| 2.68 | 0.111 |
| 3.35 | 0.117 ← rising |

### Inference quality improvement

| Step | Samples w/ notes | Max pitched | Avg pitched |
|------|-----------------|-------------|-------------|
| 200 | 2/4 | 117 | 58.0 |
| 1000 | 3/4 | 85 | 36.5 |
| 1800 | 4/4 | 85 | 46.5 |
| 2200 | 4/4 | 109 | 74.8 |
| 2400 | 4/4 | 103 | 78.0 |
| 2800 | 4/4 | 97 | 76.0 |

By step 1800+ (epoch ~2.4), **all 4 samples produce pitched notes** (vs 3/4 at 2 epochs).
Average notes per sample increased from 44 to 78.

### Pitch variety — still monotone

Despite more training, pitches remain mostly monotone (1-2 unique pitches per sample).
The model generates varied rhythm and lyrics but maps everything to a single pitch.
Some predictions at early steps showed hallucinated octaves (A10, B16, C17) — not real variety.

## Comparison: XML vs MXC across all runs

| Run | Format | train_loss | eval_loss | Pitched notes? | Pitch variety? |
|-----|--------|-----------|-----------|---------------|---------------|
| Gemma-3 4B, 2ep | XML | — | — | No | — |
| Ministral-3 r=32, 2ep | XML | 0.120 | — | No (degenerate IDs) | — |
| Ministral-3 r=8, 2ep | XML | 0.173 | 0.158 | No (midi blocks) | — |
| Ministral-3 r=8 clean, 2ep | XML | 0.065 | 0.048 | No (layout halluc.) | — |
| Qwen3.5-9B, 2ep | XML | 0.187 | — | Rests only | — |
| DeepSeek-OCR-2, 2ep | XML | 0.312 | 0.197 | Rests only | — |
| **Ministral-3 r=8, MXC, 2ep** | **MXC** | **0.142** | **0.109** | **Yes (up to 94)** | No (monotone) |
| **Ministral-3 r=8, MXC, 4ep** | **MXC** | **0.106** | **0.117** | **Yes (up to 109)** | No (monotone) |
| **Ministral-3 r=32, MXC, 2ep** | **MXC** | **0.121** | **0.109** | **Yes (up to 91)** | Transient only |

## Experiment 3: Ministral-3 r=32, MXC, 2 epochs

**WandB run**: `nr2suw9c` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/nr2suw9c

| Setting | Value |
|---|---|
| Model | Ministral-3 3B (QLoRA 4-bit, r=32) |
| Target format | MXC |
| Trainable params | 67.5M (1.72%) — 4× more than r=8 |
| train_loss / eval_loss | 0.121 / 0.109 |
| Runtime | 1h 50m |

### Eval loss — nearly identical to r=8

| Epoch | r=8 eval_loss | r=32 eval_loss |
|-------|--------------|----------------|
| 0.33 | 0.136 | 0.133 |
| 0.67 | 0.120 | 0.120 |
| 1.00 | 0.114 | 0.114 |
| 1.34 | 0.112 | 0.111 |
| 1.67 | 0.109 | 0.109 |

### Pitch variety — transient then collapsed

Mid-training (steps 800-1000) showed up to **10 unique real pitches** with a
coherent descending scale pattern (C#5 → B4 → A4 → G4 → F#4 → E4 → D4 → C#4).
This was the first time any model produced varied pitches.

However, by step 1200+ the pitch variety **collapsed back to monotone** (1 unique
pitch per sample). The model briefly explored pitch diversity but converged to the
single-pitch solution that minimizes loss.

### Key finding

Higher LoRA rank does NOT solve pitch variety. The identical eval_loss curves for
r=8 and r=32 confirm that adapter capacity is not the bottleneck — the model
converges to the same solution regardless of rank. The pitch problem is upstream,
likely in the vision encoder's ability to discriminate note positions on staff lines.

## Vision encoder analysis

Training images are 835×1181 pixels (uniform). Vision encoder properties:

| | Ministral-3 (Pixtral) | Qwen3.5-9B |
|---|---|---|
| Patch size | 14px | 16px |
| Spatial merge | none | 2×2 |
| Effective resolution | 14×14 per token | 32×32 per token |
| Vision tokens per page | ~5,040 | ~962 |

Ministral-3 has 5× more visual detail per image, but still cannot discriminate
pitch. This suggests the bottleneck may not be resolution but the model's ability
to learn the position→pitch mapping (a fine-grained spatial reasoning task).

## Conclusions

1. **MXC format is essential** — the only way to get the model to generate actual notes.
   XML format never reaches note content within inference token budget.

2. **Ministral-3 3B can learn rhythm, lyrics, and structure but NOT pitch** from images.
   This holds across r=8 and r=32. The model converges to a single-pitch solution
   regardless of adapter capacity.

3. **2 epochs is optimal** for Ministral-3 on this data. Both r=8 and r=32 produce
   nearly identical eval_loss curves. 4 epochs overfits.

4. **LoRA rank is not the bottleneck** — r=8 and r=32 reach the same eval_loss (0.109)
   and the same pitch behavior (monotone).

5. **The pitch problem is upstream** — either the vision encoder cannot provide
   sufficient spatial signal, or the 3B language model lacks capacity to learn
   the position→pitch mapping from ~3K training examples.

## Experiment 4: Qwen3.5-9B, MXC, 2 epochs

**WandB run**: `3h8sdtea` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/3h8sdtea

| Setting | Value |
|---|---|
| Model | Qwen3.5-9B (QLoRA 4-bit, r=16) |
| Target format | MXC |
| Trainable params | 51.0M (0.54%) |
| Vision tokens per image | 962 (16px patches, 2×2 spatial merge, full 835×1181 resolution) |
| train_loss / eval_loss | 0.174 / 0.154 |
| Runtime | 4h 13m |

### Eval loss — still dropping

| Epoch | eval_loss |
|-------|-----------|
| 0.33 | 0.219 |
| 0.67 | 0.186 |
| 1.00 | 0.168 |
| 1.34 | 0.159 |
| 1.67 | **0.154** |

Not plateaued — more epochs should help.

### Pitch accuracy — breakthrough

At step 1400 (end of training), comparing predicted vs reference pitches:

| Sample | Pred/Ref notes | Pitch accuracy | Type accuracy | Unique pitches |
|---|---|---|---|---|
| lc6211535 | 38/406 | **45%** | 79% | 9 |
| lc5800427 | 74/277 | **62%** | 70% | 10 |
| lc30321734 | 68/82 | **78%** | 87% | 20 |
| lc6482032 | 95/409 | 8% | 74% | 15 |

Example — `lc5800427` first 10 notes (step 1000):

```
Pred: E4 A4 B4 C5 C5 B4  D5 C5 B4 A4
Ref:  E4 A4 B4 C5 C5 B4  D5 C5 B4 A4   ← exact match
```

The model learned to **read pitch from staff positions** — a qualitative leap.

### Inference quality progression

| Step | Samples w/ notes | Max unique pitches | Avg unique |
|------|-----------------|-------------------|-----------|
| 0 | 0/4 | 0 | 0.0 |
| 100 | 3/4 | 10 | 5.2 |
| 500 | 3/4 | 22 | 11.2 |
| 1000 | 4/4 | 17 | 12.8 |
| 1400 | 4/4 | 20 | 13.5 |

By step 100 (7% of training), the model already produces pitched notes with variety.
Pitch variety stabilizes around 10-20 unique pitches per sample from step 500 onward.

### What the model gets right

- **Pitch**: 45-78% accuracy (sample-dependent), reading from image
- **Rhythm/type**: 70-87% accuracy — quarters, eighths, dotted, correctly differentiated
- **Lyrics**: bilingual (`L1:s:My L2:s:Mein`), correct syllabic encoding
- **Accidentals**: `Cb5`, `Gb4`, `F#4 acc=sharp` used in context
- **Articulations**: fermatas, slurs, beams on correct notes
- **Multi-voice**: `v=1 st=1` assignments for piano staves

### What needs improvement

- **Note coverage**: only 38-95 notes predicted vs 82-409 in reference (model stops early)
- **One sample poor** (lc6482032): 8% pitch accuracy — may be a harder/unusual page
- **Eval loss not converged**: 0.154 and still dropping — more training likely helps

## Cross-model comparison (all MXC runs)

| Model | Params | Vision tokens | eval_loss | Pitch variety | Pitch accuracy |
|---|---|---|---|---|---|
| Ministral-3 r=8 | 3.9B | 1,858 | 0.109 | 1 (monotone) | ~0% |
| Ministral-3 r=32 | 3.9B | 1,858 | 0.109 | 1-2 | ~0% |
| Ministral-3 r=8 4ep | 3.9B | 1,858 | 0.117 (overfit) | 1 (monotone) | ~0% |
| **Qwen3.5-9B r=16** | **9.5B** | **962** | **0.154** | **9-20** | **45-78%** |

### Token budget analysis (verified from source code)

Image + instruction tokens, measured with real training data:

| Model | Image+instruction tokens | Room for MXC (max_length=4096) |
|---|---|---|
| Ministral-3 | 1,869 | 2,227 |
| Qwen3.5-9B | 975 | 3,121 |

Both models fit the median MXC page (~1,200 tokens). Longer pages are truncated
from the right (verified in unsloth-zoo source: `_truncate_by_side` with
`padding_side="right"` → `slice(0, max_len)`).

## Experiment 5: Qwen3.5-9B, MXC, continued epochs 3-4 (from checkpoint)

**WandB run**: `20xw5s9a` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/20xw5s9a

| Setting | Value |
|---|---|
| Model | Qwen3.5-9B (resumed from epoch 2 adapter via `--resume-adapter`) |
| Fresh scheduler | Yes (cosine LR, 2 epochs from scratch) |
| train_loss / eval_loss | 0.112 / 0.150 |
| Runtime | 4h 16m |

### Eval loss — still dropping, not overfitting

Combined with the initial run (effective total 4 epochs):

| Effective epoch | eval_loss | Source |
|-----------------|-----------|--------|
| 0.33 | 0.219 | run 1 |
| 0.67 | 0.186 | run 1 |
| 1.00 | 0.168 | run 1 |
| 1.34 | 0.159 | run 1 |
| 1.67 | 0.154 | run 1 |
| 2.33 | 0.170 | run 2 (fresh scheduler warmup) |
| 2.67 | 0.163 | run 2 |
| 3.00 | 0.154 | run 2 |
| 3.34 | 0.154 | run 2 |
| 3.67 | **0.150** | run 2 |

Note: eval_loss initially rises in run 2 due to fresh scheduler warmup, then
recovers and improves to 0.150 — a new best.

### Pitch accuracy — significant improvement

Comparing best results across epochs:

| Sample | Pitch acc (ep 2) | Pitch acc (ep 3-4 best) | Notes predicted |
|---|---|---|---|
| lc6211535 | 45% | **95%** (step 1300) | 37/406 |
| lc5800427 | 62% | **66%** (step 1000) | 74/277 |
| lc30321734 | 78% | **83%** (step 1000) | 58/82 |
| lc6482032 | 8% | **14%** (step 1300) | 83/409 |

### Page complexity analysis

lc6211535 (the 95% accuracy sample) is NOT a simple page — it has 3 staves per
system (Voice + Piano RH + Piano LH), 82 measures, 406 total notes, dense piano
passages with running eighths/sixteenths, triplets, clef changes, and bilingual
lyrics. The voice enters at bar 34 after 33 bars of rest.

The model correctly outputs rest measures for bars 1-33, then reads the vocal
entry pitches with 95% accuracy. The main limitation is **output length** (2048
inference tokens), not pitch accuracy — the model runs out of tokens before
covering the piano part.

### Note coverage limitation

The model predicts 36-95 notes out of 82-409 reference notes. This is primarily
an inference token budget issue:
- `inference_max_new_tokens` was set to 2048
- This is an **inference-time setting**, not a model or training constraint
- Increasing to 4096 should roughly double note coverage

## Experiment 6: Qwen3.5-9B, MXC, continued epochs 5-6, inference 4096 tokens

**WandB run**: `mzcdnaor` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/mzcdnaor

| Setting | Value |
|---|---|
| Model | Qwen3.5-9B (resumed from epoch 4 adapter) |
| inference_max_new_tokens | 4096 (doubled from 2048) |
| train_loss / eval_loss | 0.086 / 0.159 |
| Runtime | 5h 13m |

### Eval loss — plateaued

| Effective epoch | eval_loss |
|---|---|
| 5.0 | 0.164 |
| 5.3 | 0.161 |
| 5.7 | 0.157 |
| 6.0 | 0.163 |
| 6.3 | 0.159 |

Oscillating around 0.157-0.163 — no longer improving. Model has converged on
3K lieder samples.

### Standardized metrics (eval_mxc.py)

Final step comparison across all Qwen3.5-9B runs:

| Effective epochs | Avg pitch sim | Avg rhythm | Avg combined | Coverage | Unique pitches |
|---|---|---|---|---|---|
| 2 | 31% | 32% | 38% | 34% | 14.5 |
| 4 | 32% | 30% | 37% | 33% | 13.8 |
| **6 (4096 tok)** | **33%** | **29%** | **38%** | **54%** | **18.8** |

Best per-sample result: lc30321734 at **75% pitch similarity** (consistent across
epochs 2-6).

### Effect of doubling inference tokens

| Metric | 2048 tokens | 4096 tokens |
|---|---|---|
| Avg coverage | 34% | **54%** |
| Avg predicted events | 103 | **191** |
| Max consecutive match | 44 | **105** |

Doubling inference tokens increased coverage from 34% to 54% without hurting
accuracy — confirming output length was a bottleneck.

## Experiment 7: Qwen3.5-9B r=32, MXC, 4 epochs

**WandB run**: `6uiezgde` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/6uiezgde

| Setting | Value |
|---|---|
| Model | Qwen3.5-9B (QLoRA 4-bit, r=32) |
| Trainable params | 102M (1.07%) — 2× the r=16 runs |
| Target format | MXC |
| Epochs | 4 |
| inference_max_new_tokens | 4096 |
| n_examples | 10 |
| train_loss / eval_loss | 0.131 / 0.154 |
| Runtime | 10h 48m |

### Eval loss

| Epoch | eval_loss |
|---|---|
| 0.67 | 0.192 |
| 1.34 | 0.166 |
| 2.01 | 0.154 |
| **2.68** | **0.149** ← best |
| 3.35 | 0.154 ← overfitting |

Best at epoch 2.7. Same pattern as r=16 — overfitting starts around epoch 3.

### Standardized metrics (10 samples, eval_mxc.py)

Best results at step 2400-2800 (epoch 3.2-3.7):

| Step (epoch) | Avg pitch | Avg rhythm | Avg combined | Coverage | Max pitch | Max match |
|---|---|---|---|---|---|---|
| 200 (0.3) | 15% | 27% | 15% | 57% | 26% | 59 |
| 800 (1.1) | 22% | 29% | 25% | 64% | 63% | 59 |
| 1600 (2.1) | 31% | 34% | 33% | 80% | 66% | 123 |
| 1800 (2.4) | 31% | 45% | 33% | 75% | 72% | 86 |
| **2400 (3.2)** | **36%** | **43%** | **37%** | 69% | **83%** | **121** |
| 2800 (3.7) | 34% | 46% | 38% | 73% | 73% | 121 |

### r=16 vs r=32 vs r=64 comparison

| Metric | r=16 | r=32 | r=64 |
|---|---|---|---|
| Trainable params | 51M (0.54%) | 102M (1.07%) | 204M (2.12%) |
| Best eval_loss | 0.154 | **0.149** | 0.146 |
| Avg pitch similarity | 33% | **36%** | 32% |
| Avg pitched-only sim | — | **35%** | — |
| Avg rhythm similarity | 32% | **46%** | 46% |
| Max pitch similarity | 75% | **83%** | 81% |
| Max consecutive match | 105 | **123** | 86 |
| Avg unique pitches | 18.8 | **27.4** | 25.6 |

### Metric correction (2026-03-22)

The original `eval_mxc.py` `pitch_similarity` metric included rests in the
sequence comparison. A sample with many rests and no pitched notes could score
non-zero from matching rest events. Added `pitched_only_similarity` that
excludes rests.

On the r=32 best step (2400), the corrected metric shows:
- Avg pitch similarity (all events): 36%
- Avg pitched-only similarity: **35%** (close — rests weren't inflating much overall)
- Best sample lc5015573: 88% positional pitch accuracy
- Worst sample lc6211535: 0% pitched-only (all rests in prediction, correctly reported)

### Key findings

- **r=32 is the sweet spot** — best pitch (36%), best rhythm (46%), best max pitch (83%)
- **r=64 does NOT improve over r=32** — despite lower eval_loss (0.146 vs 0.149),
  inference metrics are slightly worse on pitch (32% vs 36%)
- **LoRA rank is now fully explored** (r=16, 32, 64) — the bottleneck is training data

## Conclusions

1. **Qwen3.5-9B reads pitch from images** — up to 84% pitched-only similarity,
   88% positional pitch accuracy on best samples, 35% average across 10 pages.

2. **LoRA r=32 is optimal** — r=64 overfits without improving inference quality.
   On Ministral-3, rank made no difference (couldn't learn pitch). On Qwen3.5-9B,
   r=32 provides the right capacity/regularization balance.

3. **MXC format is essential** — without it, no model reaches note content.

4. **Optimal training: 2-3 epochs** — all ranks overfit after epoch 3 on 3K samples.

5. **Model has converged at ~35% avg pitched-only similarity** on 3K lieder.
   The bottleneck is training data, not model capacity or training duration.

## What the model gets right vs wrong

**Reliably correct:**
- MXC format syntax (never produces invalid MXC)
- Part structure (Voice + Piano, correct IDs)
- Key signatures (8/10 correct)
- Lyrics (bilingual, correct syllabic encoding)
- Pitch on simpler passages (up to 88% positional accuracy)
- Rhythm/note types (46% similarity at r=32)

**Struggles with:**
- Dense piano passages with complex polyphony
- Very long pages (coverage drops on pages with 400+ notes)
- Time signature (7-8/10 correct)
- Pages where voice has many rest measures (model runs out of tokens before notes start)

## Experiment 8: Model comparison — 4 models × MXC × 3 epochs

All runs: r=32, MXC targets, 3 epochs, 10 inference examples, 4096 inference tokens.

### Gemma-3 4B

**WandB run**: `3wr70bas`

| Setting | Value |
|---|---|
| Model | gemma-3-4b-pt (SigLIP vision) |
| Trainable params | 77M (1.76%) |
| Best eval_loss | 0.216 (step 1500) |
| Runtime | 10h 10m |

**Result: 1% pitched-only similarity.** Cannot learn pitch — same as Ministral-3.
4B models lack capacity for pitch discrimination regardless of architecture.

### Qwen3-VL 8B

**WandB run**: `9bh3nzv9`

| Setting | Value |
|---|---|
| Model | Qwen3-VL-8B-Instruct (same VL arch as Qwen3.5-9B) |
| Trainable params | ~102M |
| Best eval_loss | 0.166 (step 2000) |
| Runtime | 5h 43m |

**Result: 16% pitched-only similarity, 24% positional.** Learns some pitch but
significantly worse than Qwen3.5-9B (35%). Surprising given similar size — the
Qwen3.5 architecture improvements matter.

### Qwen3-VL 32B

**WandB run**: `bnjb5ry5`

| Setting | Value |
|---|---|
| Model | Qwen3-VL-32B-Instruct (largest model tested) |
| Best eval_loss | 0.147 (step 1500) |
| Runtime | 14h |

**Result: 23% pitched-only similarity.** Does not outperform the 3.5× smaller
Qwen3.5-9B (35%). Architecture matters more than scale.

### DeepSeek-OCR-2

**WandB run**: `22zgvj58`

| Setting | Value |
|---|---|
| Model | DeepSeek-OCR-2 3B (OCR-specialized) |
| Best eval_loss | 0.315 |
| Runtime | 2h |

**Result: 0% pitched-only similarity.** 3B model cannot learn pitch, consistent
with Ministral-3 and Gemma-3 results. Note: initial eval showed 0% due to a bug
where the WandB callback logged XML references instead of MXC — fixed by adding
auto-conversion in eval_mxc.py. Re-evaluation confirmed genuinely 0%.

### Cross-model comparison (all MXC runs, best step, 10 samples)

| Model | Size | Pitched-only sim | Note-type sim | Rhythm | eval_loss |
|---|---|---|---|---|---|
| DeepSeek-OCR-2 | 3B | 0% | — | 4% | 0.315 |
| Gemma-3 4B | 4.4B | 1% | — | 11% | 0.216 |
| Ministral-3 r=32 | 3.9B | ~0% | — | 10% | 0.109 |
| Qwen3-VL 8B | 8B | 16% | — | 23% | 0.166 |
| Qwen3-VL 32B | 32B | 23% | — | 31% | 0.147 |
| **Qwen3.5-9B r=32** | **9.5B** | **35%** | **42%** | **46%** | **0.149** |

### Key insight: model architecture matters as much as size

Qwen3-VL 8B (16% pitch) underperforms Qwen3.5-9B (35% pitch) despite similar
parameter count. Qwen3-VL 32B (23%) underperforms the 3.5× smaller Qwen3.5-9B.
The Qwen3.5 architecture revisions (unified early fusion) matter more than scale.

## Experiment 9: OLiMPiC dataset — Qwen3.5-9B r=32, 3 epochs

**WandB run**: `af6jitjq`

| Setting | Value |
|---|---|
| Model | Qwen3.5-9B r=32 (same as best OpenScore recipe) |
| Dataset | zzsi/olimpic (15,014 system-level piano crops) |
| Target format | MXC |
| Best eval_loss | 0.031 (step 11000) |
| Runtime | 22h 35m |

### Comparison: OpenScore (full pages) vs OLiMPiC (system crops)

| Metric | OpenScore (3K pages) | OLiMPiC (15K systems) |
|---|---|---|
| eval_loss | 0.149 | **0.031** |
| Pitched-only sim | 35% | **39%** |
| **Note-type sim** | **42%** | **65%** |
| Rhythm sim (raw duration) | 43% | 63% |
| Combined sim (pitch+type) | 37% | 28% |
| Coverage | 69% | 120% |
| Max pitch sim | 83% | 85% |
| Max longest match | 121 events | 15 events |
| Key sig | 8/10 | 5/10 |
| Time sig | 7/10 | **10/10** |

### What note-type similarity reveals

The new `note_type_similarity` metric compares note types (`e`/`q`/`s`/`h`/`w`)
independently of raw duration values, which differ due to `divisions`
normalization across datasets.

- OLiMPiC: 65% note-type vs 63% rhythm — the model learned correct note types;
  duration values are consistent but use a different `divisions` base than
  the ground truth
- OpenScore: 42% note-type vs 43% rhythm — both lower, meaning full pages are
  harder for rhythm learning too

### Pitch error analysis

Typical errors on OLiMPiC predictions (step 9000, 10 samples):

| Error type | Frequency | Explanation |
|---|---|---|
| Off by a third/fourth (3-5 semitones) | 52% | One staff line off |
| Off by 1-2 semitones | 18% | Accidental errors |
| Off by an octave | 12% | Right note name, wrong register |
| Off by a fifth | 11% | Two staff lines off |

Errors skew downward (model predicts lower than reference). G#3 appears as a
frequent "default" prediction when the model is uncertain.

### Why combined similarity is low

Combined similarity (28%) is much lower than pitch (39%) and note-type (65%)
separately. This means pitch and rhythm errors happen on **different notes** —
the model gets pitch right on some notes and rhythm right on others, but rarely
both together.

### Manual inspection findings

Side-by-side comparison of predicted vs reference MXC revealed systematic issues
that aggregate metrics partially masked:

1. **`divisions` mismatch**: Model predicts `div=12` when reference has `div=2`,
   causing all duration values to differ by a constant factor (6x). Rhythms are
   semantically correct but string-compare fails.
2. **Consistent pitch shifts**: Model predicts Eb4 where reference has G4 — a
   persistent 4-semitone error on certain samples.
3. **Note-type confusion**: Model predicts 16th notes (with double beams) where
   reference has eighths (single beam) — misreading rhythm subdivision.
4. **Hallucinated repetition**: Some samples show the model repeating the same
   chord (e.g., G#4+E5) with identical durations, clearly not reading the image.

## Conclusions (updated 2026-03-26)

1. **Qwen3.5-9B r=32 is the best model** across both datasets — 35% on OpenScore
   pages, 39% on OLiMPiC system crops.

2. **System-level crops are easier** — 5× lower eval_loss, +23% note-type
   accuracy, perfect time signature detection. But pitch improvement is modest
   (+4%), suggesting pitch reading difficulty is not primarily about image
   complexity.

3. **Note-type accuracy (65%) is much higher than pitch accuracy (39%)** on
   OLiMPiC — the model learns rhythm better than pitch from images.

4. **Combined accuracy is poor (28%)** — the model rarely gets pitch AND rhythm
   right on the same note. This is a fundamental limitation of the current
   approach.

5. **More SFT on the same data will not reach near-perfect accuracy.** Eval loss
   plateaued, pitch accuracy is noisy rather than trending up, and systematic
   errors (divisions mismatch, pitch shifts, hallucinated repetition) are
   structural, not convergence issues.

6. **Architecture matters more than model size** — Qwen3.5-9B beats Qwen3-VL-32B.

## Next steps

## Experiment 10: Synthetic data diagnostic (Levels 1-4)

**Dataset**: `zzsi/synthetic-scores` — 1,000 samples per level, 800/100/100 splits.
**Model**: Qwen3.5-9B r=32 MXC, 3 epochs per level.

### Difficulty ladder results

| Level | Content | Pitch accuracy | Note-type | eval_loss |
|---|---|---|---|---|
| 1 | Single staff, C major, quarter notes | **100%** | **100%** | 0.0003 |
| 2 | + varied rhythms (half, eighth, dotted) | **100%** | **100%** | 0.0008 |
| 3 | + key signatures + accidentals | **95%** | **100%** | 0.0032 |
| 4 | + rests + ties | **95-99%** | **100%** | 0.0296 |

### Key finding: accidentals are the accuracy bottleneck

Levels 1-2 achieve perfect accuracy. Level 3 introduces key signatures and
accidentals (♯♭♮), dropping pitch accuracy to 95%. Level 4 adds rests and ties
but doesn't drop further — the model handles rests/ties easily. The 5% error
persists from accidentals.

### Error analysis (Level 3, step 400+, 3165 notes)

175 errors total (5.5%). The errors are **not random**:

| Error pattern | Count | % of errors |
|---|---|---|
| D → D# | 90 | 51% |
| D5 → D#5 | 55 | 31% |
| A → Ab | 25 | 14% |
| D → Dn0 | 5 | 3% |

**All D→D# errors occur in key=3 (A major: F#, C#, G#).** The model sees 3 sharps
in the key signature and over-generalizes — it applies sharps to D as well, but D
is natural in A major. The model hasn't learned which specific notes each key
signature sharp/flat applies to (F, C, G in A major, not D).

This is a **key signature interpretation error**, not a visual reading error. The
model reads note positions correctly (the D is on the right line) but misapplies
the key signature context.

### Levels 5-7 results

| Level | Content | Pitch | Note-type | eval_loss |
|---|---|---|---|---|
| 5 | Grand staff (treble+bass) | **94%** | **100%** | 0.0016 |
| 6 | + chords in right hand | **97%** | **90%** | 0.0019 |
| 7 | Voice+piano+lyrics | **95-98%** | **99%** | ~0.002 |

- **Level 5**: Adding bass clef barely drops accuracy (94% vs 95% at Level 3).
  Multi-staff reading is not a bottleneck.
- **Level 6**: Chords cause the first note-type accuracy drop (90%). Reading
  multiple stacked noteheads and outputting `+N` chord markers is harder.
- **Level 7**: 95-98% pitch, 99% note-type. Appears high, but this is misleading.

### Level 7 is too easy compared to real openscore

| | Level 7 synthetic | Real openscore lieder |
|---|---|---|
| MXC length | 3,361 chars | **15,026 chars** (4.5× longer) |
| Notes | 181 | 291 |
| Chords | **0** | **109** (38% of notes) |
| Rests | 7 | 37 |
| Piano | Monophonic per hand | Dense chords + polyphony |

Level 7's piano parts are monophonic — no chords. Real lieder piano accompaniment
is full of chords (109 chord notes vs 0). Level 7 is essentially Level 5 + lyrics,
which explains why it didn't drop. The hard parts of real music (dense piano
chords, complex voicing, long pages) are missing.

### Implications for real music transcription

The 35% pitch accuracy on openscore lieder is driven by:
1. **Key signature misapplication** (~5% error on simple scores)
2. **Dense piano chords** (Level 6 shows 10% note-type drop even with simple chords;
   real lieder has much denser chord textures)
3. **Page density** (real pages are 4.5× longer with more notes per system)
4. **Complex rhythms** not in synthetic data (triplets, syncopation, grace notes, ornaments)

Each factor compounds. The synthetic diagnostic confirms the model CAN read pitch
(100% on simple scores) but real music complexity degrades accuracy through
multiple interacting factors, not a single bottleneck.
to 35% (full lieder pages).

## Experiment 11: Level 7 with 5K samples — Qwen3.5-9B r=32, 3 epochs

**WandB run**: `rdt13jav` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/rdt13jav

| Setting | Value |
|---|---|
| Model | Qwen3.5-9B r=32 MXC |
| Dataset | zzsi/synthetic-scores level7, 5K samples (4K train / 500 dev / 500 test) |
| Level 7 content | Voice + piano (chords in RH) + lyrics, 24 measures |
| Epochs | 3 (2979 steps) |
| Best eval_loss | 0.000425 (epoch 2.4) |
| Runtime | ~28h |

### Eval loss — still decreasing at end

| Epoch | eval_loss |
|---|---|
| 0.2 | 0.004633 |
| 0.8 | 0.000792 |
| 1.4 | 0.000730 |
| 2.0 | 0.000488 |
| 2.2 | 0.000447 |
| 2.4 | **0.000425** |

Val loss did not plateau — the model could benefit from more epochs. However,
training loss was near zero (~1e-6), suggesting the model memorized the training
set while still generalizing to the dev set.

### Accuracy evaluation (50 test samples, eval_run.py)

| Metric | Value |
|---|---|
| **Pitched-only similarity** | **47%** |
| Positional accuracy | 55% |
| Note-type similarity | 54% |
| Rhythm similarity | 54% |
| Combined similarity | 64% |
| Note coverage | 105% |
| Unique pitches | 17.2 pred / 32.0 ref |
| Max pitch similarity | 80% |
| Max longest match | 318 events |
| Header accuracy | key=96%, time=100%, parts=100% |

### Bimodal distribution

The accuracy is highly bimodal across samples:
- 23/50 samples (46%) have >50% pitch accuracy (range 56-80%)
- 13/50 samples (26%) have <10% pitch accuracy (range 1-5%)
- 14/50 samples (28%) fall in between (12-49%)

This suggests the model succeeds on some score configurations and fails
completely on others — not a gradual degradation.

### Comparison: 1K vs 5K samples

The 1K baseline reported "74% pitch" but that was measured on only 10 noisy
WandB inference samples. With a proper 50-sample evaluation:

| | 1K samples (WandB n=10) | 5K samples (eval n=50) |
|---|---|---|
| Pitched-only sim | ~56% (noisy) | **47%** (reliable) |
| eval_loss | ~0.002 | **0.000425** |
| Unique pitches pred | ~16 | 17.2 |

The 5K run achieved 5× lower eval_loss, but pitch accuracy did not improve
significantly. The eval_loss improvement is driven by better syntax/format
accuracy, not better pitch reading. The model still covers only ~17/32 unique
pitches in the reference.

### Inference speed (Blackwell RTX PRO 6000)

| Metric | Value |
|---|---|
| Throughput | 26 tok/s (steady state) |
| Per-sample time | ~153s (4096 tokens generated) |
| 50-sample eval | ~2.1 hours |

Tested acceleration options:
- **flash-attn 2.8.3**: Produces garbage output on Blackwell SM120 — incompatible
- **flash-linear-attention + causal-conv1d**: 12% throughput boost (29 tok/s) but
  produces garbage when loading checkpoints trained without it (different computation graph)
- **vLLM v0.17.0**: Has Qwen3.5 architecture support but tokenizer compatibility
  issues with unsloth model — not yet working
- **PyTorch SDPA**: Used by default, works correctly

For new training runs, installing flash-linear-attention from the start would
enable both the training speedup and compatible inference. Existing checkpoints
must be evaluated without it.


---

**Continued in [`2026-04-01-context-length-and-eos.md`](2026-04-01-context-length-and-eos.md)** —
analysis of training target truncation, the EOS learning bug, and the
`max_length=8192` fix.
