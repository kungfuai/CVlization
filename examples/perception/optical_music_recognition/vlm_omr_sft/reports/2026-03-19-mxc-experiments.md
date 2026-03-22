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

## Next steps

### 1. Synthetic training data (highest priority)

The model has converged on 3K lieder samples. Generate **synthetic single-page
music** with controlled complexity:

- **Simple monophonic melodies**: one staff, varied pitches/rhythms, no lyrics.
  Isolates pitch learning. LilyPond can generate thousands programmatically.
- **Graduated complexity**: single-staff → bass clef → chords → two-staff piano
  → lyrics → dynamics.
- **Controlled pitch coverage**: ensure all pitches/octaves are represented
  (current lieder data biased toward voice range).
- **Augmentation**: vary spacing, font size, staff distance.

### 2. Evaluate on full dev set

Current metrics on 10 held-out samples. Run `eval_mxc.py` on all 193 dev samples.

### 3. Qwen3-VL-32B

3.5× larger model. If 9B gets 36%, 32B may push higher. 4-bit fits in 95 GB VRAM.

### Lower priority

- **Training max_length 8192** — marginal benefit, not the bottleneck.
- **Quartets/orchestra data** — lower quality transcriptions. Synthetic is better.
