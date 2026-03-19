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

## Conclusions

1. **MXC format is essential** — the only way to get the model to generate actual notes.
   XML format never reaches note content within inference token budget.

2. **Ministral-3 3B can learn rhythm, lyrics, and structure but NOT pitch** from images.
   This is likely a vision encoder resolution limitation — the 3B model cannot resolve
   which staff line a note head sits on.

3. **2 epochs is optimal** for this model/data combination. More epochs overfit without
   improving pitch discrimination.

4. **Data cleaning helps** — stripping non-visible metadata reduced eval_loss from 0.158
   (raw XML) to 0.048 (clean XML). MXC further focuses learning on musical content.

## Next steps

1. **Larger model with MXC** — Qwen3.5-9B has a stronger vision encoder that may resolve
   pitch from staff positions. The 2-epoch sweet spot means this is a ~2h experiment.

2. **Pitch-focused evaluation** — build an automated metric comparing predicted vs reference
   pitches per measure (not just counting notes).

3. **Curriculum learning** — train first on simple monophonic lines (voice-only pages)
   before full voice+piano scores.

4. **Image augmentation** — the model may need zoomed/cropped views of individual staves
   rather than full-page images to learn fine-grained pitch discrimination.
