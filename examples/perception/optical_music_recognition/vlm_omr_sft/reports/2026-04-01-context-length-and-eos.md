# MXC Experiments: Context Length & EOS Learning (2026-04-01)

**Previous**: [`2026-03-19-mxc-experiments.md`](2026-03-19-mxc-experiments.md) —
established Qwen3.5-9B r=32 MXC as the best baseline (47% pitched-only on
Level 7 synthetic, n=50).

This document covers the discovery that training target truncation was a
major bottleneck, and the experiments that fixed it.

## Experiment 12: Level 7 with max_length=8192 (truncation fix)

**WandB run**: `hl8c36sk`

| Setting | Value |
|---|---|
| Model | Qwen3.5-9B r=32 MXC |
| Dataset | zzsi/synthetic-scores level7, 5K samples |
| `max_length` | **8192** (was 4096) |
| `inference_max_new_tokens` | 8192 |
| `n_examples` | 3 (was 10 to reduce callback overhead) |
| `log_examples_every_n_steps` | 200 (was 50) |
| Epochs | 3 (2979 steps) |
| Best eval_loss | **0.000410** (step 2400) |
| Peak VRAM | 31.9 GB (33.6%) |
| Runtime | 8h 28m |

### Motivation

Level 7 MXC references are ~4200 tokens, which exceeds `max_length=4096`.
Combined with ~975 image tokens + instruction, the full sequence is ~6256
tokens — truncated to 4096 removes the tail of the MXC AND the final EOS
token. The model never sees a complete score ending during training.

### Accuracy evaluation (50 test samples)

| Metric | 4K context | **8K context** | Change |
|---|---|---|---|
| **Pitched-only sim** | 47% | **67%** | **+20pp** |
| Positional accuracy | 55% | 61% | +6pp |
| Note-type sim | 54% | 59% | +5pp |
| Rhythm sim | 54% | 59% | +5pp |
| Combined sim | 64% | **73%** | +9pp |
| Max pitch sim | 80% | 81% | +1pp |
| Max longest match | 318 | **335** | +17 |
| Key accuracy | 48/50 | **50/50** | 100% |
| Per-sample time | 152.9s | 153.5s | same |

**Pitch accuracy jumped from 47% to 67% — a 42% relative improvement.**
Just from fixing `max_length` truncation, with no other changes.

### EOS emission analysis

Comparing prediction length to reference length across levels:

| Level | max_length | Training truncation | Pred/ref ratio | EOS learned? |
|---|---|---|---|---|
| Level 1 | 4096 | No (ref ~884 chars) | **1.00** | Yes |
| Level 2 | 4096 | No (ref ~900 chars) | **1.00** | Yes |
| Level 3 | 4096 | No (ref ~1000 chars) | **1.00-1.01** | Yes |
| Level 7 (4K) | 4096 | **Yes** (ref ~6300 chars) | **~4096 tokens always** | **No** |
| Level 7 (8K) | 8192 | No (ref ~6300 chars) | **1.20-2.04** | Partial |

**Conclusion**: Training target truncation directly breaks EOS learning. When
the assistant response is truncated at `max_length`, the final `<|im_end|>`
token is cut off, and the model never learns when to stop.

The 8K run partially fixes this (pred lengths closer to reference), but the
model still overshoots by 20-100%. This may be because 3 epochs on complex
multi-part scores isn't enough for reliable EOS learning, or because the
synthetic data's structural patterns give weak termination signals.


---

**Continued in [`2026-04-08-ablation-and-instability.md`](2026-04-08-ablation-and-instability.md)** —
ablation study on Level 7 factors, training instability discovery on Level 6b,
and the stable-hyperparameters fix.
