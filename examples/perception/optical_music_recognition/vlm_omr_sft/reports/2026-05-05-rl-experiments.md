# RL Experiments for OMR (2026-05-02 to 05-05)

**Previous**: [`2026-04-12-findings.md`](2026-04-12-findings.md) — SFT findings
and early RL analysis.

## Summary

Four GRPO attempts, all degraded accuracy. Root causes identified and
fixed iteratively. Two key discoveries:
1. Vision LoRA is unnecessary (97% without it)
2. The reward function was fundamentally misaligned with the eval metric

## GRPO runs

| Run | Config | Step-0 | After GRPO | Issue |
|---|---|---|---|---|
| v1 | Openscore, 3 separate rewards | ~0%? | 3% | Vision LoRA silently dropped |
| v2 | Level 7, combined LCS reward | ~0%? | 5% | Same adapter bug |
| v3 | Level 6, from_pretrained(adapter) | ~95% | 25% | vLLM ignores vision LoRA during generation |
| v4 | Level 6, merged model | 69% | 28% | Reward misaligned with eval metric |

## Infrastructure issues (resolved)

### 1. Adapter loading (v1, v2)
`get_peft_model(vision=False) + set_peft_model_state_dict()` silently
dropped vision LoRA keys because the LoRA structure didn't match the
SFT adapter (which had vision+language LoRA).

### 2. Vision LoRA / vLLM incompatibility (v3)
`FastVisionModel.from_pretrained(adapter_path)` loaded correctly (verified
100% pitch sim), but GRPO uses vLLM internally for generation. vLLM does
NOT support LoRA on vision layers → model was partially blind during
generation, but reward was computed on full model (could see).

### 3. Merged model + random LoRA (v4)
Merged all LoRA into base weights, added fresh language-only LoRA.
Model could see (69% at step 0) but the random language LoRA degraded
output. GRPO updated the random LoRA but accuracy still dropped.

## Reward misalignment (the core problem)

In ALL GRPO runs, reward went UP while accuracy went DOWN:

| Run | Reward trend | Accuracy trend |
|---|---|---|
| v3 | +1.15 → +1.44 | 95% → 25% |
| v4 | +1.22 → +1.31 | 69% → 28% |

**Root cause:** The reward function used a custom LCS computation that
did not match the eval metric. The eval uses `SequenceMatcher.ratio()`
from difflib (which computes 2×matches / (len_a + len_b)). The reward
used a hand-rolled LCS normalized by reference length only, plus part
count and length control components.

**Fix:** Replaced the combined reward with the EXACT same computation
as the eval metric: `SequenceMatcher(None, pred_pitches, ref_pitches).ratio()`.
No part count or length control — just pitch accuracy. This guarantees
the reward moves in the same direction as the eval metric.

## SFT vision ablation

| Config | finetune_vision_layers | Level 7a pitched-only |
|---|---|---|
| Standard | True | ~91% (MXC, from earlier experiments) |
| **Ablation** | **False** | **97%** (MXC2) |

**Vision LoRA is unnecessary and possibly harmful.** The base Qwen3.5-9B
visual encoder is already sufficient for music notation. Training vision
layers may add noise.

**Implication:** Future SFT should always use `finetune_vision_layers=False`.
This makes the SFT adapter directly compatible with GRPO (no merge step
needed, no vLLM incompatibility).

## Best-of-N inference

| | Greedy | Best-of-8 |
|---|---|---|
| Level 7 (20 samples) | 6.2% | **22.5%** (+16pp) |
| Improved samples | — | 20/20 (100%) |

Every sample improved with sampling diversity. The model HAS better
outputs in its distribution — it just doesn't pick them consistently
with greedy decoding.

## Next steps

### GRPO v5: fixed reward (SequenceMatcher), correct adapter loading

**v5 diagnostic (Level 7a, 10 steps):**
- Step-0: 100% pitch sim (adapter loaded correctly)
- All 10 steps: reward=+3.0, std=0, grad_norm=0
- Model too good on Level 7a — no variance between generations
- Result: 97% → 97% (no-op, but confirmed pipeline works)

**v5 diagnostic (Level 9, 10 steps):**
- Step-0: 100% pitch sim
- Reward has variance: +1.22 to +2.88, grad_norm up to 11.75
- Result: **84% → 84%** — first non-degrading GRPO run!

**v5 full (Level 9, 100 steps):**
- Reward first 10 avg: +2.33, last 10 avg: +2.04 (slight decline)
- Result: **84% → 84%** — maintained but did NOT improve

### All GRPO runs summary

| Run | Issue | Step-0 | After | Improved? |
|---|---|---|---|---|
| v1 | Broken adapter (vision LoRA dropped) | ~0% | 3% | No — degraded |
| v2 | Same adapter bug | ~0% | 5% | No — degraded |
| v3 | vLLM ignores vision LoRA | ~95% | 25% | No — degraded |
| v4 | Misaligned reward (LCS ≠ eval) | 69% | 28% | No — degraded |
| v5 L7a | None (too easy, no gradient) | 100% | 97% | N/A — no-op |
| v5 L9 diag | None (correct) | 100% | 84% | No — maintained |
| **v5 L9 full** | **None (correct, 100 steps)** | **100%** | **84%** | **No — maintained** |

### Why GRPO doesn't improve accuracy on synthetic data

After fixing all infrastructure bugs, GRPO correctly maintains the SFT
model (84%) but cannot push it higher, despite having learning signal.

**Variance analysis (100-step full run):**

| Variance bucket | Steps | % | Gradient? |
|---|---|---|---|
| Zero std (<0.01) | 37 | 37% | No |
| Low std (0.01-0.3) | 27 | 27% | Weak |
| Medium std (0.3-1.0) | 27 | 27% | Yes |
| High std (>1.0) | 9 | 9% | Strong |

36% of steps have useful gradient. The model DOES produce varying
quality outputs. But:

1. **Reward slightly degraded over training.** First 50 steps: avg
   reward +2.13, 20 perfect, 8 poor. Last 50: +1.97, 18 perfect,
   12 poor. GRPO is slowly making the model worse even with beta=1.0.

2. **4 generations per step is too few.** With 37% zero-variance steps,
   most of the training is no-ops. The 9% high-variance steps are too
   rare to drive consistent improvement.

3. **LR and beta may be mistuned.** LR=5e-7 + beta=1.0 makes updates
   extremely small. The useful gradient steps produce tiny weight
   changes that can't accumulate fast enough.

### Literature review: RL for document understanding (2025-2026)

Multiple systems have successfully applied GRPO/RL to document OCR:

| System | Task | SFT→RL gain | Technique |
|---|---|---|---|
| FD-RL | Document OCR | 87→90% | Format-decoupled rewards |
| HunyuanOCR | Text spotting | significant | IoU + edit distance |
| Infinity-Parser | Document parsing | SOTA | Hungarian matching |
| RL-Struct | JSON generation | 65→90% | Hierarchical rewards |

Key differences from our setup:
- **16 generations per prompt** (we used 4) — DeepSeek-R1 recipe
- **Decomposed rewards** (validity + structure + content, not one scalar)
- **reward=0 for unparseable output** (hard constraint on validity)
- **Token-level loss** (not per-sequence averaging which penalizes long output)
- **Large batch sizes** (512+ total samples)

Also relevant: "SFT Memorizes, RL Generalizes" paper found that
excessive SFT locks the model into rigid patterns that RL cannot
escape. Our model had extensive SFT (3 curriculum stages × 3 epochs).

### What to try next

1. **num_generations=16** — the most impactful change. DeepSeek-R1
   uses 16. With 4, we had 37% zero-variance steps.

2. **Decomposed reward**:
   - Validity: does the output parse as MXC2? (0 or 1)
   - Structure: correct part count? (0 or 1)
   - Content: SequenceMatcher.ratio per part (0 to 1)
   - Weight: validity=0 (hard gate), structure=0.2, content=0.8

3. **DAPO fixes**: token-level loss aggregation, dynamic sampling to
   skip zero-variance batches, entropy monitoring.

4. **Less SFT before RL**: try RL from a 1-epoch SFT model instead
   of the full curriculum chain. The "SFT Memorizes" paper suggests
   minimal SFT + RL outperforms extensive SFT + RL.

5. **Try openscore**: the model has much more variance on real data
   (23% baseline). But need to verify the model produces parseable
   output first.
