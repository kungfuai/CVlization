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
model (84%) but cannot push it higher. The reasons:

1. **The model is at its LoRA capacity ceiling.** With r=32 (102M
   trainable params out of 9.5B), the model may have already learned
   everything it can from the SFT data. GRPO can't teach new knowledge
   — it can only make the model more consistently pick its best outputs.

2. **Generation variance is too low on synthetic data.** On Level 9,
   many steps had reward_std close to 0 (all 4 generations were equally
   good). GRPO needs variance to compute gradients. The SFT model
   is already very consistent on synthetic data.

3. **The SFT-to-RL gap hypothesis is wrong for this task.** In math/code
   RL, the model knows the answer sometimes but doesn't always produce
   it — RL makes it more consistent. For OMR, the model's errors are
   systematic (misreading specific notation features), not random. RL
   can't fix systematic errors because there's no "better generation"
   in the model's distribution for those cases.

4. **Best-of-8 showed only +16pp on Level 7** (6% → 22%), and that was
   on out-of-distribution data (L9 adapter tested on L7). On in-
   distribution data (L9 on L9), the variance is much lower — best-of-4
   during GRPO barely differs from greedy.

### What could still work

1. **GRPO on openscore (23% baseline).** The model has much more
   variance on real data. Best-of-8 showed consistent improvement.
   But the 23% baseline may be too weak for stable RL.

2. **DPO with rejection sampling.** Generate 8 outputs per openscore
   image, rank by eval metric, train DPO on (best, worst) pairs.
   Avoids the variance problem (DPO uses pairs, not group ranking).

3. **Larger LoRA rank (r=64 or r=128).** If the model is at capacity
   with r=32, a higher rank might allow GRPO to learn new patterns.

4. **More generations per step (8 or 16).** With only 4 generations,
   variance is low on synthetic data. More generations = better
   ranking signal.

5. **Train on openscore directly with SFT vision=False.** The 97%
   Level 7a result with vision=False suggests the base encoder is
   strong. A clean SFT→RL pipeline with vision=False may work.
