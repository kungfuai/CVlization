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

### 1. GRPO v5 with fixed reward (immediate)
- Reward: `SequenceMatcher.ratio()` on pitched-only (exact eval metric)
- Model: SFT with `finetune_vision_layers=False` (no merge needed)
- Data: Level 6 (95% SFT baseline)
- Config: beta=1.0, LR=5e-7, 50 steps

### 2. Retrain SFT chain with vision=False
- Level 7a(vision=False) → Level 9 → openscore
- Use MXC2 throughout
- This creates RL-compatible adapters at every level

### 3. DPO as GRPO alternative
- Generate best/worst pairs from best-of-8 sampling
- Train DPO (no reward function needed)
- More stable for long-form structured output
