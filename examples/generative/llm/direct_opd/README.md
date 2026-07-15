# Direct-OPD: Weak-to-Strong On-Policy Distillation

Transfer RL-induced policy shifts from a weak teacher to a stronger student
without re-running sparse-reward RL on the target model.

**Paper:** [Weak-to-Strong Generalization via Direct On-Policy Distillation](https://huggingface.co/papers/2607.05394)
(Feng, Gao, Chi et al., Tsinghua AIR / ByteDance Seed, 2026)

**Upstream:** https://github.com/BytedTsinghua-SIA/Direct-OPD

## What It Does

Direct-OPD compares a post-RL weak teacher against its own pre-RL reference
and treats the per-token log-ratio as a dense implicit reward. This reward
signal is applied on the stronger student's on-policy rollouts, effectively
teaching the student *what RL improved* without the student needing to run
RL itself.

Key steps per training iteration:

1. **Student generates** responses on-policy (sampling)
2. **Select top-K tokens** from the student's distribution at each position
3. **Compute teacher delta** = `log pi_teacher(v) - log pi_teacher_ref(v)` for
   those tokens — positive means RL made the teacher *more* likely to use that
   token
4. **Weight by student probability** (Rao-Blackwellized advantage)
5. **Policy gradient update** pushing the student toward tokens RL preferred
6. **KL regularization** to prevent drift from the student's initial policy

## What to Expect

- **First run:** Downloads three models (~3.5 GB student + ~3 GB teacher +
  ~3 GB teacher reference) and the Skywork math dataset subset. Total first-run
  download is ~10 GB.
- **VRAM:** ~20-22 GB peak with 3 models loaded simultaneously in bfloat16,
  8-bit optimizer, and gradient checkpointing. Requires a 24 GB GPU (A10, A5000,
  RTX 4090, etc.) or larger.
- **Output:** Checkpoints and training logs written to `outputs/` in the example
  directory. Metrics include policy gradient loss, KL divergence, mean teacher
  delta, and advantage magnitude.
- **Runtime:** Each training step takes approximately 30-90 seconds depending
  on response length and GPU speed. The default 20-step smoke test completes in
  under 30 minutes.

## Quick Start

```bash
# Build the Docker image
bash examples/generative/llm/direct_opd/build.sh

# Run training (default 20 steps)
bash examples/generative/llm/direct_opd/train.sh

# Override config
bash examples/generative/llm/direct_opd/train.sh --config config.yaml
```

## Models

| Role | Model | Size | Description |
|------|-------|------|-------------|
| Student | `Qwen/Qwen3-1.7B` | 1.7B | Stronger model being improved |
| Teacher | `hbx/JustRL-DeepSeek-1.5B` | 1.5B | Post-RL weak teacher |
| Teacher Ref | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | Pre-RL baseline |

The "weak-to-strong" aspect: the 1.5B teacher pair provides the training
signal, while the 1.7B student is the target. The teacher is weaker in
capacity but has learned useful policy shifts through RL on math tasks.

## Configuration

All hyperparameters are in `config.yaml`. Key settings:

```yaml
opd:
  top_k: 16        # Student tokens evaluated per position
  kl_coef: 0.1     # KL regularization strength
  adaptive_kl: true # Auto-adjust KL coefficient

training:
  max_steps: 20            # Increase for longer training
  max_response_len: 512    # Max generation length
  learning_rate: 1.0e-6    # Conservative for stability
  gradient_checkpointing: true  # Required for 24GB GPUs
```

## Training Metrics

The training log (`outputs/train_log.jsonl`) records per-step:

- `loss` — total loss (pg_loss + kl_coef * kl_loss)
- `pg_loss` — Direct-OPD policy gradient loss
- `kl_loss` — KL divergence from initial student
- `mean_delta` — average teacher log-ratio (strength of RL signal)
- `delta_pos_frac` — fraction of top-K tokens with positive delta
- `mean_adv` — mean weighted advantage
- `grad_norm` — gradient norm (clipped to 1.0)
- `gpu_mem_gb` — peak GPU memory usage

## Differences from Upstream

This example is a self-contained reimplementation of the core Direct-OPD
objective for demonstration purposes. Differences from the full upstream
repository:

| Aspect | Upstream | This Example |
|--------|----------|-------------|
| Framework | verl (Ray + vLLM + FSDP) | PyTorch + transformers |
| GPUs | 8× A100 | 1× 24GB GPU |
| Generation | vLLM rollout engine | transformers `generate()` |
| Optimizer | AdamW | AdamW 8-bit |
| Data | Full 105K Skywork math | 500-prompt subset |
| Training | 300 steps, batch 128 | 20 steps, batch 1 |

The algorithmic core — top-K delta computation, probability-weighted
advantage, and adaptive KL — is faithful to the paper.

## References

- [Direct-OPD Paper (arXiv:2607.05394)](https://arxiv.org/abs/2607.05394)
- [Direct-OPD GitHub](https://github.com/BytedTsinghua-SIA/Direct-OPD)
- [verl Framework](https://github.com/volcengine/verl)
- [JustRL Paper](https://arxiv.org/abs/2504.13837)
