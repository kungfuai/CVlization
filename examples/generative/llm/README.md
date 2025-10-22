# Text Generation Examples

This directory contains fine-tuning examples for large language models (LLMs), organized by framework.

## Structure

```
examples/text_gen/
├── unsloth/          # Unsloth fine-tuning examples (fast & memory-efficient)
│   ├── llama_3b_sft/      # Llama 3.2 3B SFT
│   ├── gpt_oss_sft/       # GPT-OSS 20B SFT
│   ├── gpt_oss_grpo/      # GPT-OSS 20B GRPO (RL)
│   └── qwen_7b_sft/       # Qwen 2.5 7B SFT
│
└── trl/              # TRL (HuggingFace) fine-tuning examples
    └── sft/               # Supervised Fine-Tuning (works with any model)
```

## Framework Comparison

### Unsloth
- **Speed**: 2-3x faster than standard fine-tuning
- **Memory**: 50% less VRAM usage
- **Methods**: SFT, GRPO
- **Best for**: Maximum performance on consumer GPUs

**Examples:**
- [Llama 3B SFT](unsloth/llama_3b_sft/) - Fastest small model fine-tuning
- [GPT-OSS 20B SFT](unsloth/gpt_oss_sft/) - Large model (20B) on single GPU
- [GPT-OSS GRPO](unsloth/gpt_oss_grpo/) - Reinforcement learning for code generation
- [Qwen 7B SFT](unsloth/qwen_7b_sft/) - Multilingual model fine-tuning

### TRL (Transformer Reinforcement Learning)
- **Integration**: Native HuggingFace ecosystem
- **Methods**: SFT, DPO, PPO, GRPO
- **Documentation**: Extensive official docs
- **Best for**: Standard workflows, production environments

**Examples:**
- [SFT](trl/sft/) - Standard HF fine-tuning with TRL (configurable model)

## Quick Start

### Unsloth Examples

```bash
# Llama 3B (fastest, lowest memory)
cd examples/text_gen/unsloth/llama_3b_sft
bash build.sh
bash train.sh

# GPT-OSS 20B (larger model)
cd examples/text_gen/unsloth/gpt_oss_sft
bash build.sh
bash train.sh

# GPT-OSS GRPO (reinforcement learning)
cd examples/text_gen/unsloth/gpt_oss_grpo
bash build.sh
bash train.sh
```

### TRL Examples

```bash
# SFT with TRL (Qwen 0.5B by default, or any model via config)
cd examples/text_gen/trl/sft
bash build.sh
bash train.sh  # Works without HF_TOKEN for Qwen
```

## Benchmarking: Unsloth vs TRL

To compare performance, set both to use the same model (e.g., Llama 3.2 3B):

**Setup:**
1. Update `trl/sft/config.yaml` to use `meta-llama/Llama-3.2-3B-Instruct`
2. Set `HF_TOKEN` for both examples
3. Use identical dataset (Alpaca), hyperparameters, and LoRA config

**Run both and compare:**
```bash
# TRL baseline
cd examples/text_gen/trl/sft
export HF_TOKEN=your_token
time bash train.sh 2>&1 | tee trl_log.txt

# Unsloth optimized
cd examples/text_gen/unsloth/llama_3b_sft
export HF_TOKEN=your_token
time bash train.sh 2>&1 | tee unsloth_log.txt
```

**Expected results:**
- Unsloth: ~2-3x faster training time
- Unsloth: ~50% lower peak GPU memory
- Similar final model quality

## Custom Datasets

All examples support custom datasets via `config.yaml`. See [CUSTOM_DATASETS.md](CUSTOM_DATASETS.md) for detailed instructions.

**Quick example:**
```yaml
dataset:
  path: "your-username/your-dataset"  # HF Hub or local path
  format: "alpaca"  # or "sharegpt" or "custom"
  split: "train"
```

## Model Support

| Model | Size | Unsloth | TRL | Notes |
|-------|------|---------|-----|-------|
| Llama 3.2 | 3B | ✅ | ✅ | Best for comparison |
| GPT-OSS | 20B | ✅ | - | MoE architecture |
| Qwen 2.5 | 7B | ✅ | - | Multilingual |

## Training Methods

### SFT (Supervised Fine-Tuning)
- Train on instruction-response pairs
- Best for: Task-specific adaptations
- Examples: All `*_sft/` directories

### GRPO (Group Relative Policy Optimization)
- Reinforcement learning with reward functions
- Best for: Reasoning, code generation, alignment
- Examples: `unsloth/gpt_oss_grpo/`

### Coming Soon
- **DPO** (Direct Preference Optimization) - Preference-based training
- **PPO** (Proximal Policy Optimization) - Classic RL for LLMs

## Prerequisites

### Hardware
- NVIDIA GPU with 8GB+ VRAM (RTX 3090, 4090, A10, A100, etc.)
- For 20B models: 15GB+ VRAM recommended

### Software
- Docker with NVIDIA runtime
- HuggingFace account (for gated models like Llama)

### GPU Memory Requirements

| Model | Method | Min VRAM | Recommended |
|-------|--------|----------|-------------|
| Llama 3B | SFT | 8GB | 12GB |
| Qwen 7B | SFT | 12GB | 16GB |
| GPT-OSS 20B | SFT | 15GB | 20GB |
| GPT-OSS 20B | GRPO | 20GB | 24GB |

## Next Steps

1. **Start simple**: Begin with `unsloth/llama_3b_sft/` for fastest results
2. **Compare frameworks**: Run both Unsloth and TRL examples to see the difference
3. **Scale up**: Move to larger models (7B, 20B) as needed
4. **Try RL**: Explore GRPO for reasoning and code generation tasks
5. **Custom data**: Adapt examples to your own datasets

## Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Custom Datasets Guide](CUSTOM_DATASETS.md)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
