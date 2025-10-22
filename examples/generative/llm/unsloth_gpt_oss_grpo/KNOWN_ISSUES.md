# Known Issues

## ~~MoE Forward Pass Error During GRPO Training~~ [RESOLVED]

### Issue
Training fails during the generation step with a TorchDynamo compilation error in the MoE forward pass:

```
File "/opt/conda/lib/python3.11/site-packages/unsloth_zoo/temporary_patches/gpt_oss.py", line 1231, in forward
    hidden_states = moe_forward_inference(decoder_layer.mlp, hidden_states)
```

### Status: ✅ RESOLVED
The issue was caused by incompatible versions of PyTorch and Triton. **Fixed by upgrading to newer versions**.

### What We've Verified
- ✅ Dataset format is correct (`reasoning_effort`: "low" is included)
- ✅ Reward function signatures match Unsloth format (`completions, **kwargs`)
- ✅ GRPOConfig includes required `max_prompt_length` and `max_completion_length`
- ✅ Model loads successfully with `offload_embedding=True`
- ✅ LoRA setup works correctly (0.02% parameters trainable)
- ✅ GRPO trainer initializes without errors

### Root Cause
The error occurs in `/opt/conda/lib/python3.11/site-packages/unsloth_zoo/temporary_patches/gpt_oss.py` - a temporary patch file, suggesting this is a known compatibility issue between:
- GPT-OSS 20B MoE architecture
- TorchDynamo compilation
- GRPO generation step

### Solution
Upgrade to newer PyTorch and Triton versions:
- **torch >= 2.8.0** (was 2.6.0)
- **triton >= 3.4.0** (was 3.2.0)
- **trl == 0.22.2** (was 0.23.0)

The Dockerfile has been updated to use these versions automatically.

### Related Issues
- **Similar TorchDynamo errors with GPT-OSS**: https://github.com/unslothai/unsloth/issues/3205 (120B model, recompilation issues)
- **TorchDynamo verbose logging**: https://github.com/unslothai/unsloth/issues/3321 (debugging steps)
- Unsloth GRPO docs: https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning
- Unsloth notebook (works in Colab): https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb
- The notebook works in Colab (likely different torch/triton versions)

### Environment (Original - Broken)
- Unsloth: 2025.10.1
- Transformers: 4.56.2
- Torch: 2.6.0+cu124 ❌
- Triton: 3.2.0 ❌
- TRL: 0.23.0 ❌
- Python: 3.11
- GPU: NVIDIA A10 (21.988 GB)

### Environment (Fixed - Working)
- Unsloth: 2025.10.1
- Transformers: 4.56.2
- Torch: 2.8.0+cu128 ✅
- Triton: 3.4.0 ✅
- TRL: 0.22.2 ✅
- Python: 3.11
- GPU: NVIDIA A10 (21.988 GB)
