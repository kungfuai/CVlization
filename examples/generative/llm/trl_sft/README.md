## Llama 3B Supervised Fine-Tuning with TRL

This example demonstrates how to fine-tune Llama 3.2 3B using **TRL (Transformer Reinforcement Learning)** from HuggingFace with LoRA adapters.

### What is TRL?

TRL is HuggingFace's official library for training language models with supervised fine-tuning (SFT) and reinforcement learning methods (PPO, DPO, GRPO). It provides:
- Standard HuggingFace integration
- Support for multiple training methods (SFT, DPO, PPO, GRPO)
- Excellent documentation and community support
- Production-ready training pipelines

### Features

- **Llama 3.2 3B**: Meta's latest small language model
- **TRL SFTTrainer**: HuggingFace's standard fine-tuning approach
- **LoRA**: Parameter-efficient fine-tuning (trains only 0.15% of parameters)
- **4-bit Quantization**: Memory-efficient training with bitsandbytes
- **Alpaca Dataset**: High-quality instruction-following examples

### Comparison: TRL vs Unsloth

This example uses the **same model, dataset, and hyperparameters** as `examples/text_gen/unsloth/llama_3b_sft/` to enable direct performance comparison.

| Aspect | TRL | Unsloth |
|--------|-----|---------|
| **Speed** | Baseline | ~2-3x faster |
| **Memory** | Baseline | ~50% less VRAM |
| **Ecosystem** | Native HF integration | Unsloth-specific |
| **Methods** | SFT, DPO, PPO, GRPO | SFT, GRPO (optimized) |
| **Documentation** | Extensive official docs | Growing community docs |
| **Best For** | Standard workflows, production | Maximum speed/efficiency |

**When to use TRL:**
- Standard HuggingFace workflows
- Need DPO/PPO methods
- Production environments with established HF pipelines
- Prefer official, well-documented libraries

**When to use Unsloth:**
- Training speed is critical
- Limited VRAM (consumer GPUs)
- Want maximum performance optimizations

### Prerequisites

1. NVIDIA GPU with at least 8GB VRAM (3090, 4090, A10, etc.)
2. HuggingFace account and token (required for Llama models)
3. Docker with NVIDIA runtime support

### Quickstart

1. Set your HuggingFace token:
```bash
export HF_TOKEN=your_huggingface_token
```

2. Build the Docker image:
```bash
bash examples/text_gen/trl/llama_3b_sft/build.sh
```

3. Run training:
```bash
bash examples/text_gen/trl/llama_3b_sft/train.sh
```

### Training Details

- **Model**: meta-llama/Llama-3.2-3B-Instruct (3 billion parameters)
- **Dataset**: yahma/alpaca-cleaned (52K instruction-following examples)
- **Method**: Supervised Fine-Tuning with LoRA
- **Max Steps**: 20 (for quick testing)
- **Batch Size**: 2 per device with 4x gradient accumulation (effective batch size: 8)
- **Learning Rate**: 2e-4
- **LoRA Rank**: 16
- **Quantization**: 4-bit (bitsandbytes NF4)
- **Context Length**: 2048 tokens

Training takes approximately 5-10 minutes on an A10 GPU for 20 steps.

### Dataset Format

The example supports three dataset formats:

#### 1. Alpaca Format (default)
```json
{
  "instruction": "Classify the following into animals, plants, and minerals",
  "input": "Oak tree, copper ore, elephant",
  "output": "Animals: elephant\nPlants: oak tree\nMinerals: copper ore"
}
```

#### 2. ShareGPT Format
```json
{
  "conversations": [
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis is the process..."}
  ]
}
```

#### 3. Custom Format
```json
{
  "text": "### Instruction:\nExplain quantum computing\n\n### Response:\nQuantum computing..."
}
```

Configure the format in `config.yaml`:
```yaml
dataset:
  path: "yahma/alpaca-cleaned"
  format: "alpaca"  # or "sharegpt" or "custom"
```

### Configuration

All training is controlled via `config.yaml`. Key sections:

#### Model Configuration
```yaml
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true
```

#### LoRA Settings
```yaml
lora:
  r: 16  # LoRA rank (higher = more capacity, slower)
  alpha: 16  # LoRA scaling factor
  dropout: 0
  target_modules:  # Which layers to apply LoRA
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

#### Training Settings
```yaml
training:
  max_steps: 20  # Number of training steps
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  optim: "adamw_8bit"  # Memory-efficient optimizer
```

### Custom Datasets

To use your own dataset:

1. **Prepare data in one of the supported formats** (see Dataset Format section above)

2. **Update config.yaml**:
```yaml
dataset:
  path: "your-hf-username/your-dataset"  # or local path
  format: "alpaca"  # or "sharegpt" or "custom"
  split: "train"
  # Optional: limit for testing
  max_samples: 1000
```

3. **For local datasets**, mount the directory in `train.sh`:
```bash
-v "/path/to/your/data:/workspace/data" \
```

See [CUSTOM_DATASETS.md](../CUSTOM_DATASETS.md) for detailed instructions on data preparation.

### Output

The fine-tuned model will be saved to:
- `./llama-alpaca-finetune/final_model/`

This directory contains:
- LoRA adapter weights (`adapter_model.safetensors`)
- Adapter configuration (`adapter_config.json`)
- Tokenizer files

### Loading the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./llama-alpaca-finetune/final_model",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./llama-alpaca-finetune/final_model"
)

# Generate
prompt = "### Instruction:\nExplain machine learning\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### Benchmarking vs Unsloth

To compare TRL vs Unsloth performance:

1. **Run TRL training**:
```bash
cd examples/text_gen/trl/llama_3b_sft
time bash train.sh 2>&1 | tee trl_training.log
```

2. **Run Unsloth training**:
```bash
cd examples/text_gen/unsloth/llama_3b_sft
time bash train.sh 2>&1 | tee unsloth_training.log
```

3. **Compare metrics**:
   - Training time (from `time` command)
   - Peak GPU memory (check `nvidia-smi` during training)
   - Final loss values (from logs)

Expected results:
- Unsloth should be **2-3x faster**
- Unsloth should use **~50% less VRAM**
- Final model quality should be similar

### Troubleshooting

**Out of Memory (OOM) errors:**
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_seq_length` to 1024
- Enable gradient checkpointing (add to config)

**Slow training:**
- Increase batch size if you have VRAM headroom
- Enable `packing=True` in SFTConfig for better GPU utilization
- Consider switching to Unsloth for 2-3x speedup

**Model quality issues:**
- Increase `max_steps` for more training
- Try different learning rates (1e-4 to 5e-4)
- Increase LoRA rank to 32 or 64

### Next Steps

After mastering SFT with TRL, explore:
- **DPO (Direct Preference Optimization)**: Align models with human preferences
- **PPO (Proximal Policy Optimization)**: Reinforcement learning for reasoning
- **GRPO (Group Relative Policy Optimization)**: Memory-efficient RL (see `unsloth/gpt_oss_grpo/`)

### Reference

- [TRL Documentation](https://huggingface.co/docs/trl)
- [TRL GitHub](https://github.com/huggingface/trl)
- [SFTTrainer Guide](https://huggingface.co/docs/trl/sft_trainer)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
