# Gemma-3 4B Text Fine-tuning (SFT)

Fine-tune Gemma-3 4B with Unsloth for 2x faster training using supervised fine-tuning (SFT).

## Features

- **Fast Training**: Unsloth provides 2x speedup over standard fine-tuning
- **Memory Efficient**: 4-bit quantization reduces VRAM to ~8GB
- **LoRA/PEFT**: Parameter-efficient fine-tuning with low-rank adaptation
- **Train on Responses Only**: Masks instruction/input parts, trains only on model responses
- **Gemma-3 Chat Template**: Properly formatted for Gemma-3 conversation style

## Quick Start

### 1. Build the Docker image
```bash
cvl run gemma3-4b-sft build
```

### 2. Run smoke test (30 steps, 1000 samples)
```bash
cvl run gemma3-4b-sft test
```

### 3. Full training
Edit `config.yaml` to adjust training parameters:
- Remove or increase `max_samples` for full dataset
- Set `num_train_epochs: 1` instead of `max_steps`
- Adjust `learning_rate`, `batch_size`, etc.

Then run:
```bash
cvl run gemma3-4b-sft train
```

## Configuration

All training parameters are in `config.yaml`:

### Model Settings
- `model.name`: Base model (default: `unsloth/gemma-3-4b-it`)
- `model.max_seq_length`: Maximum sequence length (default: 2048)
- `model.load_in_4bit`: Enable 4-bit quantization (default: true)

### LoRA Settings
- `lora.r`: LoRA rank (default: 8)
- `lora.alpha`: LoRA alpha (default: 8)
- `lora.dropout`: Dropout rate (default: 0)

### Dataset Settings
- `dataset.path`: HuggingFace dataset path (default: `mlabonne/FineTome-100k`)
- `dataset.max_samples`: Limit samples for testing (default: 1000)

### Training Settings
- `training.per_device_train_batch_size`: Batch size per device (default: 2)
- `training.gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `training.learning_rate`: Learning rate (default: 2e-4)
- `training.max_steps`: Max training steps (default: 30 for testing)
- `training.num_train_epochs`: Number of epochs (use instead of max_steps for full training)
- `training.do_eval`: Enable validation during training (default: true)
- `training.val_split_ratio`: Validation split ratio (default: 0.1)
- `training.val_steps`: Validate every N steps (default: 10)

**Note**: We use "val" (validation) terminology in our config, but HuggingFace Trainer logs will show "eval_loss" - these refer to the same thing.

## Dataset Format

The default dataset (`mlabonne/FineTome-100k`) uses conversation format with a `conversations` field. The script automatically:
1. Standardizes various conversation formats
2. Applies Gemma-3 chat template
3. Masks instruction parts (trains only on responses)

### Custom Datasets

To use your own dataset, ensure it has a `conversations` field with this structure:
```python
[
    {"role": "user", "content": "Question here"},
    {"role": "assistant", "content": "Answer here"}
]
```

Or the script will attempt to standardize other common formats automatically.

## Memory Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: ~8GB (with 4-bit quantization)
- **Disk**: ~15GB for model + cache

## Output

The fine-tuned model is saved to `outputs/final_model/` and includes:
- LoRA adapter weights
- Tokenizer configuration
- Model config

### Loading the Fine-tuned Model

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="outputs/final_model",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

## Tips

1. **For Testing**: Keep `max_samples: 1000` and `max_steps: 30` for quick validation
2. **For Full Training**: Remove `max_samples` and set `num_train_epochs: 1`
3. **Memory Issues**: Reduce `per_device_train_batch_size` or `max_seq_length`
4. **Faster Convergence**: Increase `gradient_accumulation_steps` for larger effective batch size
5. **Better Results**: Reduce `learning_rate` to `2e-5` for longer training runs

## References

- Original Notebook: https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb
- Unsloth GitHub: https://github.com/unslothai/unsloth
- Gemma-3 Model Card: https://huggingface.co/google/gemma-3-4b-it
- FineTome-100k Dataset: https://huggingface.co/datasets/mlabonne/FineTome-100k
