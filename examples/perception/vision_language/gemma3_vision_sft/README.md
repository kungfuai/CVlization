# Gemma-3 Vision Fine-tuning (SFT)

Fine-tune Gemma-3 vision models with Unsloth for efficient vision-language training.

## Features

- **Vision + Language Training**: Fine-tune both vision encoder and language model
- **Fast Training**: Unsloth provides efficient training with minimal memory overhead
- **Memory Efficient**: 4-bit quantization reduces VRAM to ~16GB
- **LoRA/PEFT**: Parameter-efficient fine-tuning for both vision and language layers
- **Dual Model Support**: Works with both Gemma-3 and Gemma-3N variants
- **LaTeX OCR Example**: Image-to-text task for mathematical formula recognition

## Quick Start

### 1. Build the Docker image
```bash
cvl run gemma3-vision-sft build
```

### 2. Run smoke test (30 steps, 100 samples)
```bash
cvl run gemma3-vision-sft test
```

### 3. Full training
Edit `config.yaml` to adjust training parameters:
- Remove or increase `max_samples` for full dataset
- Set `num_train_epochs: 2` instead of `max_steps`
- Adjust model choice (Gemma-3 vs Gemma-3N)

Then run:
```bash
cvl run gemma3-vision-sft train
```

## Configuration

All training parameters are in `config.yaml`:

### Model Settings
- `model.name`: Choose `unsloth/gemma-3-4b-pt` (Gemma-3) or `unsloth/gemma-3n-E4B` (Gemma-3N)
- `model.load_in_4bit`: Enable 4-bit quantization (default: true)
- `model.max_length`: Maximum sequence length (default: 2048)
- `model.chat_template`: Use `gemma-3` or `gemma-3n` depending on model

### LoRA Settings
- `lora.r`: LoRA rank (default: 16 for Gemma-3, use 32 for Gemma-3N)
- `lora.alpha`: LoRA alpha (default: 16)
- `lora.finetune_vision_layers`: Train vision encoder (default: true)
- `lora.finetune_language_layers`: Train language model (default: true)

### Dataset Settings
- `dataset.path`: HuggingFace dataset path (default: `unsloth/LaTeX_OCR`)
- `dataset.max_samples`: Limit samples for testing (default: 100)
- `dataset.instruction`: Task instruction for the model

### Training Settings
- `training.per_device_train_batch_size`: Batch size per device (default: 1)
- `training.gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `training.learning_rate`: Learning rate (default: 2e-4)
- `training.max_steps`: Max training steps (default: 30 for testing)
- `training.num_train_epochs`: Number of epochs (use instead of max_steps for full training)
- `dataset.val_split_ratio`: Validation split ratio (default: 0.2)
- `training.val_steps`: Validate every N steps (default: 10)

**Note**: We use "val" (validation) terminology in our config, but HuggingFace Trainer logs will show "eval_loss" - these refer to the same thing.

## Model Variants

### Gemma-3 (Default)
```yaml
model:
  name: "unsloth/gemma-3-4b-pt"
  chat_template: "gemma-3"
lora:
  r: 16
```

### Gemma-3N (Alternative)
```yaml
model:
  name: "unsloth/gemma-3n-E4B"
  chat_template: "gemma-3n"
lora:
  r: 32  # Higher rank for better results
```

Gemma-3N typically requires more steps (60 vs 30) but may provide better results.

## Dataset Format

The default dataset (`unsloth/LaTeX_OCR`) contains:
- `image`: PIL Image of mathematical formula
- `text`: LaTeX representation

The script converts to conversation format:
```python
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Write the LaTeX representation for this image."},
            {"type": "image", "image": <PIL.Image>}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "\\frac{1}{2}"}]
    }
]
```

### Custom Datasets

Your dataset should have `image` and `text` fields. The script will automatically convert them to conversation format.

## Memory Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: ~16GB (with 4-bit quantization)
- **Disk**: ~20GB for model + cache

## Output

The fine-tuned model is saved to `outputs/final_model/` and includes:
- LoRA adapter weights for both vision and language
- Processor configuration
- Model config

### Loading the Fine-tuned Model

```python
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="outputs/final_model",
    load_in_4bit=True,
)

FastVisionModel.for_inference(model)
```

## Tips

1. **For Testing**: Keep `max_samples: 100` and `max_steps: 30` for quick validation
2. **For Full Training**: Remove `max_samples` and set `num_train_epochs: 2`
3. **Memory Issues**: Keep `per_device_train_batch_size: 1` for vision models
4. **Better Results**: Try Gemma-3N with higher LoRA rank (r=32)
5. **Faster Convergence**: Increase `gradient_accumulation_steps` for larger effective batch size

## Inference Example

After training, test on new images:

```python
image = PIL.Image.open("formula.png")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Write the LaTeX representation for this image."}
        ]
    }
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, temperature=1.0, top_p=0.95, top_k=64)
result = processor.batch_decode(outputs)[0]
print(result)
```

## References

- Gemma-3 Vision Notebook: https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Vision.ipynb
- Gemma-3N Vision Notebook: https://github.com/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Vision.ipynb
- Unsloth GitHub: https://github.com/unslothai/unsloth
- Gemma-3 Model: https://huggingface.co/google/gemma-3-4b
- LaTeX OCR Dataset: https://huggingface.co/datasets/unsloth/LaTeX_OCR
