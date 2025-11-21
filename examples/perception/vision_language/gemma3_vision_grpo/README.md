# Gemma-3 Vision GRPO (Reinforcement Learning)

Fine-tune Gemma-3 vision model with GRPO for math visual reasoning.

## Features

- **GRPO Training**: Group Relative Policy Optimization for reinforcement learning
- **Math Visual Reasoning**: Fine-tune on MathVista dataset for solving math problems from images
- **Custom Reward Functions**: Separate rewards for formatting and answer correctness
- **Structured Reasoning**: Train model to produce structured outputs with `<REASONING>` and `<SOLUTION>` delimiters
- **Fast Inference**: Uses vLLM for efficient generation during training
- **Memory Efficient**: 4-bit quantization with ~20GB VRAM requirement
- **Language-Only Fine-tuning**: Does NOT train vision layers (only language model)

## Quick Start

### 1. Build the Docker image
```bash
cvl run gemma3-vision-grpo build
```

### 2. Run smoke test (60 steps, 50 samples)
```bash
cvl run gemma3-vision-grpo test
```

### 3. Full training
Edit `config.yaml` to adjust training parameters:
- Remove `max_samples` for full dataset
- Set `num_train_epochs: 2` instead of `max_steps`
- Increase `num_generations` if you have more VRAM

Then run:
```bash
cvl run gemma3-vision-grpo train
```

## Configuration

All training parameters are in `config.yaml`:

### Model Settings
- `model.name`: Use `unsloth/gemma-3-4b-it` (instruct model for GRPO)
- `model.load_in_4bit`: Enable 4-bit quantization (default: true)
- `model.use_gradient_checkpointing`: Memory optimization (default: "unsloth")

### LoRA Settings
- `lora.r`: LoRA rank (default: 16)
- `lora.alpha`: LoRA alpha (default: 16)
- `lora.finetune_vision_layers`: Train vision encoder (default: false for GRPO)
- `lora.finetune_language_layers`: Train language model (default: true)

**Important**: GRPO only trains language layers, not vision encoder.

### Dataset Settings
- `dataset.path`: HuggingFace dataset path (default: `AI4Math/MathVista`)
- `dataset.split`: Dataset split (default: "testmini")
- `dataset.max_samples`: Limit samples for testing (default: 50)
- `dataset.image_size`: Resize images to reduce memory (default: 512)

### Reward Settings
- `reward.reasoning_start/end`: Delimiters for reasoning section (default: `<REASONING>`, `</REASONING>`)
- `reward.solution_start/end`: Delimiters for solution section (default: `<SOLUTION>`, `</SOLUTION>`)
- `reward.formatting_reward`: Reward for proper delimiter usage (default: 1.0)
- `reward.correctness_reward`: Reward for correct answer (default: 2.0)

### Training Settings
- `training.per_device_train_batch_size`: Batch size per device (default: 1)
- `training.gradient_accumulation_steps`: Gradient accumulation (default: 2)
- `training.learning_rate`: Learning rate (default: 5e-6, lower than SFT)
- `training.num_generations`: Number of completions per prompt (default: 4)
- `training.max_steps`: Max training steps (default: 60 for testing)
- `training.loss_type`: GRPO loss variant (default: "dr_grpo")

## How GRPO Works

GRPO is a reinforcement learning algorithm that:

1. **Generates Multiple Completions**: For each prompt, generate `num_generations` different responses
2. **Computes Rewards**: Apply custom reward functions to each completion
3. **Updates Policy**: Use rewards to update the model to prefer high-reward outputs

### Reward Functions

The training uses two reward functions:

**Formatting Reward** (`formatting_reward_func`):
- Awards +1.0 for exactly one `<REASONING>...</REASONING>` block
- Awards +1.0 for exactly one `<SOLUTION>...</SOLUTION>` block

**Correctness Reward** (`correctness_reward_func`):
- Awards +2.0 if the solution matches the ground truth answer
- Awards 0.0 otherwise

Total possible reward per example: 4.0 (2.0 for formatting + 2.0 for correctness)

## Dataset Format

The MathVista dataset contains:
- `decoded_image`: PIL Image showing a math problem (chart, diagram, etc.)
- `question`: Text question about the image
- `answer`: Numeric ground truth answer

The script converts to conversation format with reasoning structure:
```python
[
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is the value of x? Provide reasoning between <REASONING> and </REASONING> and solution between <SOLUTION> and </SOLUTION>"}
        ]
    }
]
```

Expected model output:
```
<REASONING>
From the diagram, I can see that the angle is 45 degrees...
</REASONING>

<SOLUTION>
45
</SOLUTION>
```

## Memory Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: ~20GB (with 4-bit quantization)
  - Higher than SFT due to `num_generations` parallel completions
  - Reduce `num_generations` if you hit OOM
- **Disk**: ~25GB for model + cache + vLLM

## Output

The fine-tuned model is saved to `outputs/final_model/` and includes:
- LoRA adapter weights for language model
- Tokenizer configuration
- Model config

### Loading the Fine-tuned Model

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="outputs/final_model",
    load_in_4bit=True,
)

FastVisionModel.for_inference(model)
```

## Tips

1. **For Testing**: Keep `max_samples: 50` and `max_steps: 60` for quick validation
2. **For Full Training**: Remove `max_samples` and set `num_train_epochs: 2`
3. **Memory Issues**: Reduce `num_generations` from 4 to 2
4. **Better Results**: Increase `correctness_reward` to emphasize accuracy over formatting
5. **Faster Training**: GRPO converges faster than SFT (60 steps often sufficient)
6. **vLLM Requirement**: GRPO needs vLLM for fast generation; set `UNSLOTH_VLLM_STANDBY=1`

## Inference Example

After training, test on new images:

```python
import PIL.Image
from unsloth import FastVisionModel

# Load model
model, tokenizer = FastVisionModel.from_pretrained("outputs/final_model", load_in_4bit=True)
FastVisionModel.for_inference(model)

# Prepare input
image = PIL.Image.open("math_problem.png")
instruction = (
    "What is the answer? Provide reasoning between <REASONING> and </REASONING> "
    "and final answer between <SOLUTION> and (put a float here) </SOLUTION>"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]
    }
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

# Generate
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256,
                        use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
```

## GRPO vs SFT

| Aspect | SFT | GRPO |
|--------|-----|------|
| Training Type | Supervised | Reinforcement Learning |
| Vision Training | ✓ Trains vision layers | ✗ Language only |
| Rewards | Cross-entropy loss | Custom reward functions |
| Convergence | Slower (100-1000 steps) | Faster (60-100 steps) |
| Memory | ~16GB | ~20GB (due to multiple generations) |
| Use Case | General vision-language | Tasks with clear rewards |

## Custom Rewards

You can modify reward functions in `train.py`:

```python
def formatting_reward_func(completions, **kwargs):
    """Reward for proper formatting."""
    scores = []
    for completion in completions:
        score = 0
        # Add your custom logic here
        # Example: check for specific patterns, length, etc.
        scores.append(score)
    return scores

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Reward for correct answers."""
    scores = []
    for completion, gt_answer in zip(completions, answer):
        # Add your custom logic here
        # Example: extract answer, compare to ground truth
        score = 2.0 if is_correct(completion, gt_answer) else 0.0
        scores.append(score)
    return scores
```

## References

- Gemma-3 Vision GRPO Notebook: https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Vision-GRPO.ipynb
- Unsloth GitHub: https://github.com/unslothai/unsloth
- GRPO Paper: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- MathVista Dataset: https://huggingface.co/datasets/AI4Math/MathVista
- Gemma-3 Model: https://huggingface.co/google/gemma-3-4b-it
