## GPT-OSS (OpenAI) Fine-tuning with Unsloth

This example demonstrates how to fine-tune OpenAI's GPT-OSS 20B model using Unsloth for faster and more memory-efficient training.

### Features

- **OpenAI GPT-OSS 20B**: State-of-the-art open language model (Apache 2.0 license)
- **Fast Training**: 3x faster inference, 50% less VRAM with Unsloth
- **128k Context**: Supports extremely long context (using 1k for testing)
- **LoRA Fine-tuning**: Efficient parameter updates with rank-8 LoRA
- **Reasoning Dataset**: Uses HuggingFace Multilingual-Thinking for reasoning tasks

### Prerequisites

1. NVIDIA GPU with at least 16GB VRAM (A10, A100, RTX 4090, etc.)
2. HuggingFace account and token for model access
3. Docker with NVIDIA runtime support

### Quickstart

1. Set your HuggingFace token:
```bash
export HF_TOKEN=your_huggingface_token
```

2. Build the Docker image:
```bash
bash examples/text_gen/gpt_oss_finetune/build.sh
```

3. Run training:
```bash
bash examples/text_gen/gpt_oss_finetune/train.sh
```

### Training Details

- **Model**: unsloth/gpt-oss-20b (20 billion parameters)
- **Dataset**: HuggingFaceH4/Multilingual-Thinking (reasoning tasks)
- **Max Steps**: 20 (for quick testing)
- **Batch Size**: 1 per device with 4x gradient accumulation
- **Learning Rate**: 2e-4
- **LoRA Rank**: 8 (optimized for larger base model)
- **Quantization**: 4-bit (bitsandbytes)
- **Context Length**: 1024 tokens (can be extended to 128k)

Training takes approximately 2-3 hours on an A10 GPU for the full example.

#### Dataset

The pre-configured dataset is **HuggingFaceH4/Multilingual-Thinking**, which contains reasoning tasks with chain-of-thought examples in chat format (ShareGPT style with `messages` column).

### Using Your Own Data

All training configuration is controlled via `config.yaml`. To use your own dataset:

1. **Edit `config.yaml`** and update the `dataset` section:

```yaml
dataset:
  path: "your-username/your-dataset"  # HuggingFace dataset or local path
  format: "sharegpt"  # Choose: sharegpt or custom
  split: "train"
  max_samples: 1000  # Optional: limit dataset size for testing
```

2. **Supported dataset formats:**

   - **`sharegpt`**: Expects column `messages` (list of chat messages with role/content)
   - **`custom`**: Expects column `text` (pre-formatted chat strings)

3. **Example for local dataset:**
```yaml
dataset:
  path: "./my_conversations.json"
  format: "custom"
```

4. **Adjust training parameters in `config.yaml`:**
```yaml
training:
  max_steps: 100
  per_device_train_batch_size: 1  # Keep low for 20B model
  learning_rate: 2.0e-4
```

### Output

The fine-tuned model will be saved to:
- `./gpt-oss-finetune/final_model/`

Checkpoints are saved every 10 steps in:
- `./gpt-oss-finetune/checkpoint-*/`

### Extending Context Length

GPT-OSS supports up to 128k context. To use longer context:

1. Update `max_seq_length` in `train.py` (e.g., 8192, 16384, or 32768)
2. Reduce batch size if needed to fit in VRAM
3. Consider using Unsloth's Flex Attention for 6x longer context

### Reference

This example is based on:
- [OpenAI GPT-OSS Announcement](https://openai.com/index/gpt-oss/)
- [Unsloth GPT-OSS Guide](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune)
- [Unsloth Blog: GPT-OSS Fine-tuning](https://unsloth.ai/blog/gpt-oss)
