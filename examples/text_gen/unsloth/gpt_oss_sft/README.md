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
bash examples/text_gen/unsloth/gpt_oss_sft/build.sh
```

3. Run training:
```bash
bash examples/text_gen/unsloth/gpt_oss_sft/train.sh
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

See [CUSTOM_DATASETS.md](../../CUSTOM_DATASETS.md) for detailed instructions on using your own datasets, including:
- Supported formats (ShareGPT, Custom)
- Local file loading (.json, .jsonl, .csv, .parquet)
- HuggingFace dataset integration
- JSON structure examples
- Troubleshooting tips

**Note:** GPT-OSS uses the `messages` column for ShareGPT format (not `conversations`).

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
