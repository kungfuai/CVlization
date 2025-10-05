## Qwen 2.5 7B Fine-tuning with Unsloth

This example demonstrates how to fine-tune Qwen 2.5 7B model using Unsloth for faster and more memory-efficient training.

### Features

- **Qwen 2.5 7B Instruct**: Powerful 7B parameter model from Alibaba Cloud
- **Fast Training**: 2x faster inference, 70% less VRAM with Unsloth
- **40k Context Support**: Supports long context (using 2k for testing)
- **LoRA Fine-tuning**: Efficient parameter updates with rank-16 LoRA
- **Alpaca Dataset**: Clean instruction-following dataset

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
bash examples/text_gen/qwen_7b_finetune/build.sh
```

3. Run training:
```bash
bash examples/text_gen/qwen_7b_finetune/train.sh
```

### Training Details

- **Model**: unsloth/Qwen2.5-7B-Instruct (7 billion parameters)
- **Dataset**: yahma/alpaca-cleaned (instruction-following tasks)
- **Max Steps**: 20 (for quick testing)
- **Batch Size**: 2 per device with 4x gradient accumulation
- **Learning Rate**: 2e-4
- **LoRA Rank**: 16
- **Quantization**: 4-bit (bitsandbytes)
- **Context Length**: 2048 tokens (can be extended to 40k)

Training takes approximately 1-2 hours on an A10 GPU for the full example.

#### Dataset

The pre-configured dataset is **yahma/alpaca-cleaned**, which contains ~52K instruction-following examples formatted for Qwen's chat template.

### Using Your Own Data

See [CUSTOM_DATASETS.md](../CUSTOM_DATASETS.md) for detailed instructions on using your own datasets, including:
- Supported formats (Alpaca, ShareGPT, Custom)
- Local file loading (.json, .jsonl, .csv, .parquet)
- HuggingFace dataset integration
- JSON structure examples
- Troubleshooting tips

### Output

The fine-tuned model will be saved to:
- `./qwen-7b-finetune/final_model/`

Checkpoints are saved every 10 steps in:
- `./qwen-7b-finetune/checkpoint-*/`

### Extending Context Length

Qwen 2.5 supports up to 40k context. To use longer context:

1. Update `max_seq_length` in `train.py` (e.g., 4096, 8192, or 16384)
2. Reduce batch size if needed to fit in VRAM
3. Consider using gradient checkpointing for memory efficiency

### Reference

This example is based on:
- [Qwen 2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Unsloth Documentation](https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune)
- [Qwen 2.5 Coder Fine-tuning Blog](https://unsloth.ai/blog/qwen-coder)
