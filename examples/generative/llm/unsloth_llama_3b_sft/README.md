## Llama 3.2 3B Fine-tuning with Unsloth

This example demonstrates how to fine-tune Llama 3.2 3B Instruct model using Unsloth for faster and more memory-efficient training.

### Features

- **Fast Training**: 2x faster than standard fine-tuning with Unsloth
- **Memory Efficient**: 70% less VRAM usage with 4-bit quantization
- **LoRA Fine-tuning**: Efficient parameter updates using Low-Rank Adaptation
- **Alpaca Dataset**: Uses the cleaned Alpaca instruction dataset

### Prerequisites

1. NVIDIA GPU with at least 8GB VRAM (recommended: 16GB+)
2. HuggingFace account and token for model access
3. Docker with NVIDIA runtime support

### Quickstart

1. Set your HuggingFace token:
```bash
export HF_TOKEN=your_huggingface_token
```

2. Build the Docker image:
```bash
bash examples/text_gen/unsloth/llama_3b_sft/build.sh
```

3. Run training:
```bash
bash examples/text_gen/unsloth/llama_3b_sft/train.sh
```

### Training Details

- **Model**: unsloth/Llama-3.2-3B-Instruct
- **Dataset**: yahma/alpaca-cleaned (instruction-following)
- **Max Steps**: 1000
- **Batch Size**: 2 per device with 4x gradient accumulation
- **Learning Rate**: 2e-4
- **LoRA Rank**: 16
- **Quantization**: 4-bit (bitsandbytes)

The training takes approximately 1-2 hours on a modern GPU (e.g., RTX 4090, A100).

#### Dataset

The pre-configured dataset is **yahma/alpaca-cleaned**, which contains ~52K instruction-following examples in the format:
- `instruction`: The task description
- `input`: Optional context or input
- `output`: The expected response

This dataset is ideal for teaching models to follow instructions and respond helpfully.

### Using Your Own Data

See [CUSTOM_DATASETS.md](../CUSTOM_DATASETS.md) for detailed instructions on using your own datasets, including:
- Supported formats (Alpaca, ShareGPT, Custom)
- Local file loading (.json, .jsonl, .csv, .parquet)
- HuggingFace dataset integration
- JSON structure examples
- Troubleshooting tips

### Output

The fine-tuned model will be saved to:
- `./llama-alpaca-finetune/final_model/`

Checkpoints are saved every 100 steps in:
- `./llama-alpaca-finetune/checkpoint-*/`

### Reference

This example is based on:
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai)
- [KDnuggets Tutorial](https://www.kdnuggets.com/fine-tuning-llama-using-unsloth)
