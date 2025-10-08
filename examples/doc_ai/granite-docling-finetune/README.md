# Granite-Docling Fine-tuning Example

Fine-tune the [Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) vision-language model for document extraction tasks using TRL SFTTrainer with LoRA/DoRA adapters.

## Features

- **Efficient fine-tuning**: Uses 4-bit quantization + LoRA for single GPU training
- **Vision-language model**: Fine-tunes text tower only, vision tower frozen
- **Proper masking**: Only trains on assistant responses (DocTags)
- **Chat formatting**: Uses Granite's native chat template
- **Memory efficient**: Works on 24GB GPU (A10/RTX 6000 Ada)
- **TRL integration**: Built on HuggingFace TRL library

## Quick Start

### 1. Build the Docker image

```bash
./build.sh
```

### 2. Prepare your dataset

Create a JSON or JSONL file with the following structure:

```json
[
  {
    "image_path": "/path/to/image1.png",
    "prompt": "Extract text and layout as DocTags.",
    "doctags": "<doc>...</doc>"
  },
  {
    "image_path": "/path/to/image2.png",
    "prompt": "Extract text and layout as DocTags.",
    "doctags": "<doc>...</doc>"
  }
]
```

Or for JSONL:

```jsonl
{"image_path": "/path/to/image1.png", "prompt": "Extract text and layout as DocTags.", "doctags": "<doc>...</doc>"}
{"image_path": "/path/to/image2.png", "prompt": "Extract text and layout as DocTags.", "doctags": "<doc>...</doc>"}
```

### 3. Fine-tune the model

```bash
# Basic usage with default settings
./train.sh --train-data /path/to/data.jsonl

# Custom hyperparameters
./train.sh \
  --train-data /path/to/data.jsonl \
  --output-dir ./my_model \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --lora-r 16
```

## Configuration

### Environment Variables

You can also set these via environment variables:

```bash
export TRAIN_DATA="/path/to/data.jsonl"
export OUTPUT_DIR="./outputs/granite_docling_sft"
export BATCH_SIZE=1
export GRAD_ACCUM=8
export NUM_EPOCHS=2
export LR=1e-4
export MAX_SEQ_LEN=2048
export LORA_R=16
export LORA_ALPHA=32
export USE_DORA=true
export BF16=true

./train.sh
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-data` | `docling_dpbench` | Path to training data (JSON/JSONL) or HF dataset name |
| `--output-dir` | `./outputs/granite_docling_sft` | Output directory for LoRA adapters |
| `--batch-size` | `1` | Per-device training batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--epochs` | `2` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--max-seq-len` | `2048` | Maximum sequence length |
| `--lora-r` | `16` | LoRA rank |

### LoRA Configuration

The training uses LoRA (Low-Rank Adaptation) to efficiently fine-tune only the text tower:

- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **LoRA rank (r)**: 16 (default)
- **LoRA alpha**: 32 (default)
- **LoRA dropout**: 0.05
- **DoRA**: Enabled by default (set `USE_DORA=false` to disable)
- **Quantization**: 4-bit NF4 with double quantization

## Dataset Format

The script supports three dataset formats:

### 1. Local JSON file

```json
[
  {
    "image_path": "/workspace/images/doc1.png",
    "prompt": "Extract text and layout as DocTags.",
    "doctags": "<doc><title>Example</title><para>Text content...</para></doc>"
  }
]
```

### 2. Local JSONL file

```jsonl
{"image_path": "/workspace/images/doc1.png", "prompt": "Extract text and layout as DocTags.", "doctags": "<doc>...</doc>"}
{"image_path": "/workspace/images/doc2.png", "prompt": "Extract text and layout as DocTags.", "doctags": "<doc>...</doc>"}
```

### 3. HuggingFace dataset

Set `TRAIN_DATA=dataset_name` and `TRAIN_SPLIT=train`:

```bash
export TRAIN_DATA="docling_dpbench"
export TRAIN_SPLIT="train"
export VAL_SPLIT="validation"
./train.sh
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with 24GB VRAM (A10, RTX 6000 Ada, A6000)
- **For 16GB GPUs** (T4, RTX 4000): Reduce `MAX_SEQ_LEN=1024`, `LORA_R=8`, and resize images
- **CPU**: Not recommended (very slow)
- **RAM**: 32GB+ recommended
- **Disk**: ~10GB for model + checkpoints

## Output

After training, you'll find:

```
outputs/granite_docling_sft/
├── lora_adapters/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── preprocessor_config.json
└── tokenizer files...
```

## Inference with Fine-tuned Model

### Using transformers

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

# Load base model
processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M")
model = AutoModelForVision2Seq.from_pretrained("ibm-granite/granite-docling-258M")

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "./outputs/granite_docling_sft/lora_adapters")
model.eval()

# Inference
image = Image.open("document.png")
messages = [
    {"role": "system", "content": "You are a precise document extraction model. Output valid DocTags only."},
    {"role": "user", "content": [{"type": "text", "text": "Extract text and layout as DocTags."}]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[[image]], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Using vLLM (for production)

```bash
# Merge adapters first (optional but recommended for vLLM)
python -c "
from transformers import AutoModelForVision2Seq
from peft import PeftModel

model = AutoModelForVision2Seq.from_pretrained('ibm-granite/granite-docling-258M')
model = PeftModel.from_pretrained(model, './outputs/granite_docling_sft/lora_adapters')
model = model.merge_and_unload()
model.save_pretrained('./merged_model')
"

# Serve with vLLM
vllm serve ./merged_model --dtype float32
```

**Note**: On older GPUs without bfloat16 support, use `--dtype float32` to avoid "!!!!" outputs.

## Training Tips

1. **Start small**: Test with a small dataset first to verify everything works
2. **Monitor GPU memory**: Use `nvidia-smi` to check VRAM usage
3. **Adjust batch size**: If OOM, reduce `BATCH_SIZE` and increase `GRAD_ACCUM` proportionally
4. **Learning rate**: Start with `1e-4`, try `5e-5` or `2e-4` if needed
5. **LoRA rank**: Higher rank (32-64) = more capacity but more VRAM
6. **Sequence length**: Longer sequences = better context but more VRAM
7. **Checkpointing**: Model saves every 500 steps, adjust with `save_steps`

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce memory usage
export MAX_SEQ_LEN=1024
export LORA_R=8
export BATCH_SIZE=1
export GRAD_ACCUM=16
./train.sh --train-data data.jsonl
```

### "!!!!" Outputs During Inference

This happens on older GPUs without bfloat16 support. Solution:

```python
# Use float32 instead of bfloat16
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # Changed from bfloat16
)
```

### Dataset Loading Errors

Ensure your dataset has the correct fields:
- `image_path` or `image` or `image_file`
- `prompt` (optional, will default to "Extract text and layout as DocTags.")
- `doctags` or `ground_truth` or `labels`

### Slow Training

- Check GPU utilization with `nvidia-smi`
- Increase `BATCH_SIZE` if VRAM allows
- Use `BF16=true` on modern GPUs (Ampere/Ada)
- Reduce `MAX_SEQ_LEN` if not needed

## Advanced Usage

### Custom Target Modules

```bash
export TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
./train.sh --train-data data.jsonl
```

### 8-bit Quantization (instead of 4-bit)

Modify `train.py`:

```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # Remove 4-bit flags
)
```

### Multi-GPU Training

The script uses `device_map="auto"` which supports multi-GPU. Just ensure Docker can access all GPUs:

```bash
docker run --gpus all ...  # Already set in train.sh
```

## References

- [Granite-Docling Model](https://huggingface.co/ibm-granite/granite-docling-258M)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Docling Project](https://github.com/docling-project/docling)

## License

Apache 2.0 (same as base model)
