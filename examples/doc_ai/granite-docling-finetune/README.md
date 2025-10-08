# Granite-Docling Fine-tuning Example

Fine-tune the [Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) vision-language model for document extraction tasks using TRL SFTTrainer with LoRA/DoRA adapters.

Outputs [DocTags](https://github.com/docling-project/docling/discussions/354) - a structured markup format for document elements (text, tables, layout) optimized for LLMs.

## Features

- **Efficient fine-tuning**: Uses 4-bit quantization + LoRA for single GPU training
- **Vision-language model**: Fine-tunes text tower only, vision tower frozen
- **Proper masking**: Only trains on assistant responses (DocTags)
- **Chat formatting**: Uses Granite's native chat template
- **Memory efficient**: Works on 24GB GPU (A10/RTX 6000 Ada)
- **TRL integration**: Built on HuggingFace TRL library

## Quick Start

```bash
# 1. Build the Docker image
./build.sh

# 2. Train with default settings (uses ds4sd/docling-dpbench HuggingFace dataset)
./train.sh

# Or train with your own data
./train.sh --train-data /path/to/data.jsonl --epochs 3
```

That's it! The model will fine-tune on document extraction and save LoRA adapters to `./outputs/granite_docling_sft/lora_adapters/`.

## Configuration

### Common Options

```bash
# CLI parameters
./train.sh \
  --train-data ds4sd/docling-dpbench \  # HF dataset or path to JSONL
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --lora-r 16

# Environment variables
MAX_TRAIN_SAMPLES=20 ./train.sh  # Limit samples for testing
export TRAIN_SPLIT=test          # Dataset split (dpbench only has 'test')
export MAX_SEQ_LEN=1024          # Reduce for 16GB GPUs
export USE_DORA=false            # Disable DoRA if needed
```

**Key defaults:** 4-bit quantization, LoRA rank 16, DoRA enabled, trains text tower only (vision frozen)

**LoRA target layers:** Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP (`gate_proj`, `up_proj`, `down_proj`) in the text decoder

## Custom Data

**Required format:** JSONL with `image_path`, `prompt`, and `doctags` fields:

```jsonl
{"image_path": "/path/to/doc1.png", "prompt": "Extract text and layout as DocTags.", "doctags": "<doc><title>Title</title><para>Text...</para></doc>"}
{"image_path": "/path/to/doc2.png", "prompt": "Extract text and layout as DocTags.", "doctags": "<doc><para>More text...</para></doc>"}
```

**Fields:**
- `image_path`: Path to document image (PNG/JPG)
- `prompt`: Instruction text (can be any document extraction prompt)
- `doctags`: Ground truth output (DocTags XML or plain text)

Then train:
```bash
./train.sh --train-data /path/to/data.jsonl
```

**Helper script** (optional): `prepare_data.py` can create sample data or validate your dataset:
```bash
# Create dummy samples for testing
python prepare_data.py --mode sample --output test.jsonl --num-samples 10

# Validate your dataset
python prepare_data.py --mode validate --dataset my_data.jsonl
```

**Note:** The `convert` mode in `prepare_data.py` is a template - you'll need to adapt it to your annotation format.

## Hardware Requirements

- **Tested on**: NVIDIA A10 (24GB VRAM)
- **16GB GPU**: Set `MAX_SEQ_LEN=1024 LORA_R=8`

**Output structure:**
```
outputs/granite_docling_sft/
├── lora_adapters/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── preprocessor_config.json
└── tokenizer files...
```

## Inference

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

# Load model with LoRA adapters
processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M")
model = AutoModelForVision2Seq.from_pretrained("ibm-granite/granite-docling-258M")
model = PeftModel.from_pretrained(model, "./outputs/granite_docling_sft/lora_adapters")

# Run inference
image = Image.open("document.png")
messages = [
    {"role": "system", "content": "You are a precise document extraction model."},
    {"role": "user", "content": [{"type": "text", "text": "Extract text and layout."}]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[[image]], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

**Out of Memory:** Reduce `MAX_SEQ_LEN=1024 LORA_R=8 GRAD_ACCUM=16`

**Slow training:** Check `nvidia-smi` for GPU utilization, increase `BATCH_SIZE` if possible

**Dataset errors:** Ensure JSONL has `image_path`, `prompt`, and `doctags` fields

## References

- [Granite-Docling Model](https://huggingface.co/ibm-granite/granite-docling-258M)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Docling Project](https://github.com/docling-project/docling)

## License

Apache 2.0 (same as base model)
