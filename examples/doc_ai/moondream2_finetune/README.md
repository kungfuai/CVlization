# Moondream2 Fine-tuning Example

Fine-tune [Moondream2](https://huggingface.co/vikhyatk/moondream2) (1.86B parameter vision-language model) for document understanding tasks using pure PyTorch.

## Features

- **Pure PyTorch**: No TRL, Unsloth, or other frameworks - just standard PyTorch training loop
- **Efficient**: Only trains text model (vision encoder frozen for datasets <100k images)
- **Small model**: 1.86B parameters, runs on T4 GPU
- **Document Q&A**: Fine-tune for document-specific question answering
- **Apache 2.0 license**: Fully open source

## Quick Start

```bash
# 1. Build the Docker image
./build.sh

# 2. Prepare captcha dataset (or use your own data)
python prepare_captcha_data.py --output-dir data/captcha --split train --subset 10

# 3. Fine-tune with the captcha dataset
./train.sh \
  --data data/captcha/train_subset_10.jsonl \
  --epochs 1 \
  --batch-size 1 \
  --lr 3e-6 \
  --grad-accum 1

# 4. Test the fine-tuned model
python test_model.py \
  --model ./outputs/moondream2_ft \
  --image data/captcha/images/captcha_train_00000.png \
  --question "What is the text in this captcha image?"
```

## Data Format

Create a JSONL file with image paths and Q&A pairs:

```jsonl
{"image": "path/to/doc1.jpg", "question": "What is the total amount?", "answer": "$1,234.56"}
{"image": "path/to/doc2.jpg", "question": "What is the invoice number?", "answer": "INV-2024-001"}
{"image": "path/to/doc3.jpg", "question": "Who is the recipient?", "answer": "John Doe"}
```

**Fields:**
- `image` or `image_path`: Path to document image (relative or absolute)
- `question` or `prompt`: Question about the document
- `answer` or `response`: Expected answer (ground truth)

### Quick Start with Captcha Dataset

We provide a script to download the [captcha-images dataset](https://huggingface.co/datasets/project-sloth/captcha-images) (10,000 images, 6-digit numbers):

```bash
# Download full training set (6,000 images)
python prepare_captcha_data.py --output-dir data/captcha --split train

# Or create a small subset for quick testing (10 samples)
python prepare_captcha_data.py --output-dir data/captcha --split train --subset 10

# Train on the subset
./train.sh --data data/captcha/train_subset_10.jsonl --epochs 1 --batch-size 4
```

**Dataset details:**
- 10,000 captcha images (6,000 train / 2,000 val / 2,000 test)
- 200x200 pixels, 6-digit numbers
- License: WTFPL (Do What You Want)
- No API key required (downloads from HuggingFace)

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | `data/train.jsonl` | Path to training data (JSONL) |
| `--output-dir` | `./outputs/moondream2_ft` | Output directory for fine-tuned model |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `1` | Batch size (always 1, use grad_accum for effective batch size) |
| `--lr` | `3e-6` | Learning rate (official recommendation) |
| `--grad-accum` | `128` | **Gradient accumulation steps (CRITICAL - official uses 128!)** |

### Environment Variables

```bash
export DATA_PATH="data/train.jsonl"
export OUTPUT_DIR="./my_model"
export EPOCHS=2
export BATCH_SIZE=6
export LR=3e-5

./train.sh
```

## Hardware Requirements

- **Tested on**: NVIDIA T4 (16GB VRAM), A10 (24GB VRAM)
- **Memory usage**:
  - Model uses FP16 for efficiency (~3.5GB model + optimizer states)
  - Standard causal attention: ~6GB GPU memory
  - Bidirectional attention (experimental): ~10GB GPU memory (requires dedicated GPU)
- **Recommended settings for T4/A10**:
  - Batch size: 1 (use `--grad-accum` for effective larger batches)
  - Learning rate: `3e-6` (official Moondream recommendation)
- **Training time**: ~3 seconds per step on T4

### Tested Training Results

**Small test (10 samples, 1 epoch, T4):**
- Loss: 25.14 → 4.36 (82% reduction)
- Training speed: ~3 steps/second
- Memory: ~12GB VRAM used

**Full training (1000 samples, 1 epoch, A10):**
- Token accuracy: 27% → 81-100% (final steps at 87.5%)
- Validation accuracy: 30% exact match (3/10 correct)
- Training speed: ~1.1 steps/second
- Memory: ~16GB VRAM used
- Common errors: Off by 1-2 digits (e.g., "907565" vs "907785")

## Training Details

### What Gets Trained

- ✅ **Text model**: Fully trainable (language decoder)
- ❌ **Vision encoder**: Frozen (recommended for datasets <100k images)

### Why Freeze Vision Encoder?

For most tasks, fine-tuning the vision encoder provides little benefit and can hurt performance. Only unfreeze for datasets with 100k+ images.

### Architecture

- **Base model**: Moondream2 (1.86B params)
- **Vision encoder**: SigLIP
- **Text model**: Phi-1.5 based decoder
- **Framework**: Pure PyTorch (no TRL/PEFT)

## Inference with Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "./outputs/moondream2_ft",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("./outputs/moondream2_ft")

# Run inference
image = Image.open("document.jpg")
question = "What is the total amount?"

# Encode and query
encoded = model.encode_image(image)
answer = model.answer_question(encoded, question, tokenizer)
print(answer)
```

## Example Use Cases

1. **Invoice processing**: Extract amounts, dates, invoice numbers
2. **Form understanding**: Parse structured documents
3. **Receipt OCR**: Extract items, totals, vendor information
4. **Document Q&A**: Answer questions about document content
5. **Layout understanding**: Identify document structure

## Troubleshooting

**Out of Memory (OOM)**:
```bash
# 1. Standard causal attention uses ~6GB (recommended)
./train.sh --data ... --grad-accum 8

# 2. If still OOM, ensure no other processes are using GPU
nvidia-smi  # Check GPU usage

# 3. Bidirectional attention (--use_bidirectional_image_attn) requires ~10GB
# and may cause OOM on shared GPUs - only use if you have dedicated GPU
```

**OOM with bidirectional attention**:
- The `--use_bidirectional_image_attn` flag creates explicit 4D attention masks that require ~4GB extra memory
- Cannot be mitigated with float16 (already enabled) or lower batch size (already at 1)
- Use standard causal attention instead (default, better accuracy anyway)
- See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed comparison

**Slow training**:
- Check GPU utilization with `nvidia-smi`
- Increase batch size if VRAM allows
- Reduce image resolution in preprocessing

**Poor results**:
- Increase epochs (try 3-5)
- Verify data quality and format
- Check that answers are in the training data
- Ensure diverse training examples
- Lower gradient accumulation (4-8 works better than 128 for small datasets)

## Advanced Usage

### Attention Mechanism Options

The training script supports two attention mechanisms:

**1. Standard Causal Attention (Default, Recommended)**
```bash
./train.sh --data ... --val_data ...
```
- Uses implicit causal masking (memory efficient)
- Better accuracy on captcha task (25-30%)
- ~6GB GPU memory
- Works on consumer GPUs

**2. 730-Token Bidirectional Attention (Experimental)**
```bash
./train.sh --data ... --val_data ... --use_bidirectional_image_attn
```
- Bidirectional attention for first 730 tokens (image), causal for text
- Inspired by official Moondream native implementation
- Requires ~10GB GPU memory (explicit 4D attention masks)
- Lower accuracy on captcha task (20%)
- May be beneficial for tasks similar to Moondream's original training

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed comparison and benchmarks.

### Custom Model Revision

By default uses stable revision `2024-07-23`. To use latest:

Edit `train.py` line 382:
```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    trust_remote_code=True,
    # revision='2024-07-23',  # Comment out for latest
    torch_dtype=torch.float16
)
```

### Multi-GPU Training

The code currently uses single GPU. For multi-GPU, wrap model with `DataParallel`:

```python
model = torch.nn.DataParallel(model)
```

## References

- [Moondream2 Model](https://huggingface.co/vikhyatk/moondream2)
- [Roboflow Fine-tuning Tutorial](https://blog.roboflow.com/finetuning-moondream2/)
- [Moondream GitHub](https://github.com/vikhyat/moondream)

## License

Apache 2.0 (same as Moondream2 base model)
