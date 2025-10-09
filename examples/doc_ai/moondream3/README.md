## Moondream3 - Advanced Vision Language Model for OCR & Visual Reasoning

This example demonstrates how to use Moondream3, a 9B parameter Mixture-of-Experts (MoE) vision language model with frontier-level visual reasoning capabilities while maintaining fast inference.

**Original Model**: [moondream/moondream3-preview on HuggingFace](https://huggingface.co/moondream/moondream3-preview)

### Features

- **Advanced Architecture**: 9B total params, 2B active (MoE) - efficient and powerful
- **Extended Context**: 32k token context window (vs 2k in Moondream2)
- **OCR Capabilities**: Improved text transcription with natural reading order
- **Image Captioning**: Generate short, normal, or long captions
- **Visual QA**: Ask questions about images
- **Object Detection**: Detect and locate objects in images
- **Pointing**: Identify object coordinates in images
- **Structured Output**: Convert images to markdown tables and structured formats
- **Fast Inference**: Optimized MoE architecture with model compilation
- **Self-contained**: Fully dockerized with all dependencies

### Prerequisites

1. NVIDIA GPU with at least 12GB VRAM (RTX 4090, A10G, A100, etc.)
2. Docker with NVIDIA runtime support
3. **HuggingFace account with model access** (required - Moondream3 is a gated model)
   - Create account: https://huggingface.co/join
   - Request access: https://huggingface.co/moondream/moondream3-preview
   - Get token: https://huggingface.co/settings/tokens

### Quickstart

> **Important**: Moondream3 is a gated model. You must request access and wait for approval before using it.

1. **Request access** to the model at https://huggingface.co/moondream/moondream3-preview
   - Click "Agree and access repository" button
   - Wait for approval (usually instant or within a few hours)

2. Set your HuggingFace token (required):
```bash
export HF_TOKEN=your_huggingface_token
```
   Get your token from: https://huggingface.co/settings/tokens

3. Build the Docker image:
```bash
bash examples/doc_ai/moondream3/build.sh
```

4. Run OCR on an image:
```bash
bash examples/doc_ai/moondream3/predict.sh --image examples/sample.jpg
```

### Usage

#### OCR - Text Transcription

Extract text from a document:
```bash
bash examples/doc_ai/moondream3/predict.sh \
    --image path/to/document.jpg \
    --task ocr \
    --ocr-mode ordered
```

OCR modes:
- `default` - Basic text transcription
- `ordered` - Text in natural reading order (recommended)
- `detailed` - Detailed extraction with layout structure
- `markdown` - Convert document to markdown format

#### Image Captioning

Generate image captions:
```bash
bash examples/doc_ai/moondream3/predict.sh \
    --image path/to/image.jpg \
    --task caption \
    --caption-length normal
```

Caption lengths: `short`, `normal`, `long`

#### Visual Question Answering

Ask questions about an image:
```bash
bash examples/doc_ai/moondream3/predict.sh \
    --image path/to/image.jpg \
    --task query \
    --prompt "How many people are in this image?"
```

#### Object Detection

Detect objects in an image:
```bash
bash examples/doc_ai/moondream3/predict.sh \
    --image path/to/image.jpg \
    --task detect
```

Detect specific objects:
```bash
bash examples/doc_ai/moondream3/predict.sh \
    --image invoice.jpg \
    --task detect \
    --object "table"
```

#### Pointing (Coordinate Detection)

Get coordinates of objects:
```bash
bash examples/doc_ai/moondream3/predict.sh \
    --image photo.jpg \
    --task point \
    --object "person"
```

### Advanced Usage

#### JSON Output

Save results as JSON with metadata:
```bash
bash examples/doc_ai/moondream3/predict.sh \
    --image document.jpg \
    --output results.json \
    --format json
```

#### Direct Docker Run

For more control:
```bash
docker run --runtime nvidia \
    -v $(pwd)/examples/doc_ai/moondream3:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    moondream3 \
    python3 predict.py \
        --image examples/sample.jpg \
        --task ocr \
        --ocr-mode markdown
```

#### Disable Model Compilation

For better compatibility (slower inference):
```bash
docker run --runtime nvidia \
    -v $(pwd)/examples/doc_ai/moondream3:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    moondream3 \
    python3 predict.py \
        --image image.jpg \
        --task caption \
        --no-compile
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image or URL (default: `examples/sample.jpg`)
- `--model-id` - HuggingFace model ID (default: `moondream/moondream3-preview`)
- `--task` - Task type: `ocr`, `caption`, `query`, `detect`, `point` (default: `ocr`)
- `--prompt` - Custom prompt for OCR or query tasks
- `--ocr-mode` - OCR preset: `default`, `ordered`, `detailed`, `markdown` (default: `ordered`)
- `--caption-length` - Caption length: `short`, `normal`, `long` (default: `normal`)
- `--object` - Object name for detect or point tasks
- `--output` - Output file path (default: `outputs/result.txt`)
- `--format` - Output format: `txt` or `json` (default: `txt`)
- `--device` - Device: `cuda` or `cpu` (default: `cuda`)
- `--no-compile` - Disable model compilation (slower but more compatible)

### Model Details

- **Name**: Moondream3 Preview
- **Size**: 9B total parameters, 2B active (MoE)
- **Architecture**: Mixture-of-Experts Vision Language Model
  - 24 layers (4 dense + 20 MoE)
  - 64 experts per MoE layer (8 active per token)
  - SigLIP-based vision encoder
  - SuperBPE tokenizer
- **Context**: 32k tokens (16x larger than Moondream2)
- **License**: Business Source License 1.1 with Additional Use Grant
- **Model Page**: [https://huggingface.co/moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview)

### Performance

- **Speed**: Fast inference with MoE architecture and model compilation
- **Memory**: ~12GB VRAM for inference
- **Quality**: Frontier-level visual reasoning
- **Efficiency**: 2B active params provide excellent speed/quality tradeoff

### Capabilities

1. **OCR**: Advanced text transcription with reading order and structure
2. **Captioning**: Detailed image descriptions
3. **Visual QA**: Answer complex questions about images
4. **Object Detection**: Identify and locate objects
5. **Pointing**: Return coordinates of objects in images
6. **Document Understanding**: Extract and structure information from documents
7. **Visual Reasoning**: Complex visual understanding tasks

### Supported Document Types

- Scanned documents
- PDFs (convert to images first)
- Screenshots
- Photographs
- Invoices and receipts
- Forms and tables
- Mixed content documents
- Complex visual layouts

### Output Examples

#### OCR Output
```
INVOICE
Date: October 9, 2025
Invoice #: 12345

Item          Qty    Price
Widget A        2    $10.00
Widget B        1    $25.00

Total: $45.00
```

#### Markdown Output
```markdown
# INVOICE

**Date**: October 9, 2025
**Invoice #**: 12345

| Item | Qty | Price |
|------|-----|-------|
| Widget A | 2 | $10.00 |
| Widget B | 1 | $25.00 |

**Total**: $45.00
```

#### Caption Output
```
A simple invoice document showing two line items for widgets with quantities
and prices, totaling $45.00, dated October 9, 2025.
```

#### Query Output
```
Q: What is the total amount?
A: The total amount is $45.00
```

#### Detection Output
```
Q: Detect all objects in this image
A: This image contains a table with 2 rows, text labels (Item, Qty, Price),
   numeric values, and header information including date and invoice number.
```

### How It Works

Moondream3 automatically downloads from HuggingFace on first run (~18GB). The model is cached in `data/container_cache` and reused for subsequent runs. The model uses a Mixture-of-Experts architecture where only 2B out of 9B parameters are active for each token, providing excellent efficiency.

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Close other GPU applications
2. Use a GPU with more VRAM (12GB+ recommended)
3. Try CPU inference with `--device cpu` (much slower)
4. Disable model compilation with `--no-compile`

#### Model Download Issues

If the model fails to download:
1. **Verify you have requested and been granted access** to the gated model at https://huggingface.co/moondream/moondream3-preview
2. Set `HF_TOKEN` environment variable with a valid token
3. Check internet connectivity
4. Ensure sufficient disk space (~18GB for model)

Common error: "Cannot access gated repo" means you need to request access and/or set HF_TOKEN

#### Slow Inference

For faster inference:
1. Ensure GPU is being used (`--device cuda`)
2. Pre-download model to cache
3. Use bfloat16 precision (automatic on CUDA)
4. Enable model compilation (default, use `--no-compile` to disable)

### Testing

Run the smoke test to verify installation:
```bash
bash examples/doc_ai/moondream3/test.sh
```

### Directory Structure

```
examples/doc_ai/moondream3/
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── predict.py          # Main inference script
├── build.sh            # Build Docker image
├── predict.sh          # Run inference wrapper
├── test.sh             # Smoke test
├── README.md           # This file
├── .gitignore          # Git ignore rules
├── examples/           # Sample images
└── outputs/            # Results saved here
```

### Output Location

By default, outputs are saved to:
- `examples/doc_ai/moondream3/outputs/`

### Model Cache

Model weights are cached in:
- `data/container_cache/huggingface/`

Size: ~18GB (persists across runs)

### Comparison with Other Models

| Feature | Moondream3 | Moondream2 | dots.ocr |
|---------|-----------|-----------|----------|
| Size | 9B (2B active) | 1.93B | 1.7B |
| Context | 32k tokens | 2k tokens | 4k tokens |
| OCR | ✅ Excellent | ✅ Good | ✅ Excellent |
| Captioning | ✅ Yes | ✅ Yes | ❌ No |
| Visual QA | ✅ Yes | ✅ Yes | ❌ No |
| Object Detection | ✅ Yes | ✅ Basic | ❌ No |
| Pointing | ✅ Yes | ❌ No | ❌ No |
| Speed | ⚡ Fast | ⚡⚡ Very Fast | ⚡ Fast |
| Setup | 🟢 Easy | 🟢 Easy | 🟡 Moderate |
| VRAM | 12GB+ | 6GB+ | 8GB+ |

### Upgrade from Moondream2

Moondream3 offers significant improvements over Moondream2:
- **16x larger context** (32k vs 2k tokens)
- **Better visual reasoning** with MoE architecture
- **Additional capabilities**: pointing, advanced object detection
- **Structured output**: markdown conversion, better formatting
- **Same easy API** - minimal code changes needed

### References

- [Moondream3 on HuggingFace](https://huggingface.co/moondream/moondream3-preview)
- [Official Website](https://moondream.ai/)
- [Moondream3 Blog Post](https://moondream.ai/blog/moondream-3-preview)
- [Moondream Documentation](https://docs.moondream.ai/)
- [GitHub Repository](https://github.com/vikhyat/moondream)

### License

This example uses Moondream3 which is licensed under the Business Source License 1.1 with an Additional Use Grant. Commercial use is allowed for self-hosting within your company, but third-party service offerings are restricted. See the [HuggingFace model page](https://huggingface.co/moondream/moondream3-preview) for full license details.
