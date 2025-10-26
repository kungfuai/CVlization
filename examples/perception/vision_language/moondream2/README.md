## Moondream2 - Vision Language Model for OCR & Document Understanding

This example demonstrates how to use Moondream2, a compact 1.93B parameter vision language model designed for efficient document OCR, image captioning, and visual question answering.

### Features

- Compact Size: Only 1.93B parameters - runs on consumer GPUs
- OCR Capabilities: Improved text transcription with natural reading order
- Image Captioning: Generate short, normal, or long captions
- Visual QA: Ask questions about images
- Fast Inference: Optimized for speed
- Multiple Tasks: OCR, captioning, object detection, and more
- Self-contained: Fully dockerized with all dependencies

### Prerequisites

1. NVIDIA GPU with at least 6GB VRAM (RTX 3060, A10, etc.)
2. Docker with NVIDIA runtime support
3. HuggingFace account (optional, for faster downloads)

### Quickstart

1. Set your HuggingFace token (optional but recommended):
```bash
export HF_TOKEN=your_huggingface_token
```

2. Build the Docker image:
```bash
bash examples/perception/vision_language/moondream2/build.sh
```

3. Run OCR on an image:
```bash
bash examples/perception/vision_language/moondream2/predict.sh --image examples/sample.jpg
```

### Usage

#### OCR - Text Transcription

Extract text from a document:
```bash
bash examples/perception/vision_language/moondream2/predict.sh \
    --image path/to/document.jpg \
    --task ocr \
    --ocr-mode ordered
```

OCR modes:
- `default` - Basic text transcription
- `ordered` - Text in natural reading order (recommended)
- `detailed` - Detailed extraction with layout structure

#### Image Captioning

Generate image captions:
```bash
bash examples/perception/vision_language/moondream2/predict.sh \
    --image path/to/image.jpg \
    --task caption \
    --caption-length normal
```

Caption lengths: `short`, `normal`, `long`

#### Visual Question Answering

Ask questions about an image:
```bash
bash examples/perception/vision_language/moondream2/predict.sh \
    --image path/to/image.jpg \
    --task query \
    --prompt "How many people are in this image?"
```

#### Custom Prompts

Use custom prompts for specific tasks:
```bash
bash examples/perception/vision_language/moondream2/predict.sh \
    --image invoice.jpg \
    --task query \
    --prompt "Extract the invoice number and total amount"
```

### Advanced Usage

#### JSON Output

Save results as JSON with metadata:
```bash
bash examples/perception/vision_language/moondream2/predict.sh \
    --image document.jpg \
    --output results.json \
    --format json
```

#### Direct Docker Run

For more control:
```bash
docker run --runtime nvidia \
    -v $(pwd)/examples/perception/vision_language/moondream2:/workspace \
    -v $(pwd)/${CVL_HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    moondream2 \
    python3 predict.py \
        --image examples/sample.jpg \
        --task ocr \
        --ocr-mode ordered
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image or URL (default: `examples/sample.jpg`)
- `--model-id` - HuggingFace model ID (default: `vikhyatk/moondream2`)
- `--revision` - Model version (default: `2024-08-26`)
- `--task` - Task type: `ocr`, `caption`, or `query` (default: `ocr`)
- `--prompt` - Custom prompt for OCR or query tasks
- `--ocr-mode` - OCR preset: `default`, `ordered`, `detailed` (default: `ordered`)
- `--caption-length` - Caption length: `short`, `normal`, `long` (default: `normal`)
- `--output` - Output file path (default: `outputs/result.txt`)
- `--format` - Output format: `txt` or `json` (default: `txt`)
- `--device` - Device: `cuda`, `mps`, or `cpu` (default: auto-detect)

### Model Details

- Name: Moondream2
- Size: 1.93B parameters
- Revision: 2024-08-26
- Architecture: Vision Language Model
- License: Apache 2.0
- Context: Efficient for document OCR and visual understanding

### Performance

**Benchmarks on NVIDIA A10 (23GB VRAM)**:
- Model load time: ~3s
- Inference time: ~1s (after warmup)
- Total time: ~4s for OCR task

**Requirements**:
- **Memory**: ~6GB VRAM for inference
- **Quality** (benchmark scores):
  - DocVQA: 79.3
  - TextVQA: 76.3
  - OCRBench: 61.2

### Capabilities

1. **OCR**: Text transcription with reading order
2. **Captioning**: Image descriptions
3. **Visual QA**: Answer questions about images
4. **Object Detection**: Identify objects in images
5. **Document Understanding**: Extract structured information

### Supported Document Types

- Scanned documents
- PDFs (convert to images first)
- Screenshots
- Photographs
- Invoices and receipts
- Forms and tables
- Mixed content documents

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

#### Caption Output
```
A simple invoice document with line items and a total amount.
```

#### Query Output
```
Q: What is the total amount?
A: The total amount is $45.00
```

### How It Works

This is a **simple, clean implementation** using standard transformers library with `trust_remote_code=True`. The model code is automatically downloaded from HuggingFace, keeping everything up-to-date and maintainable.

Moondream2 automatically downloads from HuggingFace on first run (~4GB). The model is cached in `$HOME/.cache/huggingface` and reused for subsequent runs.

**Key Implementation Details:**
- Uses `AutoModelForCausalLM` with `trust_remote_code=True` (standard transformers approach)
- Revision `2024-08-26` - most stable with best OCR quality
- Transformers `4.43.0` - compatible version (newer 4.57.0 gives incomplete OCR)
- Old API (`answer_question`, `caption`) - more reliable than new `query()` API
- Auto-detects optimal device and dtype (bfloat16 on CUDA, float16 on MPS, float32 on CPU)
- Simple ~290 line implementation with excellent docstrings

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Close other GPU applications
2. Use a GPU with more VRAM
3. Try CPU inference with `--device cpu` (slower)

#### Model Download Issues

If the model fails to download:
1. Set `HF_TOKEN` environment variable
2. Check internet connectivity
3. Verify HuggingFace access

#### Slow Inference

For faster inference:
1. Ensure GPU is being used (`--device cuda`)
2. Pre-download model to cache
3. Use bfloat16 precision (automatic on CUDA)

### Testing

Run the smoke test to verify installation:
```bash
bash examples/perception/vision_language/moondream2/test.sh
```

### Directory Structure

```
examples/perception/vision_language/moondream2/
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
- `examples/perception/vision_language/moondream2/outputs/`

### Model Cache

Model weights are cached in:
- `~/.cache/huggingface/`

Size: ~4GB (persists across runs)

### References

- [Moondream2 on HuggingFace](https://huggingface.co/vikhyatk/moondream2)
- [Official Website](https://moondream.ai/)
- [Moondream Blog](https://moondream.ai/blog)
- [GitHub Repository](https://github.com/vikhyat/moondream)

### License

This example uses Moondream2 which is licensed under Apache 2.0.
