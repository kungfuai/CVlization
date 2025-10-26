## Surya - Multilingual Document OCR Toolkit

This example demonstrates how to use Surya, a multilingual OCR toolkit supporting 90+ languages with advanced layout analysis, reading order detection, and table recognition capabilities.

**Original Model**: [datalab-to/surya on GitHub](https://github.com/datalab-to/surya)

### Features

- **Multilingual OCR**: 90+ languages supported
- **Layout Analysis**: Detect document structure and regions
- **Reading Order**: Determine natural reading flow
- **Table Recognition**: Extract tables from documents
- **Line Detection**: Accurate line-level text detection
- **LaTeX OCR**: Mathematical formula recognition
- **Fast Inference**: Optimized for GPU acceleration
- **Self-contained**: Fully dockerized with all dependencies

### Prerequisites

1. NVIDIA GPU with at least 8GB VRAM (can run on 6GB with tuning)
2. Docker with NVIDIA runtime support

### Quickstart

1. Build the Docker image:
```bash
bash examples/perception/ocr_and_layout/surya/build.sh
```

2. Run OCR on an image:
```bash
bash examples/perception/ocr_and_layout/surya/predict.sh --image examples/sample.jpg
```

### Usage

#### OCR - Text Extraction

Extract text from a document (automatically detects all 90+ languages):
```bash
bash examples/perception/ocr_and_layout/surya/predict.sh \
    --image path/to/document.jpg \
    --task ocr
```

#### Layout Analysis

Analyze document layout and structure:
```bash
bash examples/perception/ocr_and_layout/surya/predict.sh \
    --image path/to/document.jpg \
    --task layout
```

#### Reading Order Detection

Determine natural reading order:
```bash
bash examples/perception/ocr_and_layout/surya/predict.sh \
    --image path/to/document.jpg \
    --task order
```

### Advanced Usage

#### Batch Size Tuning for Different GPUs

Set environment variables to optimize VRAM usage:

**For 6-8GB VRAM:**
```bash
export RECOGNITION_BATCH_SIZE=8
export DETECTOR_BATCH_SIZE=1
export LAYOUT_BATCH_SIZE=1
bash examples/perception/ocr_and_layout/surya/predict.sh --image document.jpg
```

**For 12-16GB VRAM:**
```bash
export RECOGNITION_BATCH_SIZE=16
export DETECTOR_BATCH_SIZE=2
export LAYOUT_BATCH_SIZE=2
bash examples/perception/ocr_and_layout/surya/predict.sh --image document.jpg
```

**For 24GB+ VRAM (default):**
```bash
export RECOGNITION_BATCH_SIZE=32
export DETECTOR_BATCH_SIZE=4
export LAYOUT_BATCH_SIZE=4
bash examples/perception/ocr_and_layout/surya/predict.sh --image document.jpg
```

**VRAM per batch item:**
- Recognition: ~0.3-1.6 GB (depending on batch size 8-32)
- Detection: ~0.4-1.8 GB (depending on batch size 1-4)
- Layout: ~0.2-0.9 GB (depending on batch size 1-4)

#### JSON Output

Save results as JSON with metadata:
```bash
bash examples/perception/ocr_and_layout/surya/predict.sh \
    --image document.jpg \
    --output results.json \
    --format json
```

#### Direct Docker Run

For more control:
```bash
docker run --runtime nvidia \
    -v $(pwd)/examples/perception/ocr_and_layout/surya:/workspace \
    -v $(pwd)/${CVL_HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
    -e RECOGNITION_BATCH_SIZE=32 \
    -e DETECTOR_BATCH_SIZE=4 \
    -e LAYOUT_BATCH_SIZE=4 \
    surya \
    python3 predict.py \
        --image examples/sample.jpg \
        --task ocr
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image or URL (default: `examples/sample.jpg`)
- `--task` - Task type: `ocr`, `layout`, or `order` (default: `ocr`)
- `--output` - Output file path (default: `outputs/result.txt`)
- `--format` - Output format: `txt` or `json` (default: `txt`)

Note: Surya automatically detects and supports 90+ languages without requiring explicit language specification.

### Model Details

- **Name**: Surya OCR
- **Version**: 0.17.0
- **Architecture**:
  - Detection: Modified EfficientViT
  - Recognition: Modified Donut (GQA, MoE layer, UTF-16 decoding)
- **Languages**: 90+ supported
- **License**: Modified AI Pubs Open Rail-M (free for research/personal/startups <$2M)
- **Repository**: [https://github.com/datalab-to/surya](https://github.com/datalab-to/surya)

### Performance

**Training:**
- Detection: 4x A6000 GPUs, 3 days
- Recognition: 4x A6000 GPUs, 2 weeks

**Inference (Nvidia T4):**
- ~9 seconds per document

**Benchmarks:**
- Outperforms Tesseract on speed and accuracy
- Strong performance across 90+ languages

### Capabilities

1. **OCR**: Line-level text extraction in 90+ languages
2. **Layout Analysis**: Document structure detection
3. **Reading Order**: Natural text flow detection
4. **Table Recognition**: Extract tabular data
5. **LaTeX OCR**: Mathematical formula recognition
6. **Multi-language**: Process documents with mixed languages

### Supported Document Types

- Scanned documents
- PDFs (convert to images first)
- Screenshots
- Photographs
- Invoices and receipts
- Forms and tables
- Mixed content documents
- Multi-language documents

### How It Works

Surya automatically downloads model weights from HuggingFace on first run. Models are cached in `$HOME/.cache/huggingface` and reused for subsequent runs. The toolkit uses separate models for detection, recognition, layout analysis, and reading order.

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Reduce batch sizes using environment variables
2. Start with smaller values and tune upward
3. Close other GPU applications
4. Use a GPU with more VRAM

#### Model Download Issues

If models fail to download:
1. Check internet connectivity
2. Verify HuggingFace access
3. Ensure sufficient disk space (~2-3GB for models)

#### Slow Inference

For faster inference:
1. Increase batch sizes if you have more VRAM
2. Ensure GPU is being used
3. Pre-download models to cache

### Testing

Run the smoke test to verify installation:
```bash
bash examples/perception/ocr_and_layout/surya/test.sh
```

### Directory Structure

```
examples/perception/ocr_and_layout/surya/
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
- `examples/perception/ocr_and_layout/surya/outputs/`

### Model Cache

Model weights are cached in:
- `~/.cache/huggingface/`

Size: ~2-3GB (persists across runs)

### Supported Languages

90+ languages including:
- **Latin script**: English, Spanish, French, German, Italian, Portuguese, etc.
- **Cyrillic**: Russian, Ukrainian, Bulgarian, Serbian, etc.
- **CJK**: Chinese, Japanese, Korean
- **Arabic script**: Arabic, Persian, Urdu
- **Indic scripts**: Hindi, Bengali, Tamil, Telugu, etc.
- **And many more...**

See [GitHub repository](https://github.com/datalab-to/surya) for complete language list.

### References

- [Surya on GitHub](https://github.com/datalab-to/surya)
- [Surya on PyPI](https://pypi.org/project/surya-ocr/)
- [Model on HuggingFace](https://huggingface.co/datalab-to)

### License

This example uses Surya OCR which is licensed under a modified AI Pubs Open Rail-M license. It's free for research, personal use, and startups with less than $2M in funding/revenue. See the [GitHub repository](https://github.com/datalab-to/surya) for full license details.
