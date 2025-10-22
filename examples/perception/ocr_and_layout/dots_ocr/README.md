## dots.ocr - Multilingual Document OCR & Layout Parsing

This example demonstrates how to use dots.ocr, a powerful 1.7B vision-language model for multilingual document parsing. The model unifies layout detection and content recognition in a single architecture, supporting 100+ languages.

**✅ Working Solution**: This example uses `snapshot_download` to work around HuggingFace's limitation with periods in model names. The model loads and runs successfully!

### Features

- **Multilingual Support**: Process documents in 100+ languages
- **Unified Architecture**: Single model for layout detection and OCR
- **Comprehensive Parsing**: Extracts text, tables, formulas (LaTeX), and reading order
- **Compact Model**: Only 1.7B parameters with SOTA performance
- **Multiple Output Formats**: JSON, Markdown, plain text
- **Self-contained**: Fully dockerized with all dependencies

### Prerequisites

1. NVIDIA GPU with at least 8GB VRAM (RTX 3070, A10, etc.)
2. Docker with NVIDIA runtime support
3. HuggingFace account (optional, for faster model downloads)

### Quickstart

1. Set your HuggingFace token (optional but recommended):
```bash
export HF_TOKEN=your_huggingface_token
```

2. Build the Docker image:
```bash
bash examples/perception/ocr_and_layout/dots_ocr/build.sh
```

3. Run inference on a sample image:
```bash
bash examples/perception/ocr_and_layout/dots_ocr/predict.sh --image examples/sample.jpg
```

### Usage

#### Basic Inference

Process a document image:
```bash
bash examples/perception/ocr_and_layout/dots_ocr/predict.sh --image path/to/document.jpg
```

#### Custom Output Path

Save results to a specific location:
```bash
bash examples/perception/ocr_and_layout/dots_ocr/predict.sh \
    --image path/to/document.jpg \
    --output outputs/my_result.txt
```

#### JSON Output Format

Get structured JSON output:
```bash
bash examples/perception/ocr_and_layout/dots_ocr/predict.sh \
    --image path/to/document.jpg \
    --output outputs/result.json \
    --format json
```

#### Detailed Parsing

Use the detailed prompt for comprehensive analysis:
```bash
bash examples/perception/ocr_and_layout/dots_ocr/predict.sh \
    --image path/to/document.jpg \
    --detailed
```

#### Process URLs

Process images from URLs:
```bash
bash examples/perception/ocr_and_layout/dots_ocr/predict.sh \
    --image https://example.com/document.jpg
```

### Advanced Usage

#### Direct Docker Run

For more control, run the Docker container directly:

```bash
docker run --runtime nvidia \
    -v $(pwd)/examples/perception/ocr_and_layout/dots_ocr:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    dots_ocr \
    python3 predict.py \
        --image examples/sample.jpg \
        --output outputs/result.txt \
        --max-tokens 8192
```

#### Custom Prompts

Create your own prompt for specific extraction tasks:

```bash
docker run --runtime nvidia \
    -v $(pwd)/examples/perception/ocr_and_layout/dots_ocr:/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    dots_ocr \
    python3 predict.py \
        --image examples/invoice.jpg \
        --prompt "Extract all invoice details including date, amount, and line items in JSON format" \
        --format json
```

### Command-Line Options

The `predict.py` script supports the following options:

- `--image`: Path to input image or URL (default: `examples/sample.jpg`)
- `--model-path`: HuggingFace model ID or local path (default: `rednote-hilab/dots.ocr`)
- `--prompt`: Custom prompt for OCR task
- `--detailed`: Use detailed prompt for comprehensive parsing
- `--output`: Output file path (default: `outputs/result.txt`)
- `--format`: Output format - `txt` or `json` (default: `txt`)
- `--max-tokens`: Maximum tokens to generate (default: `4096`)
- `--device`: Device for inference - `auto`, `cuda`, or `cpu` (default: `auto`)

### Model Details

- **Name**: dots.ocr
- **Size**: 1.7B parameters
- **Architecture**: Vision-Language Model (VLM)
- **License**: MIT
- **Languages**: 100+ supported
- **Max Tokens**: Up to 24,000 output tokens

### Performance

- **Speed**: Fast inference on modern GPUs (< 30s for typical documents)
- **Memory**: ~8GB VRAM for full model
- **Accuracy**: SOTA performance on OmniDocBench and other benchmarks

### Output Examples

The model can extract:

1. **Plain Text**: Reading order preserved
2. **Tables**: Structured markdown tables
3. **Formulas**: LaTeX notation for mathematical expressions
4. **Layout**: Bounding boxes and categories (title, text, table, etc.)

### Supported Document Types

- Scanned documents
- PDFs (convert to images first)
- Screenshots
- Photographs of printed materials
- Mixed language documents
- Complex layouts with tables and formulas

### How It Works

This example uses `snapshot_download` to work around HuggingFace's limitation with periods in model names. The model is automatically downloaded to the cache and loaded from the snapshot directory, avoiding Python module import issues.

The first run will download the model (~3.5GB), subsequent runs use the cached model.

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Reduce `--max-tokens` (try 2048 or 1024)
2. Use a GPU with more VRAM
3. Process smaller images

#### Model Download Issues

If the model fails to download:
1. Set `HF_TOKEN` environment variable
2. Check internet connectivity
3. Verify HuggingFace access

#### Slow Inference

For faster inference:
1. Ensure flash-attention is installed (included in requirements)
2. Use GPU instead of CPU
3. Pre-download model weights to cache

### Testing

Run the smoke test to verify installation:
```bash
bash examples/perception/ocr_and_layout/dots_ocr/test.sh
```

### Directory Structure

```
examples/perception/ocr_and_layout/dots_ocr/
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── predict.py          # Main inference script
├── build.sh            # Build Docker image
├── predict.sh          # Run inference wrapper
├── test.sh             # Smoke test
├── README.md           # This file
├── .gitignore          # Git ignore rules
├── examples/           # Sample images (add your own)
└── outputs/            # Results saved here
```

### Output Location

By default, outputs are saved to:
- `examples/perception/ocr_and_layout/dots_ocr/outputs/`

This location is inside the example directory for easy access and review.

### References

- [dots.ocr GitHub](https://github.com/rednote-hilab/dots.ocr)
- [dots.ocr HuggingFace](https://huggingface.co/rednote-hilab/dots.ocr)
- [Official Website](https://www.dotsocr.net/)
- [Model Paper/Blog](https://www.marktechpost.com/2025/08/16/meet-dots-ocr-a-new-1-7b-vision-language-model-that-achieves-sota-performance-on-multilingual-document-parsing/)

### License

This example follows the MIT license of the dots.ocr model.
