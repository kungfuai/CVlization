## Chandra OCR - High-Accuracy Document OCR with Layout Preservation

This example demonstrates how to use Chandra OCR, the highest-scoring OCR model on OmniDocBench (83.1), surpassing DeepSeek-OCR (75.4), dots.ocr (79.1), olmOCR (78.5), and even GPT-4o and Gemini Flash 2 on complex documents.

**Original Model**: [datalab-to/chandra on GitHub](https://github.com/datalab-to/chandra)
**HuggingFace**: [datalab-to/chandra](https://huggingface.co/datalab-to/chandra)
**Website**: [Datalab.to](https://www.datalab.to/blog/introducing-chandra)

### Features

- **Highest Accuracy**: 83.1 ± 0.9 score on OmniDocBench, beating all competitors
- **Full Layout Preservation**: Maintains document structure in HTML/Markdown/JSON
- **Handwriting Support**: Excellent recognition of handwritten text
- **Form Recognition**: Accurate extraction of forms including checkboxes
- **Complex Elements**: Superior handling of tables, math equations, diagrams
- **Multilingual**: 40+ language support
- **Image Extraction**: Captures images and diagrams with captions
- **Production Ready**: Simple pip installation, HuggingFace integration
- **Self-contained**: Fully dockerized with all dependencies

### Key Innovation

Chandra uses advanced vision-language modeling to achieve superior accuracy on:
- Multi-column documents (outperforms GPT-4o)
- Old scanned documents (outperforms Gemini Flash 2)
- Complex tables with merged cells
- Handwritten forms and notes
- Mathematical equations
- Mixed content (text + images + diagrams)

### Prerequisites

1. NVIDIA GPU with at least 16GB VRAM (24GB+ recommended)
2. Docker with NVIDIA runtime support
3. CUDA 11.8 or higher

### Quickstart

1. Build the Docker image:
```bash
bash examples/perception/doc_ai/chandra_ocr/build.sh
```

2. Run OCR on an image:
```bash
bash examples/perception/doc_ai/chandra_ocr/predict.sh  # Uses shared test image by default
```

### Usage

#### Markdown Conversion with Layout (Default)

Convert document to markdown with full layout preservation:
```bash
bash examples/perception/doc_ai/chandra_ocr/predict.sh \
    --image path/to/document.jpg \
    --prompt-type ocr_layout
```

#### Structured OCR

Extract text with structure but simplified layout:
```bash
bash examples/perception/doc_ai/chandra_ocr/predict.sh \
    --image path/to/document.jpg \
    --prompt-type ocr
```

#### Plain Text Extraction

Simple text extraction without structure:
```bash
bash examples/perception/doc_ai/chandra_ocr/predict.sh \
    --image path/to/document.jpg \
    --prompt-type plain_ocr
```

### Advanced Usage

#### Output Formats

Save results in different formats:

**Markdown (default):**
```bash
bash examples/perception/doc_ai/chandra_ocr/predict.sh \
    --image document.jpg \
    --output results.md \
    --format md
```

**HTML with layout:**
```bash
bash examples/perception/doc_ai/chandra_ocr/predict.sh \
    --image document.jpg \
    --output results.html \
    --format html
```

**JSON with metadata:**
```bash
bash examples/perception/doc_ai/chandra_ocr/predict.sh \
    --image document.jpg \
    --output results.json \
    --format json
```

**Plain text:**
```bash
bash examples/perception/doc_ai/chandra_ocr/predict.sh \
    --image document.jpg \
    --output results.txt \
    --format txt
```

#### Direct Docker Run

For more control over execution:
```bash
docker run --rm --gpus=all \
    -v $(pwd)/examples/perception/doc_ai/chandra_ocr:/workspace \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --shm-size=8g \
    chandra-ocr \
    python3 predict.py \
        --prompt-type ocr_layout
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image (default: shared test image from `/cvlization_repo/examples/doc_ai/leaderboard/test_data/sample.jpg`)
- `--prompt-type` - Prompt type: `ocr_layout` (default, with layout), `ocr` (structured), or `plain_ocr` (simple text)
- `--output` - Output file path (default: `outputs/result.{format}`)
- `--format` - Output format: `txt`, `json`, `md`, or `html` (default: `md`)
- `--model` - HuggingFace model name (default: `datalab-to/chandra`)

### Model Details

- **Name**: Chandra OCR
- **Release Date**: November 2025
- **Model Size**: ~7GB download
- **Architecture**: Vision-language model with layout understanding
- **License**:
  - Code: Apache 2.0
  - Model weights: Modified OpenRAIL-M (free for research, personal use, startups <$2M revenue)
- **Repository**: [https://github.com/datalab-to/chandra](https://github.com/datalab-to/chandra)
- **HuggingFace**: [datalab-to/chandra](https://huggingface.co/datalab-to/chandra)

### Performance

**OmniDocBench Scores:**
- **Chandra**: 83.1 ± 0.9 (BEST)
- olmOCR-2: 78.5
- dots.ocr: 79.1
- DeepSeek-OCR: 75.4 ± 1.0
- GPT-4o: Lower on multi-column tests
- Gemini Flash 2: Lower on old-scan tests

**Strengths:**
- Multi-column documents
- Old scanned documents
- Complex tables with merged cells
- Handwritten text
- Forms with checkboxes
- Mathematical equations
- Mixed content types

**VRAM Requirements:**
- Minimum: 16GB
- Recommended: 24GB+
- Optimal: 32GB+ (A100)

### Supported Languages

40+ languages including:
- English, Spanish, French, German, Italian
- Chinese (Simplified & Traditional), Japanese, Korean
- Arabic, Hindi, Russian
- Portuguese, Dutch, Polish, Turkish
- And many more...

### Use Cases

1. **Document Digitization**: Convert scanned documents to editable formats
2. **Form Processing**: Extract data from forms including checkboxes
3. **Historical Documents**: OCR old scanned texts and manuscripts
4. **Academic Papers**: Extract text, tables, and equations from PDFs
5. **Handwritten Notes**: Digitize handwritten documents
6. **Multi-lingual Documents**: Process documents in 40+ languages
7. **Complex Layouts**: Handle multi-column, mixed content documents
8. **Quality Archiving**: Preserve original document layout in digital form

### How It Works

Chandra uses a vision-language architecture that:

1. **Visual Understanding** - Processes the document image with a vision encoder
2. **Layout Analysis** - Understands document structure (columns, tables, forms)
3. **Element Recognition** - Identifies text, tables, images, math, checkboxes
4. **Content Extraction** - Generates structured output preserving layout
5. **Format Conversion** - Outputs HTML/Markdown/JSON with full fidelity

This enables:
- Accurate text recognition across diverse content types
- Layout preservation in output formats
- Handling of complex document elements
- Superior performance on challenging documents

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Check VRAM usage with nvidia-smi
2. Close other GPU applications
3. Use a GPU with more VRAM (24GB+ recommended)
4. Try processing smaller images or lower resolution

#### Model Download Issues

Models are automatically downloaded from HuggingFace on first run (~7GB):
1. Check internet connectivity
2. Verify HuggingFace access
3. Ensure sufficient disk space (~10GB for models + dependencies)
4. Check `~/.cache/huggingface/` for cached models

#### Slow Inference

For faster inference:
1. Ensure GPU is being used (check nvidia-smi)
2. Pre-download models to cache before production use
3. Ensure sufficient VRAM is available
4. Use SSD for model cache directory

#### Poor Quality Results

If OCR quality is unsatisfactory:
1. Check image quality (resolution, clarity)
2. Try different prompt types (ocr_layout vs ocr vs plain_ocr)
3. Ensure image is properly oriented
4. For forms/checkboxes, use `ocr_layout` prompt type
5. For simple text, try `plain_ocr`

### Directory Structure

```
examples/perception/doc_ai/chandra_ocr/
├── Dockerfile           # Container definition
├── predict.py          # Main inference script
├── build.sh            # Build Docker image
├── predict.sh          # Run inference wrapper
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── outputs/            # Results saved here (git-ignored)

# Note: Default test image is shared from:
# ../leaderboard/test_data/sample.jpg
# (mounted as /cvlization_repo/... in Docker)
```

### Output Location

By default, outputs are saved to:
- `examples/perception/doc_ai/chandra_ocr/outputs/`

### Model Cache

Model weights are cached in:
- `~/.cache/huggingface/hub/`

Size: ~7GB (persists across runs)

### Performance Comparison

| Model | OmniDocBench Score | Tables | Handwriting | Forms | Multi-column |
|-------|-------------------|--------|-------------|-------|--------------|
| **Chandra** | **83.1 ± 0.9** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| olmOCR-2 | 78.5 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| dots.ocr | 79.1 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| DeepSeek-OCR | 75.4 ± 1.0 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| GPT-4o | Lower | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Gemini Flash 2 | Lower | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Production Deployment

For production use:
1. Pre-download models: Run once to cache in `~/.cache/huggingface/`
2. Monitor VRAM usage with nvidia-smi
3. Use appropriate prompt type for your use case
4. Consider batching multiple documents
5. Monitor output quality and adjust settings as needed
6. Be aware of license restrictions (Modified OpenRAIL-M)

### Integration with CVlization

This example follows CVlization patterns:
- Dual-mode execution (standalone or via CVL)
- Optional cvlization.paths import with fallback for standalone use
- Standardized input/output paths
- Docker-based reproducibility
- HuggingFace model caching
- GPU-optimized inference

### References

- [Chandra OCR GitHub](https://github.com/datalab-to/chandra)
- [Chandra OCR on HuggingFace](https://huggingface.co/datalab-to/chandra)
- [Datalab Blog: Introducing Chandra](https://www.datalab.to/blog/introducing-chandra)
- [OmniDocBench Benchmark](https://github.com/opendatalab/OmniDocBench)
- [chandra-ocr on PyPI](https://pypi.org/project/chandra-ocr/)

### License

Chandra OCR uses a dual license:
- **Code**: Apache 2.0 License
- **Model Weights**: Modified OpenRAIL-M License
  - Free for research and personal use
  - Free for startups with <$2M annual revenue
  - Requires commercial license for larger enterprises

See the [GitHub repository](https://github.com/datalab-to/chandra) for full license details.

This example code is part of CVlization and follows the project's licensing terms.

### Community & Support

- GitHub Issues: [datalab-to/chandra/issues](https://github.com/datalab-to/chandra/issues)
- HuggingFace Model Card: [datalab-to/chandra](https://huggingface.co/datalab-to/chandra)
- CVlization: [CVlization GitHub](https://github.com/kungfuai/CVlization)
- Datalab Website: [https://www.datalab.to](https://www.datalab.to)

### Citation

If you use Chandra OCR in your research, please cite:

```bibtex
@misc{chandra2025,
  title={Chandra OCR: High-Accuracy Document Understanding with Layout Preservation},
  author={Datalab.to},
  year={2025},
  url={https://github.com/datalab-to/chandra}
}
```

### Acknowledgments

- Developed by the team at [Datalab.to](https://www.datalab.to)
- Built on top of HuggingFace Transformers
- Tested on OmniDocBench and other standard benchmarks
