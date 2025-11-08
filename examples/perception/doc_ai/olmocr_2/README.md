# olmOCR-2 - Production-Ready OCR from AllenAI

Production-ready OCR system from AllenAI using olmOCR-2-7B-1025-FP8, a 7B parameter vision-language model fine-tuned from Qwen2.5-VL-7B-Instruct with GRPO (Generalized Reward-based Policy Optimization) reinforcement learning training.

## Model Overview

**olmOCR-2-7B-1025** (October 2025) represents a significant advancement in document OCR, particularly excelling at mathematical equations, tables, and complex OCR scenarios. Designed for large-scale production pipelines processing millions of documents.

### Key Features

- **Model Size**: 7B parameters (FP8 quantization available)
- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Specialized Training**: GRPO RL training for math equations, tables, tricky OCR cases
- **Benchmark Score**: 82.4±1.1 on olmOCR-Bench (1,403 PDFs + 7,010 unit tests)
- **Throughput**: Designed for million-document scale processing
- **Cost**: Less than $200 USD per million pages converted
- **Natural Reading Order**: Automatically preserves document flow
- **Header/Footer Removal**: Intelligent content filtering

### Strengths

- Math equations (superior GRPO-trained performance)
- Tables and structured data
- Old scans and degraded documents
- Large-scale production pipelines
- Complex multi-page documents

### Supported Formats

**Input**: PDF, PNG, JPEG
**Output**: Clean Markdown with preserved structure

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with 15GB+ VRAM minimum
  - Tested on: RTX 4090, L40S, A100, H100
- **Disk**: 30GB free space (model weights + dependencies)
- **RAM**: 16GB+ recommended

### Software

- Docker with NVIDIA GPU support
- CUDA 12.4+
- Python 3.11 (in container)

## Quick Start

### 1. Build Docker Image

```bash
./build.sh
```

This creates a Docker image with:
- PyTorch 2.5.1 with CUDA 12.4 support
- olmOCR package with GPU acceleration
- Poppler-utils and fonts for document processing

### 2. Run OCR on an Image

```bash
# Using default test image
./predict.sh

# Using custom image
./predict.sh --image path/to/document.jpg --output outputs/result.md

# Process PDF
./predict.sh --image path/to/document.pdf --output outputs/result.md

# Output as JSON
./predict.sh --image path/to/document.jpg --output outputs/result.json --format json
```

### 3. Check Results

```bash
cat outputs/result.md
```

## Usage with CVlization

```bash
# List available examples
cvl list

# Get information about olmocr_2
cvl info olmocr_2

# Build the Docker image
cvl run olmocr_2 build

# Run OCR
cvl run olmocr_2 predict --image /path/to/image.jpg
```

## Command Line Options

```bash
./predict.sh [OPTIONS]

Options:
  --image PATH      Path to input image or PDF (default: shared test image)
  --output PATH     Output file path (default: outputs/result.md)
  --format FORMAT   Output format: md, json, txt (default: md)
```

## Python API

```python
#!/usr/bin/env python3
import subprocess
import sys

def run_olmocr(input_file, output_dir):
    """Run olmOCR-2 pipeline"""
    cmd = [
        sys.executable, "-m", "olmocr.pipeline",
        output_dir,
        "--markdown",
        "--pdfs" if input_file.endswith('.pdf') else "--pngs",
        input_file
    ]
    subprocess.run(cmd, check=True)

    # Read output from output_dir/markdown/
    with open(f"{output_dir}/markdown/{Path(input_file).stem}.md") as f:
        return f.read()

# Example usage
result = run_olmocr("document.pdf", "./workspace")
print(result)
```

## Performance & Benchmarks

### olmOCR-Bench Scores

| Model | Score | Notes |
|-------|-------|-------|
| **olmOCR-2-7B-1025** | **82.4±1.1** | Latest (Oct 2025) |
| Chandra OCR | 83.1±0.9 | Highest overall |
| olmOCR (earlier) | 78.5 | Previous version |
| DeepSeek-OCR | 75.4±1.0 | Context compression |

### Specialized Performance

- **Math Equations**: Superior (GRPO RL trained)
- **Tables**: Excellent structure preservation
- **Old Scans**: Strong performance on degraded documents
- **Multi-page**: Efficient large document processing

### Production Metrics

- **Cost**: <$200 USD per 1M pages
- **Throughput**: Designed for multi-node distributed processing
- **Scale**: Million-document pipelines via AWS S3 integration

## Technical Details

### Model Architecture

- **Base**: Qwen2.5-VL-7B-Instruct (vision-language foundation)
- **Training**: GRPO (Generalized Reward-based Policy Optimization)
  - Specialized reward functions for math, tables, OCR quality
  - Reinforcement learning for complex scenarios
- **Quantization**: FP8 version available for reduced VRAM

### Training Data

- **olmOCR-mix-1025**: Curated training dataset (October 2025)
- **olmOCR-bench**: Comprehensive evaluation suite
  - 1,403 diverse PDFs
  - 7,010 unit test cases
  - Covers math, tables, layouts, old scans

### Pipeline Components

1. **Document Loading**: PDF/Image ingestion with poppler
2. **Visual Encoding**: Qwen2.5-VL vision transformer
3. **OCR Generation**: Autoregressive text generation
4. **Post-processing**: Markdown formatting, header/footer removal

## Comparison with Other Models

### vs. Chandra OCR (83.1 score)
- **Chandra**: Higher overall score, better handwriting/forms
- **olmOCR-2**: Better math equations, production-scale pipelines

### vs. DeepSeek-OCR (75.4 score)
- **DeepSeek**: Context compression (7-20x tokens), data generation
- **olmOCR-2**: Higher accuracy, production pipeline focus

### vs. PaddleOCR-VL (0.9B params)
- **PaddleOCR-VL**: Ultra-efficient (0.9B), 109 languages, edge devices
- **olmOCR-2**: Larger model (7B), better quality, math/tables specialty

## Use Cases

### Ideal For

1. **Large-scale document digitization** (millions of documents)
2. **Mathematical/scientific papers** (equations, formulas)
3. **Financial documents** (tables, structured data)
4. **Academic archives** (old scans, degraded quality)
5. **Production pipelines** (AWS S3 integration, distributed processing)

### Not Ideal For

1. **Resource-constrained environments** (use PaddleOCR-VL 0.9B)
2. **Handwriting-heavy documents** (use Chandra OCR)
3. **Edge devices** (consider smaller models)
4. **Real-time processing** (latency vs accuracy trade-off)

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size (edit predict.py)
# Use FP8 quantization
# Ensure 15GB+ VRAM available
```

### Slow First Run

- First run downloads ~15GB of model weights
- Subsequent runs use cached models from ~/.cache/huggingface
- Download time depends on network speed

### Missing Fonts/Poppler

```bash
# Rebuild Docker image (includes all dependencies)
./build.sh
```

### Model Download Issues

```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--allenai--olmOCR-2*
./predict.sh
```

## External Inference Providers

For GPU-less deployments, verified commercial providers:

| Provider | Input Cost | Output Cost | Availability |
|----------|-----------|-----------|--------------|
| Cirrascale | $0.07/M tokens | $0.15/M tokens | Enterprise |
| DeepInfra | $0.09/M tokens | $0.19/M tokens | Public API |
| Parasail | $0.10/M tokens | $0.20/M tokens | Cloud |

## Development

### Project Structure

```
olmocr_2/
├── Dockerfile          # Container with olmOCR + dependencies
├── predict.py          # Python inference script
├── predict.sh          # Docker wrapper script
├── build.sh            # Docker build script
├── example.yaml        # CVlization metadata
├── README.md           # This file
└── outputs/            # Generated OCR results
```

### Modifying the Pipeline

Edit `predict.py` to customize:
- Input/output formats
- Post-processing steps
- Error handling
- Batch processing

## Resources

- **GitHub**: https://github.com/allenai/olmocr
- **Model Card**: https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8
- **Base Model**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **Benchmark Dataset**: https://huggingface.co/datasets/allenai/olmOCR-bench
- **HuggingFace Collection**: https://huggingface.co/collections/allenai/olmocr

## Citation

```bibtex
@article{olmocr2025,
  title={olmOCR-2: Production-Ready OCR with Reinforcement Learning},
  author={AllenAI Team},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/allenai/olmocr}
}
```

## License

- **Code**: Apache 2.0 (olmOCR toolkit)
- **Model**: Check model card for specific licensing terms
- **AllenAI**: Non-commercial research use encouraged

## Support

- **Issues**: https://github.com/allenai/olmocr/issues
- **CVlization Issues**: https://github.com/kungfuai/CVlization/issues
- **Model Questions**: HuggingFace model card discussions

## Changelog

### 2025-11-07

- Initial olmOCR-2-7B-1025-FP8 implementation
- Docker containerization with CUDA 12.4 support
- CVlization integration (build/predict presets)
- Python script with dual-mode execution support
- Comprehensive documentation
