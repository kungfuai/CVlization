## DeepSeek-OCR - Context Compression & Document OCR

This example demonstrates how to use DeepSeek-OCR, a revolutionary OCR system that uses optical context compression to achieve 7-20x token reduction while maintaining 97% OCR accuracy at 10x compression ratio.

**Original Model**: [deepseek-ai/DeepSeek-OCR on GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
**HuggingFace**: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
**Paper**: [arXiv:2510.18234](https://arxiv.org/abs/2510.18234)

### Features

- **Context Compression**: 7-20x token reduction compared to traditional OCR
- **High Accuracy**: 97% OCR precision at 10x compression ratio
- **High Throughput**: Process 200k+ pages/day on single A100-40G GPU
- **Multiple Tasks**: Markdown conversion, free OCR, image description, grounding
- **Production Ready**: Officially supported in vLLM for efficient inference
- **GPU Optimized**: Efficient inference with Flash Attention 2
- **Self-contained**: Fully dockerized with all dependencies

### Key Innovation

DeepSeek-OCR compresses document text into image format, reducing token usage dramatically:
- Traditional OCR: ~6000+ tokens per page (e.g., MinerU2.0)
- DeepSeek-OCR: <800 tokens per page at similar quality
- At 10x compression: Only 100 vision tokens while surpassing GOT-OCR2.0

### Prerequisites

1. NVIDIA GPU with at least 12GB VRAM (16GB+ recommended)
2. Docker with NVIDIA runtime support
3. CUDA 11.8 or higher

### Quickstart

1. Build the Docker image:
```bash
bash examples/perception/doc_ai/deepseek_ocr/build.sh
```

2. Run OCR on an image:
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh --image examples/sample.jpg
```

### Usage

#### Markdown Conversion (Default)

Convert document to markdown format with layout preservation:
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image path/to/document.jpg \
    --task markdown
```

#### Free OCR

Extract all text without specific formatting:
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image path/to/document.jpg \
    --task free_ocr
```

#### Image Description

Generate natural language description of the image:
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image path/to/document.jpg \
    --task describe
```

#### Grounding (Text Localization)

Extract text with bounding box locations:
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image path/to/document.jpg \
    --task grounding
```

### Advanced Usage

#### Output Formats

Save results in different formats:

**Markdown (default):**
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image document.jpg \
    --output results.md \
    --format md
```

**JSON with metadata:**
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image document.jpg \
    --output results.json \
    --format json
```

**Plain text:**
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image document.jpg \
    --output results.txt \
    --format txt
```

#### Inference Backend Selection

Choose between vLLM (faster) or Transformers (more compatible):

**vLLM (default, recommended):**
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image document.jpg \
    --backend vllm
```

**Transformers (fallback):**
```bash
bash examples/perception/doc_ai/deepseek_ocr/predict.sh \
    --image document.jpg \
    --backend transformers
```

#### Direct Docker Run

For more control over execution:
```bash
docker run --rm --gpus=all \
    -v $(pwd)/examples/perception/doc_ai/deepseek_ocr:/workspace \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --shm-size=8g \
    deepseek-ocr \
    python3 predict.py \
        --image examples/sample.jpg \
        --task markdown \
        --backend vllm
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image or URL (default: `examples/sample.jpg`)
- `--task` - Task type: `markdown`, `free_ocr`, `describe`, or `grounding` (default: `markdown`)
- `--output` - Output file path (default: `outputs/result.md`)
- `--format` - Output format: `txt`, `json`, or `md` (default: `md`)
- `--backend` - Inference backend: `vllm` or `transformers` (default: `vllm`)
- `--model` - HuggingFace model name (default: `deepseek-ai/DeepSeek-OCR`)

### Model Details

- **Name**: DeepSeek-OCR
- **Release Date**: October 20, 2025
- **Model Size**: 3B parameters total
  - DeepEncoder (vision encoder)
  - DeepSeek3B-MoE-A570M (language decoder)
- **Architecture**: Vision encoder + MoE decoder with context compression
- **License**: MIT
- **Repository**: [https://github.com/deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- **HuggingFace**: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

### Performance

**Token Compression:**
- 7-20x reduction compared to traditional OCR
- 97% accuracy at 10x compression
- 60% accuracy at 20x compression

**Throughput:**
- 200,000+ pages per day on single A100-40G GPU
- ~2500 tokens/second with concurrent inference on A100

**Benchmarks:**
- Surpasses GOT-OCR2.0 (256 tokens/page) using only 100 vision tokens
- Outperforms MinerU2.0 (6000+ tokens/page) with <800 vision tokens

**VRAM Requirements:**
- Minimum: 12GB (with optimizations)
- Recommended: 16GB+
- Optimal: 24GB+ (A10/A100)

### Supported Resolutions

**Native Resolutions:**
- 512×512
- 640×640
- 1024×1024
- 1280×1280

**Dynamic Resolution (Gundam mode):**
- n×640×640 + 1×1024×1024

### Use Cases

1. **LLM/VLM Training Data Generation**: Generate high-quality OCR at scale (200k+ pages/day)
2. **Document Processing Pipelines**: Efficient text extraction with minimal tokens
3. **Research Applications**: Compress academic papers, books, documentation
4. **Production OCR**: Real-time document processing with high throughput
5. **Multi-format Documents**: PDFs, scanned documents, forms, invoices
6. **Context-aware Extraction**: Preserve layout and structure in markdown

### How It Works

DeepSeek-OCR uses a unique approach called "Contexts Optical Compression":

1. **DeepEncoder** processes the document image
2. Text content is compressed into far fewer vision tokens
3. **DeepSeek3B-MoE-A570M** decoder generates output from compressed representation
4. Result: Same information in 7-20x fewer tokens

This enables:
- Faster inference (fewer tokens to process)
- Lower costs (reduced compute requirements)
- Higher throughput (process more documents)
- Better context utilization (more documents fit in context window)

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Reduce `max_model_len` in predict.py
2. Lower `gpu_memory_utilization` (default: 0.9)
3. Use smaller batch sizes
4. Close other GPU applications
5. Use a GPU with more VRAM

#### Model Download Issues

Models are automatically downloaded from HuggingFace on first run (~3GB):
1. Check internet connectivity
2. Verify HuggingFace access
3. Ensure sufficient disk space (~5GB for models + dependencies)
4. Check `~/.cache/huggingface/` for cached models

#### Slow Inference

For faster inference:
1. Use `--backend vllm` (default, much faster than transformers)
2. Ensure GPU is being used (check nvidia-smi)
3. Increase `gpu_memory_utilization` if you have VRAM headroom
4. Pre-download models to cache before production use

#### Backend Failures

If vLLM fails:
1. Script automatically tries transformers backend as fallback
2. Check CUDA compatibility (requires CUDA 11.8+)
3. Verify flash-attention is properly installed
4. Check Docker GPU runtime is configured

### Directory Structure

```
examples/perception/doc_ai/deepseek_ocr/
├── Dockerfile           # Container definition
├── predict.py          # Main inference script
├── build.sh            # Build Docker image
├── predict.sh          # Run inference wrapper
├── README.md           # This file
├── .gitignore          # Git ignore rules
├── examples/           # Sample images
│   └── sample.jpg     # Test image
└── outputs/            # Results saved here (git-ignored)
```

### Output Location

By default, outputs are saved to:
- `examples/perception/doc_ai/deepseek_ocr/outputs/`

### Model Cache

Model weights are cached in:
- `~/.cache/huggingface/hub/`

Size: ~3GB (persists across runs)

### Performance Comparison

| Method | Tokens/Page | Compression | Accuracy |
|--------|-------------|-------------|----------|
| Traditional OCR (MinerU2.0) | 6000+ | 1x | Baseline |
| GOT-OCR2.0 | 256 | ~23x | Good |
| DeepSeek-OCR (10x) | 100 | 60x | 97% |
| DeepSeek-OCR (20x) | 50 | 120x | 60% |

### Production Deployment

For production use:
1. Pre-download models: Run once to cache in `~/.cache/huggingface/`
2. Use vLLM backend for best performance
3. Monitor VRAM usage and adjust `gpu_memory_utilization`
4. Consider batching multiple documents
5. Use concurrent inference for higher throughput
6. Monitor output quality and adjust compression as needed

### Integration with CVlization

This example follows CVlization patterns:
- Dual-mode execution (standalone or via CVL)
- Standardized input/output paths
- Docker-based reproducibility
- HuggingFace model caching
- GPU-optimized inference

### References

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Paper: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Blog: DeepSeek-OCR Review](https://skywork.ai/blog/ai-agent/deepseek-ocr-review-2025-speed-accuracy-use-cases/)

### License

DeepSeek-OCR is licensed under the MIT License. See the [GitHub repository](https://github.com/deepseek-ai/DeepSeek-OCR) for full license details.

This example code is part of CVlization and follows the project's licensing terms.

### Community & Support

- GitHub Issues: [deepseek-ai/DeepSeek-OCR/issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
- HuggingFace Model Card: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- CVlization: [CVlization GitHub](https://github.com/kungfuai/CVlization)

### Citation

If you use DeepSeek-OCR in your research, please cite:

```bibtex
@article{deepseek2025ocr,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```
