## PaddleOCR-VL - Ultra-Efficient Multilingual Document OCR

This example demonstrates how to use PaddleOCR-VL, an ultra-efficient 0.9B parameter OCR model with support for 109 languages. Despite its compact size, it matches the performance of 70-200B parameter models and topped HuggingFace trending charts for 5 consecutive days.

**Original Model**: [PaddlePaddle/PaddleOCR-VL on HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
**Paper**: [PaddleOCR-VL: Vision-Language Model for Document OCR](https://arxiv.org/html/2510.14528v1)
**Demo**: [HuggingFace Space](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo)

### Features

- **Ultra-Efficient**: Only 0.9B parameters - perfect for edge devices
- **Multilingual**: Supports 109 languages out of the box
- **Trending**: #1 on HuggingFace OCR rankings, trending for 5 consecutive days
- **Complex Elements**: Recognizes text, tables, formulas, and charts
- **High Performance**: Matches 70-200B parameter models despite tiny size
- **Full-Page Parsing**: Processes entire documents with layout preservation
- **Markdown Output**: Outputs markdown with visualizations
- **Resource-Efficient**: Minimal VRAM requirements (8GB+)
- **Production Ready**: Based on proven PaddlePaddle framework
- **Self-contained**: Fully dockerized with all dependencies

### Key Innovation

PaddleOCR-VL uses a compact NaViT-style dynamic resolution visual encoder combined with the ERNIE-4.5-0.3B language model to achieve performance comparable to models 70-200x larger, making it ideal for:
- Edge device deployment
- Resource-constrained environments
- High-throughput scenarios requiring efficiency
- Multilingual document processing (109 languages)

### Prerequisites

1. NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
2. Docker with NVIDIA runtime support
3. CUDA 12.6 or higher

### Quickstart

1. Build the Docker image:
```bash
bash examples/perception/doc_ai/paddleocr_vl/build.sh
```

2. Run OCR on an image:
```bash
bash examples/perception/doc_ai/paddleocr_vl/predict.sh  # Uses shared test image by default
```

### Usage

#### Basic Document OCR

Convert document to markdown:
```bash
bash examples/perception/doc_ai/paddleocr_vl/predict.sh \
    --image path/to/document.jpg \
    --output result.md
```

#### Multilingual Documents

Process documents in any of 109 supported languages:
```bash
bash examples/perception/doc_ai/paddleocr_vl/predict.sh \
    --image chinese_document.jpg \
    --output chinese_result.md
```

### Advanced Usage

#### Output Formats

Save results in different formats:

**Markdown (default):**
```bash
bash examples/perception/doc_ai/paddleocr_vl/predict.sh \
    --image document.jpg \
    --output results.md \
    --format md
```

**JSON with metadata:**
```bash
bash examples/perception/doc_ai/paddleocr_vl/predict.sh \
    --image document.jpg \
    --output results.json \
    --format json
```

**Plain text:**
```bash
bash examples/perception/doc_ai/paddleocr_vl/predict.sh \
    --image document.jpg \
    --output results.txt \
    --format txt
```

#### Direct Docker Run

For more control over execution:
```bash
docker run --rm --gpus=all \
    -v $(pwd)/examples/perception/doc_ai/paddleocr_vl:/workspace \
    -v ${HOME}/.paddleocr:/root/.paddleocr \
    --shm-size=8g \
    paddleocr-vl \
    python3 predict.py --image your_image.jpg
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image (default: shared test image from `/cvlization_repo/examples/doc_ai/leaderboard/test_data/sample.jpg`)
- `--output` - Output file path (default: `outputs/result.md`)
- `--format` - Output format: `md`, `json`, or `txt` (default: `md`)

### Model Details

- **Name**: PaddleOCR-VL
- **Release Date**: October 2025
- **Model Size**: 0.9B parameters (ultra-compact)
- **Architecture**: NaViT-style dynamic resolution encoder + ERNIE-4.5-0.3B LM
- **Languages**: 109 languages supported
- **License**: Apache 2.0
- **Repository**: [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

### Performance

**Model Comparison:**
- **Size**: Only 0.9B parameters
- **Performance**: Matches 70-200B parameter models
- **Popularity**: #1 on HuggingFace OCR rankings
- **Trending**: 5 consecutive days on HuggingFace trending

**Strengths:**
- Ultra-efficient for resource-constrained environments
- Exceptional multilingual support (109 languages)
- Complex element recognition (text, tables, formulas, charts)
- Fast inference time due to compact size
- Suitable for edge deployment

**VRAM Requirements:**
- Minimum: 8GB
- Recommended: 16GB
- Optimal: 24GB (for larger batches)

### Supported Languages

109 languages including:
- **European**: English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Turkish, Russian, etc.
- **Asian**: Chinese (Simplified & Traditional), Japanese, Korean, Hindi, Arabic, Thai, Vietnamese, etc.
- **Other**: Hebrew, Persian, Urdu, Bengali, Tamil, Telugu, Kannada, Malayalam, and many more

### Use Cases

1. **Edge Deployment**: Ultra-efficient model for resource-constrained devices
2. **Multilingual Processing**: Handle documents in 109 different languages
3. **High-Throughput**: Process many documents quickly due to compact size
4. **Mobile Applications**: Deploy on mobile/embedded devices
5. **Academic Papers**: Extract text, tables, and formulas
6. **Business Documents**: Process invoices, forms, receipts across languages
7. **Historical Documents**: OCR old scanned texts in various languages
8. **Cost-Sensitive Deployments**: Reduce compute costs with efficient model

### How It Works

PaddleOCR-VL uses an innovative architecture:

1. **Visual Encoding** - NaViT-style dynamic resolution vision encoder
2. **Language Processing** - ERNIE-4.5-0.3B language model (0.3B params)
3. **Element Recognition** - Identifies text, tables, formulas, charts
4. **Multilingual Support** - Pre-trained on 109 languages
5. **Markdown Generation** - Outputs structured markdown with visualizations

This enables:
- Exceptional efficiency (0.9B total parameters)
- Performance matching much larger models
- Broad language coverage
- Fast inference speed
- Low resource consumption

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Check VRAM usage with nvidia-smi
2. Close other GPU applications
3. Reduce batch size (if processing multiple images)
4. Use a GPU with more VRAM (16GB+ recommended)

#### Model Download Issues

Models are automatically downloaded from HuggingFace on first run:
1. Check internet connectivity
2. Verify HuggingFace access
3. Ensure sufficient disk space (~2GB for model)
4. Check `~/.paddleocr/` for cached models

#### Slow Inference

For faster inference:
1. Ensure GPU is being used (check nvidia-smi)
2. Pre-download models to cache before production use
3. Use SSD for model cache directory
4. Consider using vLLM server for production (see PaddleOCR-VL docs)

#### Poor Quality Results

If OCR quality is unsatisfactory:
1. Check image quality (resolution, clarity)
2. Ensure image is properly oriented
3. Verify language is in the 109 supported languages
4. Try higher resolution images

#### PaddlePaddle Installation Issues

If you encounter PaddlePaddle installation errors:
1. Ensure CUDA version matches (12.6)
2. Check NVIDIA driver compatibility
3. Verify system meets minimum requirements
4. Consider using provided Docker image (recommended)

### Directory Structure

```
examples/perception/doc_ai/paddleocr_vl/
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
- `examples/perception/doc_ai/paddleocr_vl/outputs/`

### Model Cache

Model weights are cached in:
- `~/.paddleocr/`

Size: ~2GB (persists across runs)

### Performance Comparison

| Model | Parameters | Languages | VRAM | Efficiency | Edge Deploy |
|-------|-----------|-----------|------|------------|-------------|
| **PaddleOCR-VL** | **0.9B** | **109** | **8GB** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Chandra | ? | 40+ | 16GB | ⭐⭐⭐ | ⭐⭐⭐ |
| DeepSeek-OCR | 3B | - | 24GB | ⭐⭐⭐⭐ | ⭐⭐ |
| olmOCR-2 | 7B | - | 32GB | ⭐⭐ | ⭐ |

### Production Deployment

For production use:
1. Pre-download models: Run once to cache in `~/.paddleocr/`
2. Monitor VRAM usage with nvidia-smi
3. Consider vLLM server for high-throughput scenarios
4. Batch multiple documents for efficiency
5. Use appropriate output format for your pipeline
6. Test with your target languages

### Integration with CVlization

This example follows CVlization patterns:
- Dual-mode execution (standalone or via CVL)
- Optional cvlization.paths import with fallback for standalone use
- Standardized input/output paths
- Docker-based reproducibility
- HuggingFace model caching
- GPU-optimized inference

### Advantages Over Other Models

**vs. Large Models (7B+ params):**
- 7-70x smaller model size
- Lower VRAM requirements
- Faster inference
- Suitable for edge devices
- Lower deployment costs

**vs. Chandra/DeepSeek-OCR:**
- More languages supported (109 vs 40)
- More resource-efficient
- Better for edge deployment
- Faster inference due to compact size

**Unique Strengths:**
- Unmatched efficiency (0.9B params)
- Most languages supported (109)
- Best for resource-constrained scenarios
- Trending #1 on HuggingFace

### References

- [PaddleOCR-VL on HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [PaddleOCR-VL Paper](https://arxiv.org/html/2510.14528v1)
- [Demo Space](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo)
- [PaddlePaddle Framework](https://github.com/PaddlePaddle/Paddle)
- [PaddleOCR Project](https://github.com/PaddlePaddle/PaddleOCR)

### License

PaddleOCR-VL uses the Apache 2.0 License, making it free for commercial use.

See the [HuggingFace model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) for full license details.

This example code is part of CVlization and follows the project's licensing terms.

### Community & Support

- HuggingFace Model: [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- Demo Space: [PaddleOCR-VL Online Demo](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo)
- PaddleOCR GitHub: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- CVlization: [CVlization GitHub](https://github.com/kungfuai/CVlization)

### Citation

If you use PaddleOCR-VL in your research, please cite:

```bibtex
@article{paddleocr-vl2025,
  title={PaddleOCR-VL: Vision-Language Model for Document OCR},
  author={PaddlePaddle Team},
  journal={arXiv preprint arXiv:2510.14528},
  year={2025},
  url={https://huggingface.co/PaddlePaddle/PaddleOCR-VL}
}
```

### Acknowledgments

- Developed by the PaddlePaddle Team at Baidu
- Built on the PaddlePaddle deep learning framework
- Integrated with HuggingFace ecosystem
