## MolmoE-1B - Ultra-Efficient Vision Language Model

This example demonstrates multimodal understanding using **MolmoE-1B** from Allen Institute (AI2), a Mixture-of-Experts model with only 1.2B active parameters that **nearly matches GPT-4V performance**.

### Key Features

- **Exceptional Performance**: Nearly matches GPT-4V despite 1.2B active parameters
- **Ultra-Efficient**: 6.9B total with 64 experts, only 1.2B (8 experts) active per token
- **Compact Size**: ~3GB VRAM for inference
- **SOTA Small Model**: Best performance among <2B models
- **Fully Open**: Apache 2.0 license with open weights, data, and training code
- **No Distillation**: Trained from scratch on high-quality PixMo dataset
- **Fast Inference**: MoE architecture for efficient processing
- **Self-contained**: Fully dockerized with all dependencies

### Prerequisites

1. NVIDIA GPU with at least 4GB VRAM (RTX 3060, A10, etc.)
2. Docker with NVIDIA runtime support
3. HuggingFace account (optional, for faster downloads)

### Quickstart

1. Set your HuggingFace token (optional but recommended):
```bash
export HF_TOKEN=your_huggingface_token
```

2. Build the Docker image:
```bash
bash examples/perception/vision_language/molmoe_1b/build.sh
```

3. Run OCR on an image:
```bash
bash examples/perception/vision_language/molmoe_1b/predict.sh --image ../test_images/sample.jpg
```

### Usage

#### OCR - Text Extraction

Extract text from a document:
```bash
bash examples/perception/vision_language/molmoe_1b/predict.sh \
    --image path/to/document.jpg \
    --task ocr
```

#### Image Captioning

Generate detailed image captions:
```bash
bash examples/perception/vision_language/molmoe_1b/predict.sh \
    --image path/to/image.jpg \
    --task caption
```

#### Visual Question Answering

Ask questions about an image:
```bash
bash examples/perception/vision_language/molmoe_1b/predict.sh \
    --image path/to/image.jpg \
    --task vqa \
    --prompt "How many people are in this image?"
```

### Advanced Usage

#### JSON Output

Save results as JSON with metadata:
```bash
bash examples/perception/vision_language/molmoe_1b/predict.sh \
    --image document.jpg \
    --output results.json \
    --format json
```

#### Direct Docker Run

For more control:
```bash
docker run --runtime nvidia \
    -v $(pwd)/examples/perception/vision_language/molmoe_1b:/workspace \
    -v $(pwd)/${CVL_HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    molmoe-1b \
    python3 predict.py \
        --image examples/sample.jpg \
        --task ocr
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image or URL (default: `examples/sample.jpg`)
- `--model-id` - HuggingFace model ID (default: `allenai/MolmoE-1B-0924`)
- `--task` - Task type: `ocr`, `caption`, or `vqa` (default: `ocr`)
- `--prompt` - Custom prompt for VQA tasks (required for VQA)
- `--output` - Output file path (default: `outputs/result.txt`)
- `--format` - Output format: `txt` or `json` (default: `txt`)
- `--device` - Device: `cuda`, `mps`, or `cpu` (default: auto-detect)

### Model Details

- **Name**: MolmoE-1B-0924
- **Organization**: Allen Institute for AI (AI2)
- **Size**: 6.9B total parameters, 1.2B active (MoE)
- **Architecture**: Mixture-of-Experts with 64 experts, 8 active per token
- **Released**: September 2024
- **License**: Apache 2.0
- **Training**: Trained from scratch on PixMo dataset (no distillation!)

### Performance

**Benchmarks:**
- **Nearly matches GPT-4V** on both academic and human evaluations
- **SOTA among <2B models** - best performance for similarly-sized open models
- Strong on DocVQA, TextVQA, VQA v2.0, AI2D
- Exceptional value: frontier-level performance at 1.2B active parameters

**Requirements:**
- **Memory**: ~3GB VRAM for inference (FP16)
- **Efficiency**: 70% less memory than comparable dense models

### Capabilities

1. **OCR**: High-accuracy text extraction and transcription
2. **Image Captioning**: Detailed, context-aware descriptions
3. **Visual QA**: Answer complex questions about images
4. **Document Understanding**: Extract information from documents
5. **Visual Reasoning**: Strong reasoning capabilities despite small size

### Supported Use Cases

- Document processing and OCR
- Image understanding and captioning
- Visual question answering
- Educational content analysis
- Diagram and chart interpretation
- General multimodal understanding

### How It Works

This implementation uses the Hugging Face Transformers library with `trust_remote_code=True`. MolmoE-1B automatically downloads from HuggingFace on first run (~3GB for active parameters). The model is cached in `$HOME/.cache/huggingface` and reused for subsequent runs.

**Key Implementation Details:**
- Uses `AutoModelForCausalLM` and `AutoProcessor`
- MoE architecture: 64 experts total, 8 active per token
- Auto-detects optimal device (CUDA/MPS/CPU)
- Simple, clean ~300 line implementation
- Trained on PixMo dataset with 200+ word detailed captions

### What Makes MolmoE-1B Special?

1. **No Distillation**: Unlike most small VLMs, MolmoE-1B was trained from scratch without distilling from larger models
2. **PixMo Dataset**: Trained on high-quality data with speech-based annotations (200+ words per image)
3. **Fully Open**: Weights + training data + training code all publicly available
4. **Exceptional Efficiency**: MoE architecture provides frontier performance at 1.2B active parameters
5. **Research-Friendly**: Complete transparency for reproducibility and research

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Close other GPU applications
2. Use `--device cpu` for CPU inference (slower)
3. Try a GPU with more VRAM (4GB+ recommended)

#### Model Download Issues

If the model fails to download:
1. Set `HF_TOKEN` environment variable
2. Check internet connectivity
3. Verify HuggingFace access
4. Check available disk space (~3GB needed)

#### Slow Inference

For faster inference:
1. Ensure GPU is being used (`--device cuda`)
2. Pre-download model to cache
3. Use FP16 precision (automatic on CUDA)

### Testing

Run the smoke test to verify installation:
```bash
bash examples/perception/vision_language/molmoe_1b/test.sh
```

### Directory Structure

```
examples/perception/vision_language/molmoe_1b/
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── predict.py          # Main inference script
├── build.sh            # Build Docker image
├── predict.sh          # Run inference wrapper
├── test.sh             # Smoke test
├── README.md           # This file
├── example.yaml        # Example metadata
├── .gitignore          # Git ignore rules
└── outputs/            # Results saved here

# Shared test images (used by all VLM examples):
../test_images/
└── sample.jpg          # Shared test image
```

### Output Location

By default, outputs are saved to:
- `examples/perception/vision_language/molmoe_1b/outputs/`

### Model Cache

Model weights are cached in:
- `~/.cache/huggingface/`

Size: ~3GB (persists across runs)

### Comparison with Other Models

| Model | Active Params | Performance | VRAM | License |
|-------|--------------|-------------|------|---------|
| **MolmoE-1B** | **1.2B** | **~GPT-4V** | **3GB** | **Apache 2.0** |
| Moondream2 | 1.93B | Good | 6GB | Apache 2.0 |
| Moondream3 | 2B (MoE) | Very Good | 12GB | BSL 1.1 |
| Florence-2-Large | 0.77B | Good | 2GB | MIT |
| Phi-3.5-vision | 4.2B | Excellent | 8GB | MIT |

MolmoE-1B offers the best performance-to-size ratio among small VLMs.

### Advantages Over Larger Models

- **10x more efficient**: Comparable performance to much larger models
- **Edge deployment**: Small enough for resource-constrained environments
- **Fast inference**: MoE architecture enables quick responses
- **Fully open**: No restrictions on use, modification, or research
- **Cost-effective**: Lower inference costs than larger models

### References

- [MolmoE-1B on HuggingFace](https://huggingface.co/allenai/MolmoE-1B-0924)
- [Allen Institute Molmo Blog](https://allenai.org/blog/molmo)
- [Molmo GitHub Repository](https://github.com/allenai/molmo)
- [Molmo & PixMo Paper](https://arxiv.org/html/2409.17146v2)
- [Allen Institute AI2](https://allenai.org/)

### License

This example uses MolmoE-1B which is licensed under Apache 2.0. The model, weights, training data, and training code are all openly available.

### Citation

If you use MolmoE-1B in your research, please cite:

```bibtex
@article{molmo2024,
  title={Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models},
  author={Allen Institute for AI},
  journal={arXiv preprint},
  year={2024}
}
```
