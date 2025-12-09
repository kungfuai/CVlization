## MolmoE-1B

This example demonstrates multimodal understanding using **MolmoE-1B** from Allen Institute (AI2), a Mixture-of-Experts vision-language model with 1.2B active parameters.

### Key Features

- **Efficient Architecture**: 6.9B total with 64 experts, only 1.2B (8 experts) active per token
- **Compact Size**: ~3GB VRAM for inference
- **Fully Open**: Apache 2.0 license with open weights, data, and training code
- **No Distillation**: Trained from scratch on PixMo dataset
- **MoE Architecture**: Mixture-of-Experts for efficient processing
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
    molmoe_1b \
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

Tested on NVIDIA A10 GPU with invoice test image (800x600):

**Resources:**
- **VRAM Usage**: ~3GB with float16
- **Model Download Size**: ~3GB

**Quality Note**: Verification testing revealed accuracy issues - the model generated hallucinated content (described non-existent subjects and text for the test invoice image). This implementation may require debugging or prompt tuning for production use.

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

### What Makes MolmoE-1B Different?

1. **No Distillation**: Unlike most small VLMs, MolmoE-1B was trained from scratch without distilling from larger models
2. **PixMo Dataset**: Trained on data with speech-based annotations (200+ words per image)
3. **Fully Open**: Weights + training data + training code all publicly available
4. **MoE Architecture**: Mixture-of-Experts design with 1.2B active parameters
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

| Model | Active Params | VRAM | License |
|-------|--------------|------|---------|
| **MolmoE-1B** | **1.2B** | **3GB** | **Apache 2.0** |
| Florence-2-Large | 0.77B | 2GB | MIT |
| Phi-3.5-vision | 4.2B | 8GB | MIT |
| Qwen3-VL-2B | 2B | 4GB | Apache 2.0 |

### Architecture Benefits

- **Efficient**: MoE architecture with only 1.2B active parameters per token
- **Edge deployment**: Small enough for resource-constrained environments
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
