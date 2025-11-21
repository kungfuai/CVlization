## NVIDIA Nemotron Parse v1.1 - Document Structure Understanding with Spatial Grounding

This example demonstrates how to use NVIDIA Nemotron Parse v1.1, a lightweight (<1B parameters) transformer-based vision-encoder-decoder model specifically designed for document structure understanding. It extracts text, tables, and layout elements with bounding boxes and semantic class labels.

**Model**: [nvidia/NVIDIA-Nemotron-Parse-v1.1 on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)
**License**: NVIDIA Community Model License
**Architecture**: ViT-H vision encoder + mBart decoder

### Features

- **Spatial Grounding**: Extracts bounding boxes for all document elements
- **Semantic Classification**: Labels elements (titles, sections, captions, footnotes, lists, tables)
- **Table Extraction**: Handles complex table formatting and structures
- **Mathematical Content**: Supports LaTeX equation extraction
- **Compact Model**: <1B parameters for efficient inference
- **Structured Output**: Markdown with embedded bounding boxes and class labels
- **Multi-format Output**: JSON, Markdown, or plain text
- **GPU Optimized**: Supports NVIDIA Hopper, Ampere, and Turing architectures
- **vLLM Compatible**: Ready for production deployment with vLLM

### Key Innovation

Nemotron Parse combines a powerful vision encoder (ViT-H C-RADIO) with an efficient decoder to provide:
- Document structure understanding beyond simple OCR
- Spatial coordinates for precise element localization
- Semantic classification for downstream processing
- Efficient inference with <1B parameters
- Commercial-ready with NVIDIA Community License

### Prerequisites

1. NVIDIA GPU with at least 12GB VRAM (16GB+ recommended)
2. Docker with NVIDIA runtime support
3. CUDA 11.8 or higher
4. Linux operating system

### Quickstart

1. Build the Docker image:
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/build.sh
```

2. Run document parsing on an image:
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/predict.sh  # Uses shared test image by default
```

3. Run smoke test:
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/test.sh
```

### Usage

#### Basic Markdown Extraction (Default)

Extract document content as markdown with spatial grounding:
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/predict.sh \
    --image path/to/document.jpg \
    --output result.md
```

#### JSON Output with Bounding Boxes and Classes

Get structured output with spatial coordinates and element classifications:
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/predict.sh \
    --image path/to/document.pdf \
    --format json \
    --output result.json
```

The JSON output includes:
- `markdown`: Clean text content
- `bboxes`: List of [x1, y1, x2, y2] coordinates
- `classes`: Semantic labels (title, section, table, etc.)
- `structured_elements`: Combined bbox + class + text for each element

#### Plain Text Extraction

Extract plain text without formatting:
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/predict.sh \
    --image path/to/invoice.png \
    --format txt \
    --output result.txt
```

### Advanced Usage

#### Custom Prompts

The default prompt extracts bounding boxes, classes, and markdown:
```
</s><s><predict_bbox><predict_classes><output_markdown>
```

You can customize the prompt to control output:

**Markdown only (no spatial grounding):**
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/predict.sh \
    --image document.jpg \
    --prompt "</s><s><output_markdown>" \
    --output result.md
```

**Bounding boxes only:**
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/predict.sh \
    --image document.jpg \
    --prompt "</s><s><predict_bbox>" \
    --output result.json
```

#### Control Generation Length

Adjust maximum tokens for longer documents:
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/predict.sh \
    --image long-document.jpg \
    --max-new-tokens 8192 \
    --output result.md
```

#### Device Selection

Explicitly select the device:
```bash
bash examples/perception/doc_ai/nvidia_nemotron_parse/predict.sh \
    --image document.jpg \
    --device cuda \
    --output result.md
```

#### Direct Docker Run

For more control over execution:
```bash
docker run --rm --gpus=all \
    -v $(pwd)/examples/perception/doc_ai/nvidia_nemotron_parse:/workspace \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --shm-size=8g \
    cvlization/nvidia-nemotron-parse:latest \
    python3 predict.py \
        --image document.jpg \
        --format json \
        --output outputs/result.json
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image or PDF (default: shared test image)
- `--output` - Output file path (default: `outputs/result.{format}`)
- `--format` - Output format: `txt`, `json`, or `md` (default: `md`)
- `--model` - HuggingFace model ID (default: `nvidia/NVIDIA-Nemotron-Parse-v1.1`)
- `--prompt` - Custom prompt string (default: `</s><s><predict_bbox><predict_classes><output_markdown>`)
- `--max-new-tokens` - Maximum tokens to generate (default: 4096)
- `--device` - Device to use: `cuda`, `mps`, or `cpu` (default: auto-detect)

### Model Details

- **Name**: NVIDIA Nemotron Parse v1.1
- **Parameters**: <1 billion (efficient inference)
- **Vision Encoder**: ViT-H (C-RADIO model)
- **Decoder**: mBart with 10 blocks
- **Adapter**: 1D convolutions (13,184 → 3,201 tokens)
- **Training Compute**: 2.2e+22 FLOPs
- **Input Resolution**: 1024×1280 to 1648×2048 RGB images
- **Precision**: bfloat16 (Hopper/Ampere) or float16 (Turing)
- **License**: NVIDIA Community Model License (CC-BY-4.0 for tokenizer)
- **Repository**: [HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)

### Performance

**VRAM Requirements:**
- Minimum: 8GB (inference only)
- Recommended: 12-16GB (optimal performance)
- Maximum: 24GB (large batches/documents)

**Inference Speed:**
- Single page: 2-5 seconds (depending on GPU)
- Batch processing: Scales efficiently with vLLM

**Strengths:**
- Document structure understanding
- Spatial grounding with bounding boxes
- Table extraction with complex formatting
- Mathematical equation recognition (LaTeX)
- Semantic element classification
- Efficient inference (<1B params)
- Commercial deployment ready

### Supported Document Types

- **PDFs**: Technical papers, reports, presentations
- **Images**: Scanned documents, photos of documents
- **Forms**: Structured forms and templates
- **Tables**: Complex tables with merged cells
- **Mixed Content**: Text, images, equations, tables
- **PowerPoint**: Slide exports and presentations

### Semantic Classes

Nemotron Parse can identify and classify:
- **Titles**: Document and section titles
- **Sections**: Section headers and subheaders
- **Body Text**: Main content paragraphs
- **Captions**: Figure and table captions
- **Footnotes**: Footnote references and text
- **Lists**: Bulleted and numbered lists
- **Tables**: Table structures and cells
- **Equations**: Mathematical formulas
- **Figures**: Images and diagrams

### Use Cases

1. **Document Digitization**: Convert documents to structured, machine-readable formats
2. **Information Extraction**: Extract specific elements with spatial context
3. **Table Extraction**: Parse complex tables for data analysis
4. **Layout Analysis**: Understand document structure for downstream processing
5. **Form Processing**: Extract fields with precise locations
6. **Academic Papers**: Parse papers including equations and tables
7. **Invoice Processing**: Extract line items with spatial grounding
8. **Contract Analysis**: Identify clauses and sections with location data

### How It Works

Nemotron Parse uses a vision-encoder-decoder architecture:

1. **Image Encoding** - ViT-H vision encoder processes document at high resolution (1024×1280 to 1648×2048)
2. **Token Compression** - Adapter layer compresses 13,184 visual tokens to 3,201 tokens
3. **Structure Understanding** - mBart decoder generates structured output with embeddings
4. **Spatial Grounding** - Predicts bounding boxes for each document element
5. **Semantic Classification** - Assigns class labels (title, section, table, etc.)
6. **Content Generation** - Outputs markdown text with LaTeX equations

This enables:
- Precise element localization with bounding boxes
- Semantic understanding of document structure
- Rich structured output for downstream processing
- Efficient inference with compact model size

### Output Format Examples

#### Markdown Output (`.md`)

```markdown
# Document Parse Output

**Elements detected:** 15

---

# Document Title

## Section 1

This is the main body text of the document...

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |

## Section 2

More content here...
```

#### JSON Output (`.json`)

```json
{
  "markdown": "# Document Title\n\n## Section 1\n\nBody text...",
  "bboxes": [
    [100, 50, 500, 80],
    [100, 100, 500, 130],
    [100, 150, 500, 400]
  ],
  "classes": [
    "title",
    "section",
    "body"
  ],
  "structured_elements": [
    {
      "bbox": [100, 50, 500, 80],
      "class": "title",
      "text": "Document Title"
    },
    {
      "bbox": [100, 100, 500, 130],
      "class": "section",
      "text": "Section 1"
    },
    {
      "bbox": [100, 150, 500, 400],
      "class": "body",
      "text": "Body text..."
    }
  ],
  "num_elements": 3
}
```

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Check VRAM usage with `nvidia-smi`
2. Close other GPU applications
3. Reduce `--max-new-tokens` to 2048 or lower
4. Use a GPU with more VRAM (16GB+ recommended)
5. Process smaller images or reduce resolution

#### Model Download Issues

Models are automatically downloaded from HuggingFace on first run (~3-4GB):
1. Check internet connectivity
2. Verify HuggingFace access (no authentication required)
3. Ensure sufficient disk space (~10GB for models + dependencies)
4. Check `~/.cache/huggingface/` for cached models
5. Wait patiently for initial download (5-10 minutes)

#### Slow Inference

For faster inference:
1. Ensure GPU is being used (check `nvidia-smi`)
2. Pre-download models to cache before production use
3. Use bfloat16 precision on modern GPUs (Ampere/Hopper)
4. Consider vLLM deployment for production workloads
5. Use SSD for model cache directory

#### Poor Quality Results

If parse quality is unsatisfactory:
1. Check image quality and resolution (higher is better)
2. Ensure image is properly oriented
3. Try adjusting `--max-new-tokens` for longer documents
4. Use the full prompt for best results: `</s><s><predict_bbox><predict_classes><output_markdown>`
5. Check that document is in a supported language/format

#### Empty or Truncated Output

If output is incomplete:
1. Increase `--max-new-tokens` (default: 4096, try 8192)
2. Check for errors in console output
3. Verify sufficient VRAM is available
4. Ensure image format is supported (PNG, JPG)

### Directory Structure

```
examples/perception/doc_ai/nvidia_nemotron_parse/
├── Dockerfile           # Container definition
├── predict.py          # Main inference script
├── build.sh            # Build Docker image
├── predict.sh          # Run inference wrapper
├── test.sh            # Smoke test script
├── requirements.txt    # Python dependencies
├── example.yaml        # CVlization metadata
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── outputs/            # Results saved here (git-ignored)

# Note: Default test image is shared from:
# ../leaderboard/test_data/sample.jpg
# (mounted as /cvlization_repo/... in Docker)
```

### Output Location

By default, outputs are saved to:
- `examples/perception/doc_ai/nvidia_nemotron_parse/outputs/`

### Model Cache

Model weights are cached in:
- `~/.cache/huggingface/hub/`

Size: ~3-4GB (persists across runs)

### Deployment Options

#### Standalone Docker

Standard Docker deployment for single-node inference:
```bash
bash predict.sh --image document.jpg --output result.json
```

#### vLLM Deployment

For production workloads, deploy with vLLM:
```python
from vllm import LLM

model = LLM("nvidia/NVIDIA-Nemotron-Parse-v1.1")
outputs = model.generate(images=[...], prompts=[...])
```

#### NVIDIA NIM Container

NVIDIA provides an optimized NIM container for enterprise deployment:
- Pre-built TensorRT-LLM optimization
- Kubernetes-ready containerization
- Enterprise support and SLA

See [NVIDIA NIM documentation](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nim/) for details.

### Production Deployment

For production use:
1. **Pre-download models**: Run once to cache in `~/.cache/huggingface/`
2. **Monitor VRAM**: Use `nvidia-smi` to track GPU utilization
3. **Batch processing**: Process multiple documents in parallel for efficiency
4. **vLLM integration**: Use vLLM for optimized throughput
5. **Error handling**: Implement retry logic for network/GPU errors
6. **Logging**: Track processing time and success rates
7. **License compliance**: Review NVIDIA Community Model License

### Integration with CVlization

This example follows CVlization patterns:
- Dual-mode execution (standalone or via CVL)
- Optional `cvlization.paths` import with fallback
- Standardized input/output path handling
- Docker-based reproducibility
- HuggingFace model caching
- GPU-optimized inference
- Consistent CLI interface

### Comparison with Other Models

| Feature | Nemotron Parse | Chandra OCR | Docling | Granite-Docling |
|---------|----------------|-------------|---------|-----------------|
| Parameters | <1B | ~7B | Pipeline | 258M |
| Spatial Grounding | ✅ Bounding boxes | ❌ | ✅ Layout | ❌ |
| Semantic Classes | ✅ Rich labels | ❌ | ⚠️ Basic | ❌ |
| Table Extraction | ✅ | ✅ | ✅ | ✅ |
| Math Equations | ✅ LaTeX | ✅ | ❌ | ⚠️ Limited |
| VRAM | 12GB | 16-24GB | 8GB | 4GB |
| Inference Speed | Fast | Slow | Medium | Fast |
| License | NVIDIA Comm. | OpenRAIL-M | Apache 2.0 | Apache 2.0 |

**When to use Nemotron Parse:**
- Need spatial grounding (bounding boxes)
- Need semantic classification of elements
- Want efficient inference (<1B params)
- Building production pipelines
- Need commercial licensing clarity

### References

- [NVIDIA Nemotron Parse on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)
- [NVIDIA NIM Documentation](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nim/)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [CVlization Repository](https://github.com/kungfuai/CVlization)

### License

NVIDIA Nemotron Parse v1.1 is released under:
- **Model**: NVIDIA Community Model License
- **Tokenizer**: CC-BY-4.0 License

Key points:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Patent use allowed

See the [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1) for full license details.

This example code is part of CVlization and follows the project's licensing terms.

### Community & Support

- HuggingFace Discussions: [Model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)
- NVIDIA Developer Forums: [forums.developer.nvidia.com](https://forums.developer.nvidia.com)
- CVlization Issues: [GitHub](https://github.com/kungfuai/CVlization/issues)

### Citation

If you use NVIDIA Nemotron Parse in your research, please cite:

```bibtex
@misc{nemotron-parse-2024,
  title={NVIDIA Nemotron Parse v1.1: Document Structure Understanding with Spatial Grounding},
  author={NVIDIA},
  year={2024},
  url={https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1}
}
```

### Acknowledgments

- Developed by NVIDIA
- Built on HuggingFace Transformers
- C-RADIO vision encoder
- mBart decoder architecture
- Compatible with vLLM inference engine
