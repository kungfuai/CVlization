## CHURRO-3B - Historical Document OCR

CHURRO (Contextualized Historical Universal Reader for Robust OCR) is a state-of-the-art model for transcribing handwritten and printed text from historical documents across 22 centuries and 46 language clusters.

**Model**: [stanford-oval/churro-3B](https://huggingface.co/stanford-oval/churro-3B)
**Repository**: [stanford-oval/Churro](https://github.com/stanford-oval/Churro)
**Base Model**: Qwen2.5-VL-3B-Instruct
**License**: qwen-research

### Features

- **Historical Expertise**: Specialized for documents from 22 centuries
- **46 Language Clusters**: Including historical and dead languages
- **Handwriting Support**: Excellent recognition of handwritten text
- **Cost Efficient**: 15.5× lower cost than Gemini 2.5 Pro with higher accuracy
- **XML Output**: Structured output with layout preservation
- **Long Context**: Up to 20,000 tokens for complete page transcription
- **Compact Size**: 3B parameters (~6-8GB VRAM)

### Key Innovation

CHURRO achieves exceptional accuracy on historical documents by:
- Training on ~100K pages from 155 historical collections
- Understanding degraded document conditions
- Preserving layout information in XML structure
- Supporting diverse writing systems and languages
- Outperforming larger commercial models at fraction of cost

### Prerequisites

1. NVIDIA GPU with at least 8GB VRAM (16GB+ recommended for large documents)
2. Docker with NVIDIA runtime support
3. CUDA 11.8 or higher

### Quickstart

1. Build the Docker image:
```bash
bash examples/perception/doc_ai/churro_3b/build.sh
```

2. Run OCR on a document:
```bash
bash examples/perception/doc_ai/churro_3b/predict.sh  # Uses shared test image
```

3. Run smoke test:
```bash
bash examples/perception/doc_ai/churro_3b/test.sh
```

### Usage

#### Basic Transcription

Transcribe a historical document with XML structure:
```bash
bash examples/perception/doc_ai/churro_3b/predict.sh \
    --image path/to/manuscript.jpg
```

#### Plain Text Output

Strip XML markup for plain text:
```bash
bash examples/perception/doc_ai/churro_3b/predict.sh \
    --image path/to/document.png \
    --strip-xml \
    --output result.txt
```

#### Short Testing

For quick tests, use fewer tokens:
```bash
bash examples/perception/doc_ai/churro_3b/predict.sh \
    --image sample.jpg \
    --max-new-tokens 500 \
    --strip-xml
```

#### JSON Output

Save with metadata:
```bash
bash examples/perception/doc_ai/churro_3b/predict.sh \
    --image scroll.jpg \
    --format json \
    --output result.json
```

### Advanced Usage

#### Full Page Transcription

Use maximum context for complete pages:
```bash
bash examples/perception/doc_ai/churro_3b/predict.sh \
    --image full_page.jpg \
    --max-new-tokens 20000 \
    --output full_transcription.txt
```

#### Direct Docker Run

For more control:
```bash
docker run --rm --gpus=all \
    -v $(pwd)/examples/perception/doc_ai/churro_3b:/workspace \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --shm-size=8g \
    cvlization/churro-3b:latest \
    python3 predict.py \
        --image manuscript.jpg \
        --strip-xml
```

### Command-Line Options

The `predict.py` script supports:

- `--image` - Path to input image (default: shared test image)
- `--output` - Output file path (default: `outputs/result.{format}`)
- `--format` - Output format: `txt` or `json` (default: `txt`)
- `--max-new-tokens` - Maximum tokens (default: 20000 for full pages, 500 for testing)
- `--strip-xml` - Remove XML markup from output (default: keep XML)
- `--model` - Model ID (default: `stanford-oval/churro-3B`)
- `--device` - Device: `cuda`, `mps`, or `cpu` (default: auto-detect)

### Model Details

- **Name**: CHURRO-3B
- **Parameters**: 3 billion
- **Architecture**: Vision-language model (Qwen2.5-VL-3B-Instruct base)
- **Training Data**: ~100K pages from 155 historical collections
- **Languages**: 46 language clusters across 22 centuries
- **License**: qwen-research
- **Repository**: [GitHub](https://github.com/stanford-oval/Churro)
- **Paper**: Coming soon

### Performance

**VRAM Requirements:**
- Minimum: 6GB (inference only)
- Recommended: 8GB (optimal performance)
- Maximum: 16GB (large documents with full context)

**Inference Speed:**
- Short excerpts (500 tokens): 2-5 seconds
- Full pages (20000 tokens): 30-60 seconds

**Cost Comparison:**
- **CHURRO-3B**: Baseline cost
- **Gemini 2.5 Pro**: 15.5× higher cost with lower accuracy

**Strengths:**
- Historical documents (22 centuries)
- Handwritten text recognition
- Degraded document conditions
- Multilingual support (46 language clusters)
- Cost efficiency vs commercial models
- Layout preservation (XML structure)

### Supported Languages & Scripts

CHURRO supports 46 language clusters including:
- **Ancient**: Latin, Ancient Greek, Classical Chinese
- **Medieval**: Old English, Middle French, Medieval Latin
- **Modern**: English, Spanish, French, German, Italian, etc.
- **Asian**: Chinese, Japanese, Korean, Arabic, Hebrew
- **Dead Languages**: Many historical and extinct languages
- **Scripts**: Latin, Greek, Cyrillic, Arabic, Hebrew, CJK, and more

### Use Cases

1. **Historical Research**: Digitize manuscripts and archival documents
2. **Library Digitization**: Mass transcription of historical collections
3. **Academic Studies**: Extract text from primary source documents
4. **Genealogy**: Transcribe family records and historical documents
5. **Museum Collections**: Make historical texts searchable and accessible
6. **Paleography**: Study historical writing systems
7. **Linguistics**: Analyze dead and historical languages
8. **Cultural Heritage**: Preserve endangered textual heritage

### Output Format

**XML Structure (default):**
CHURRO outputs structured XML following the "HistoricalDocument" schema:
- Captures layout information
- Marks editorial interventions
- Identifies missing or damaged text
- Preserves document structure

**Plain Text (with --strip-xml):**
Clean transcribed text without markup

**JSON (with --format json):**
```json
{
  "text": "Transcribed content...",
  "model": "stanford-oval/churro-3B",
  "image_size": [1200, 1600],
  "max_new_tokens": 20000,
  "strip_xml": false
}
```

### How It Works

CHURRO uses a vision-language architecture:

1. **Image Encoding** - Processes document with vision encoder
2. **Context Understanding** - Analyzes historical context and degradation
3. **Language Detection** - Identifies language and writing system
4. **Transcription** - Generates text with layout awareness
5. **Structured Output** - Produces XML with document semantics

This enables:
- High accuracy on degraded historical documents
- Understanding of diverse writing systems
- Preservation of document structure
- Cost-effective processing at scale

### Troubleshooting

#### Out of Memory

If you encounter CUDA OOM errors:
1. Reduce `--max-new-tokens` to 10000 or lower
2. Use smaller image sizes
3. Close other GPU applications
4. Use a GPU with more VRAM

#### Model Download Issues

Models are downloaded from HuggingFace on first run (~6-8GB):
1. Check internet connectivity
2. Ensure sufficient disk space (~15GB total)
3. Check `~/.cache/huggingface/` for cached models

#### Poor Quality Results

If transcription quality is unsatisfactory:
1. Ensure image is high resolution (300+ DPI recommended)
2. Try keeping XML structure (remove `--strip-xml`)
3. Use full context window (`--max-new-tokens 20000`)
4. Check that language is in supported list

### Directory Structure

```
examples/perception/doc_ai/churro_3b/
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
```

### Output Location

By default, outputs are saved to:
- `examples/perception/doc_ai/churro_3b/outputs/`

### Model Cache

Model weights are cached in:
- `~/.cache/huggingface/hub/`

Size: ~6-8GB (persists across runs)

### Comparison with Other Models

| Feature | CHURRO-3B | Chandra OCR | Gemini 2.5 Pro | GPT-4o |
|---------|-----------|-------------|----------------|---------|
| **Parameters** | 3B | ~7B | Unknown | Unknown |
| **Historical Focus** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Languages** | 46 clusters | 40+ | Many | Many |
| **Cost Efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ |
| **XML Structure** | ✅ | ❌ | ❌ | ❌ |
| **VRAM** | 8GB | 16-24GB | N/A (API) | N/A (API) |
| **Handwriting** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Dead Languages** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**When to use CHURRO:**
- Historical documents (pre-1900)
- Dead or ancient languages
- Cost-sensitive applications
- Need structured XML output
- Offline/local processing required
- Academic research projects

### Production Deployment

For production use:
1. Pre-download models to cache
2. Adjust `--max-new-tokens` based on document types
3. Use `--strip-xml` if plain text is sufficient
4. Monitor GPU utilization with `nvidia-smi`
5. Consider batch processing for large collections
6. Implement error handling for edge cases

### Integration with CVlization

This example follows CVlization patterns:
- Dual-mode execution (standalone or via CVL)
- Optional `cvlization.paths` import with fallback
- Standardized input/output path handling
- Docker-based reproducibility
- HuggingFace model caching
- GPU-optimized inference
- Consistent CLI interface

### References

- [CHURRO on HuggingFace](https://huggingface.co/stanford-oval/churro-3B)
- [CHURRO GitHub Repository](https://github.com/stanford-oval/Churro)
- [Qwen2.5-VL Base Model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [CVlization Repository](https://github.com/kungfuai/CVlization)

### License

CHURRO-3B is released under the **qwen-research** license.

Key points:
- ✅ Research use allowed
- ✅ Academic use allowed
- ⚠️ Check license for commercial use restrictions

See the [model card](https://huggingface.co/stanford-oval/churro-3B) for full license details.

This example code is part of CVlization and follows the project's licensing terms.

### Community & Support

- HuggingFace Model Card: [stanford-oval/churro-3B](https://huggingface.co/stanford-oval/churro-3B)
- GitHub Issues: [stanford-oval/Churro](https://github.com/stanford-oval/Churro/issues)
- CVlization Issues: [GitHub](https://github.com/kungfuai/CVlization/issues)

### Citation

If you use CHURRO in your research, please cite:

```bibtex
@misc{churro2024,
  title={CHURRO: Contextualized Historical Universal Reader for Robust OCR},
  author={Stanford OVAL Lab},
  year={2024},
  url={https://github.com/stanford-oval/Churro}
}
```

### Acknowledgments

- Developed by Stanford OVAL Lab
- Based on Qwen2.5-VL architecture
- Trained on 155 historical collections
- Built with HuggingFace Transformers
