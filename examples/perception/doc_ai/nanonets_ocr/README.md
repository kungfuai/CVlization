# Nanonets OCR2 3B

Inference example using **Nanonets-OCR2-3B**, a 3.75B parameter Vision-Language Model (VLM) for advanced document understanding and OCR.

## Overview

Nanonets-OCR2-3B is a multimodal image-to-text model based on **Qwen2.5-VL-3B-Instruct** that specializes in converting complex documents into structured markdown format. It excels at extracting and preserving document structure including tables, forms, equations, charts, and handwritten content.

### Key Features

- **Structured Output**: Converts documents to clean markdown with preserved formatting
- **LaTeX Equations**: Recognizes and converts mathematical equations to LaTeX
- **Table Extraction**: Accurately extracts tables and converts to markdown/HTML tables
- **Visual Elements**: Handles charts, diagrams, signatures, and watermarks
- **Multilingual**: Supports English, Chinese, French, Spanish, and more
- **Handwriting**: Processes handwritten documents
- **Visual Question Answering (VQA)**: Answer questions about document content
- **Chart Understanding**: Extracts flow charts as Mermaid code

## Model Comparison

This example is part of our Document AI collection. Here's how it compares:

| Model | Size | Type | Strengths | Use Case |
|-------|------|------|-----------|----------|
| **nanonets-ocr** | 3.75B | VLM | Structured markdown, LaTeX, forms, VQA | Complex documents with mixed content |
| [granite-docling](../granite-docling/) | 258M | VLM | Fast, lightweight, docling format | Quick document extraction |
| [moondream3](../moondream3/) | 3B | VLM | General vision-language, VQA | Interactive document Q&A |
| [surya](../surya/) | - | Traditional OCR | Multilingual, layout detection | Pure text extraction |
| [docling](../docling-serve/) | - | Layout Analysis | PDF parsing, structure detection | Document layout understanding |

**When to use Nanonets-OCR2-3B:**
- Documents with complex tables and forms
- Scientific papers with LaTeX equations
- Mixed content (text + charts + diagrams)
- Need structured markdown output
- Visual question answering about document content

## Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU with sufficient VRAM (8GB+ recommended for 3B model)
- `nvidia-docker2` installed

## Quick Start

### 1. Build the Container

```bash
bash build.sh
```

This builds the Docker image with all dependencies (~5-10 minutes first time).

### 2. Run OCR on a Document

```bash
# Try with the shared test document (same as granite-docling)
bash predict.sh examples/sample.pdf

# Save to file
bash predict.sh examples/sample.pdf --output outputs/result.md

# JSON format
bash predict.sh examples/test_image.png --format json --output outputs/result.json
```

### 3. Visual Question Answering

```bash
# Ask questions about the document
bash predict.sh examples/test_image.png --mode vqa --question "What is shown in this image?"

# Extract specific information from PDF
bash predict.sh examples/sample.pdf --mode vqa --question "What is the main topic of this document?"
```

## Usage Examples

### Document OCR

Extract structured content from complex documents:

```bash
# Scientific paper with equations
bash predict.sh paper.pdf --output paper.md

# Form with checkboxes and tables
bash predict.sh form.jpg --output form.md

# Handwritten notes
bash predict.sh notes.jpg --output notes.md
```

### Table Extraction

```bash
# Financial report with tables
bash predict.sh report.pdf --output report.md

# The markdown output will include properly formatted tables
```

### Visual Question Answering

```bash
# Chart analysis
bash predict.sh sales_chart.png --mode vqa --question "What was the highest sales month?"

# Document understanding
bash predict.sh contract.pdf --mode vqa --question "What is the contract end date?"

# Form field extraction
bash predict.sh application.jpg --mode vqa --question "What is the applicant's name?"
```

### Batch Processing

```bash
# Process multiple documents
for file in documents/*.pdf; do
    basename=$(basename "$file" .pdf)
    bash predict.sh "$file" --output "outputs/${basename}.md"
done
```

## Output Format

### Markdown (Default)

The model converts documents to clean, structured markdown:

```markdown
# Document Title

## Section 1

Regular text content with **bold** and *italic* formatting.

### Tables

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

### Equations

Inline equation: $E = mc^2$

Block equation:
$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

### Lists

- Item 1
- Item 2
  - Nested item
```

### JSON Format

```json
{
  "input_file": "document.jpg",
  "model": "nanonets/Nanonets-OCR2-3B",
  "mode": "ocr",
  "content": "# Document Title\n\nExtracted content..."
}
```

## Advanced Options

```bash
# Use CPU (slower, no GPU required)
bash predict.sh document.jpg --device cpu

# Increase max tokens for long documents
bash predict.sh long_doc.pdf --max-tokens 8192

# Combine options
bash predict.sh complex.pdf --output result.md --device cuda --max-tokens 6000
```

## Direct Python Usage

You can also use `predict.py` directly without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python predict.py document.jpg --output output.md
python predict.py chart.png --mode vqa --question "What is the trend?"
```

## Performance Notes

- **Model Size**: 3.75B parameters (~7.5GB VRAM with bfloat16)
- **First Run**: Downloads model from HuggingFace (~7GB, cached after first run)
- **Speed**: ~2-5 seconds per page on modern GPUs (A100, RTX 4090)
- **CPU Mode**: Significantly slower but functional

## Limitations

- Maximum context length: 4096 tokens (configurable with `--max-tokens`)
- Very long documents may need to be processed page-by-page
- Handwriting recognition quality varies with writing clarity
- Complex nested structures may require output verification

## Quality Comparison

To compare quality against other models in this repo, use the shared test files:

```bash
# Process same document with different models
bash predict.sh examples/sample.pdf --output outputs/nanonets_output.md

cd ../granite-docling
bash predict.sh sample.pdf --output outputs/granite_output.md

cd ../docling-serve
bash predict.sh sample.pdf --format markdown --output outputs/docling_output.md

cd ../moondream3
bash predict.sh examples/sample.jpg --output outputs/moondream_output.md

# Compare the outputs manually or with diff tools
diff ../nanonets-ocr/outputs/nanonets_output.md ../granite-docling/outputs/granite_output.md
```

**Expected Differences:**
- **Nanonets**: Best for structured markdown with LaTeX equations
- **Granite-Docling**: Faster, more compact, docling JSON format
- **Surya**: Better for pure text extraction, multilingual OCR
- **Moondream**: Better for interactive Q&A and image understanding

## Troubleshooting

### Out of Memory

```bash
# Use CPU mode
bash predict.sh document.jpg --device cpu

# Or reduce max tokens
bash predict.sh document.jpg --max-tokens 2048
```

### Docker GPU Not Working

```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Model Download Slow

The model (~7GB) is downloaded on first run and cached in `~/.cache/huggingface/`. Subsequent runs are fast.

## Reference

- **Model Card**: https://huggingface.co/nanonets/Nanonets-OCR2-3B
- **Base Model**: Qwen2.5-VL-3B-Instruct
- **License**: Check model card for licensing details

## Contributing

To improve this example:
1. Test with diverse document types
2. Report issues with specific document categories
3. Share quality comparisons with other models
4. Suggest additional features or use cases
