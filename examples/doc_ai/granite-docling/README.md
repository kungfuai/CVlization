# Granite-Docling Document Extraction Example

End-to-end document understanding using IBM's [Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) vision-language model (VLM).

## Features

- **Single compact model**: Only 258M parameters for efficient inference
- **End-to-end**: Direct image-to-markdown/JSON conversion
- **Vision-language model**: Based on Qwen2-VL architecture with Granite 3 language backbone
- **High quality**: Rivals systems several times its size
- **Handles**: Tables, equations, code blocks, lists, and complex layouts
- **No PDF parsing needed**: Works directly with document images

## Quick Start

### 1. Build the Docker image

```bash
./build.sh
```

### 2. Extract content from a document image

```bash
# Basic usage - outputs markdown to stdout
./predict.sh document.png

# Save as JSON
./predict.sh scan.jpg --format json --output result.json

# Use GPU (requires NVIDIA Docker)
./predict.sh form.png --device cuda
```

## Usage

```bash
./predict.sh <input_image> [options]
```

### Options

- `--output <file>`: Save output to file (prints to stdout if not specified)
- `--format <format>`: Output format
  - `markdown` (default): Clean markdown with preserved structure
  - `json`: Structured JSON output
- `--device <device>`: Device for inference
  - `cpu` (default): CPU inference
  - `cuda`: GPU inference (requires NVIDIA GPU and drivers)

### Examples

```bash
# Extract to markdown
./predict.sh invoice.png

# Save as JSON
./predict.sh report.png --format json --output report.json

# Use GPU for faster inference
./predict.sh large_document.png --device cuda --format markdown
```

## Model Details

### Granite-Docling-258M

- **Architecture**: Qwen2-VL based VLM with Granite 3 language backbone
- **Parameters**: 258M (ultra-compact)
- **Vision encoder**: SigLIP2
- **Context**: Handles full document pages
- **License**: Apache 2.0

### Capabilities

- **Text extraction**: High accuracy OCR and text recognition
- **Table understanding**: Recognizes table structure (rows, columns, headers)
- **Math formulas**: Inline and display equations
- **Code blocks**: Preserves formatting
- **Layout preservation**: Maintains document hierarchy and structure
- **Multi-modal**: Processes both text and visual elements

## Comparison with Docling-Serve

| Feature | Docling-Serve | Granite-Docling |
|---------|--------------|-----------------|
| **Approach** | Pipeline (multiple models) | Single VLM |
| **Models** | DocLayNet + TableFormer + OCR | Granite-Docling-258M |
| **Size** | ~1GB+ (multiple models) | 258M parameters |
| **Input** | PDF or images | Images only |
| **Speed** | Moderate | Faster (single pass) |
| **Accuracy** | 97.9% (table extraction) | Competitive |
| **Use case** | Production pipelines | Resource-constrained environments |

## Performance

- **Speed**:
  - **GPU**: Fast inference (seconds per page)
  - **CPU**: Very slow (~15+ minutes per page, not recommended for production)
- **Memory**: ~2GB RAM for CPU inference, ~4GB VRAM for GPU
- **Accuracy**: Competitive with larger models
- **Efficiency**: 70% less memory than similar-sized alternatives

**Note**: CPU inference is extremely slow and not practical for production use. GPU is strongly recommended for this model.

## Requirements

- **CPU**: Any modern x86_64 processor
- **RAM**: 4GB minimum, 8GB recommended
- **GPU** (optional): NVIDIA GPU with CUDA support for faster inference
- **Disk**: ~3GB for Docker image + models

## Sample Image

Copy a sample image from the docling-serve example:

```bash
cp ../docling-serve/test_image.png .
./predict.sh test_image.png
```

## Output Structure

### Markdown Format
Clean markdown with:
- Hierarchical headings
- Tables in markdown format
- Lists with proper nesting
- Preserved document structure

### JSON Format
Structured data with:
- Document metadata
- Extracted content
- Optional structure information

## Limitations

- **Image input only**: Unlike docling-serve, does not directly process PDFs (convert to images first)
- **Single page**: Best for individual document pages or images
- **Quality dependent**: Accuracy depends on input image quality

## Converting PDFs to Images

If you have PDFs, convert them to images first:

```bash
# Using docling-serve's helper
cd ../docling-serve
docker run --rm -v $(pwd):/workspace -w /workspace docling-serve python create_test_image.py

# Or using other tools
convert -density 300 document.pdf page.png  # ImageMagick
pdftoppm document.pdf page -png             # Poppler
```

## References

- [Granite-Docling Model](https://huggingface.co/ibm-granite/granite-docling-258M)
- [IBM Granite-Docling Announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)
- [Docling Project](https://github.com/docling-project/docling)

## License

Apache 2.0 (model and code)
