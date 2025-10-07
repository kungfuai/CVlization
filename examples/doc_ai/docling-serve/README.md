# Docling Document Extraction Example

Extract structured content from PDFs and images using IBM's [Docling](https://github.com/docling-project/docling) toolkit.

## Features

- **Multi-format support**: Process PDF and image files
- **Multiple output formats**: JSON (structured), Markdown, or plain text
- **Advanced extraction**: Tables, equations, code blocks, and document structure
- **Powered by Docling**: Uses DocLayNet (layout analysis) and TableFormer (table structure) models

## Quick Start

### 1. Build the Docker image

```bash
./build.sh
```

### 2. Extract content from a document

```bash
# Basic usage - outputs JSON to stdout
./predict.sh sample.pdf

# Save as Markdown
./predict.sh sample.pdf --format markdown --output result.md

# Extract with tables exported separately
./predict.sh document.pdf --format json --export-tables --output result.json

# Process an image
./predict.sh scan.jpg --format text
```

## Usage

```bash
./predict.sh <input_file> [options]
```

### Options

- `--output <file>`: Save output to file (prints to stdout if not specified)
- `--format <format>`: Output format
  - `json` (default): Structured JSON with full document hierarchy
  - `markdown`: Clean markdown format
  - `text`: Plain text extraction
- `--export-tables`: Export tables separately to `outputs/tables/`
- `--export-images`: Export image metadata to `outputs/images/`

### Examples

```bash
# Extract to JSON and view in terminal
./predict.sh report.pdf

# Save as markdown for RAG pipeline
./predict.sh research_paper.pdf --format markdown --output paper.md

# Extract complex document with tables
./predict.sh financial_report.pdf --format json --export-tables --output report.json

# Process scanned document
./predict.sh scanned_invoice.png --format json --output invoice.json
```

## Output Structure

### JSON Format
```json
{
  "input_file": "sample.pdf",
  "pages": 5,
  "content": {
    "text": "...",
    "tables": [...],
    "headings": [...],
    "metadata": {...}
  }
}
```

### Markdown Format
Preserves document structure with proper heading hierarchy, lists, and tables.

## What Docling Extracts

- **Text**: With proper paragraph and section boundaries
- **Tables**: Structure-aware extraction (rows, columns, headers)
- **Equations**: Inline and display math formulas
- **Code blocks**: Preserves formatting
- **Lists**: Ordered and unordered
- **Headings**: Document hierarchy
- **Layout**: Spatial relationships between elements

## Models Used

- **DocLayNet**: Layout analysis model
- **TableFormer**: Table structure recognition
- **docling-ibm-models**: Core extraction models

All models run efficiently on CPU (no GPU required).

## Performance

- **Accuracy**: 97.9% on complex table extraction (2025 benchmark)
- **Speed**: Depends on document complexity, typically seconds per page
- **Memory**: Optimized for commodity hardware

## Directory Structure

```
docling-serve/
├── Dockerfile          # Container definition
├── build.sh           # Build script
├── predict.sh         # Inference script
├── predict.py         # Python extraction logic
├── sample.pdf         # Example PDF for testing
├── outputs/           # Generated extraction results
└── README.md          # This file
```

## Sample PDF

The included `sample.pdf` is a form examples document from the Delaware Department of Agriculture, used for testing document extraction capabilities.

**Source**: [Delaware Department of Agriculture - Form Examples](https://agriculture.delaware.gov/wp-content/uploads/sites/108/2018/01/Examples.pdf)

## Next Steps

For more advanced use cases:

1. **Add Granite-Docling VLM**: Use `ibm-granite/granite-docling-258M` for end-to-end extraction with a single 258M parameter model
2. **Deploy as API**: Use `docling-serve` to run as a FastAPI service
3. **RAG Pipeline**: Pipe markdown output to embedding models
4. **OCR Preprocessing**: Add PaddleOCR for scanned document support

## References

- [Docling GitHub](https://github.com/docling-project/docling)
- [Docling Paper (AAAI 2025)](https://arxiv.org/html/2501.17887v1)
- [Granite-Docling Model](https://huggingface.co/ibm-granite/granite-docling-258M)
