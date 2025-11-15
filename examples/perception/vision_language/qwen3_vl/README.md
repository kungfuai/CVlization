# Qwen3-VL (2B / 4B / 8B)

Unified runner for Alibaba Cloud’s Qwen3-VL Instruct family. Choose the 2B, 4B or 8B checkpoint at runtime without rebuilding different images.

## Highlights

- **Models**: `Qwen/Qwen3-VL-2B/4B/8B-Instruct`
- **Tasks**: OCR, image captioning, VQA, visual reasoning, multi-image analysis
- **VRAM guide**: 2B (~4 GB), 4B (~8 GB), 8B (~16 GB)
- **CLI friendly**: works with `cvl run qwen3_vl build|predict` presets

## Quick Start

```bash
cd examples/perception/vision_language/qwen3_vl

# Build the shared Docker image (installs latest transformers from GitHub)
bash build.sh
# or: cvl run qwen3_vl build

# Caption with the 2B model
bash predict.sh --variant 2b --image test_images/sample.jpg --task caption

# OCR two pages with the 4B model
bash predict.sh --variant 4b --images test_images/page1.png test_images/page2.png --task ocr

# VQA with custom prompt (8B)
bash predict.sh --variant 8b --image test_images/sample.jpg --task vqa --prompt "What is checked?"
```

Environment variable shortcut:

```bash
QWEN3_VL_VARIANT=4b bash predict.sh --image ...     # default variant if not passed via CLI
```

## Batch Processing

Process multiple images with different prompts from a JSONL file:

```bash
python3 batch_predict.py --batch-input requests.jsonl --output-dir results/ --variant 2b
```

**JSONL input format** (one JSON object per line):
```jsonl
{"images": ["doc1.jpg"], "prompt": "Extract all text from this document"}
{"images": ["page1.jpg", "page2.jpg"], "prompt": "Summarize these pages", "output": "summary.txt"}
{"images": ["chart.png"], "prompt": "What trends are visible?", "id": "chart_001"}
```

**Required fields:**
- `images`: List of image paths
- `prompt`: Question or instruction

**Optional fields:**
- `output`: Custom output filename (default: `request_N.txt`)
- `id`: Request identifier
- `format`: `"txt"` or `"json"`

**Example:**
```bash
# Create sample batch input
cat > batch_requests.jsonl << 'EOF'
{"images": ["test_images/sample.jpg"], "prompt": "Describe this image"}
{"images": ["test_images/sample.jpg"], "prompt": "Extract any visible text"}
EOF

# Run batch processing (model loads once, processes all requests)
python3 batch_predict.py --batch-input batch_requests.jsonl --output-dir outputs/batch/ --variant 2b

# View results
cat outputs/batch/request_1.txt
```

## CVL CLI Presets

With the updated `example.yaml`, you can use the CVL CLI instead of bash scripts:

```bash
# Build
cvl run qwen3_vl build

# Predict (pass through args exactly as you would to predict.sh)
cvl run qwen3_vl predict --variant 2b --image test_images/sample.jpg

# Smoke test (defaults to 2B, pass another variant as arg)
cvl run qwen3_vl test 4b

# Serve (vLLM OpenAI-compatible endpoint, default variant=2b)
cvl run qwen3_vl serve
```

## Script Arguments

```
python predict.py \
  --variant {2b,4b,8b}        # choose predefined model (default: 2b)
  [--model-id HF_ID]          # override exact checkpoint
  [--image path|url]          # single image
  [--images p1 p2 ...]        # multi-image (PDF pages, video frames)
  [--task caption|ocr|vqa]
  [--prompt "question"]       # required for VQA unless custom prompt set
  [--output outputs/result.txt]
  [--format txt|json]
  [--max-new-tokens N]        # override variant default (128 for 2B, 512 otherwise)
```

## Notes

- The Dockerfile installs latest `transformers` from GitHub for Qwen3-VL support plus `accelerate`, `requests`, and `Pillow`.
- A vLLM-based `serve.sh` preset is included. Set `QWEN3_VL_VARIANT`, `PORT`, `TENSOR_PARALLEL_SIZE`, etc., then run `bash serve.sh` (or `cvl run qwen3_vl serve`) to expose an OpenAI-compatible endpoint for reuse across queries.
- HuggingFace cache is mounted via `predict.sh` to avoid repeated downloads.
- `test.sh` runs a quick captioning smoke test; pass a variant (e.g., `bash test.sh 4b`).

## License

Models are Apache 2.0 (see individual HuggingFace cards). This example script is MIT, consistent with the rest of CVlization. Refer to Alibaba Cloud’s usage terms for any production deployment.
