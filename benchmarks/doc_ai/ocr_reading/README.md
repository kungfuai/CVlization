# OCR Reading Benchmark

Evaluate vision-language models on OCR text reading on SROIE receipt images.

**Dataset**: [arvindrajan92/sroie_document_understanding](https://huggingface.co/datasets/arvindrajan92/sroie_document_understanding) — 652 receipts with per-word bounding boxes
**Evaluation framework**: vendored from [GutenOCR/vllm-ocr-eval](https://github.com/pengwingokla/GutenOCR/tree/main/experiments/vllm-ocr-eval)

## Quick Start

```bash
# 1. Build the Docker image (one-time)
./adapters/qwen25_vl_7b.sh build

# 2. Run the full benchmark
./run_benchmark.sh --model qwen25_vl_7b

# Or step by step:
python dataset_builder.py --output-dir data/shards/
./adapters/qwen25_vl_7b.sh --shards data/shards/ --output results/predictions.csv
python evaluate.py --pred results/predictions.csv --output results/metrics.json
```

## Metrics

| Metric | Description |
|--------|-------------|
| CER | Character Error Rate (lower is better) |
| WER | Word Error Rate (lower is better) |
| ANLS | Average Normalized Levenshtein Similarity (higher is better) |
| BBox F1 | Bounding box precision/recall F1 (higher is better) |
| Matched Box Ratio | Fraction of ground truth boxes matched (higher is better) |

## Models

| Model | Adapter | VRAM |
|-------|---------|------|
| Qwen2.5-VL-7B-Instruct | `adapters/qwen25_vl_7b.sh` | ~16GB |

## Directory Structure

```
ocr_reading/
├── vendor/                  # Vendored from GutenOCR/vllm-ocr-eval
│   ├── run_evaluation.py    # Multi-GPU vLLM inference
│   ├── score_lines_reading.py  # CER/WER/ANLS/bbox scoring
│   ├── dataset.py           # TAR shard dataset loader
│   ├── prompt_builder.py    # Task/prompt generation
│   ├── predictor.py         # vLLM wrapper
│   └── utils/               # text, box, parse utilities
├── adapters/
│   └── qwen25_vl_7b.sh      # Docker inference + scoring
├── dataset_builder.py       # Download SROIE → TAR shards
├── evaluate.py              # Aggregate CSV → metrics JSON
└── run_benchmark.sh         # End-to-end orchestrator
```

## Attribution

The `vendor/` directory contains code from [GutenOCR](https://github.com/pengwingokla/GutenOCR) (`experiments/vllm-ocr-eval`), vendored to pin the exact version used in this benchmark.
