# CVlization Benchmarks

Standardized benchmarks for evaluating AI models across different tasks and datasets.

## Overview

This directory contains benchmark harnesses for comparing multiple models on public datasets. Unlike the `examples/` directory which contains individual model implementations, `benchmarks/` focuses on comparative evaluation.

## Structure

```
benchmarks/
├── doc_ai/                    # Document AI benchmarks
│   ├── checkbox_qa/          # CheckboxQA: Form element understanding
│   └── receipt_extraction/   # Receipt/invoice OCR comparison
│
└── vision_language/          # (Future) VLM benchmarks
    ├── vqa_v2/
    ├── docvqa/
    └── ...
```

## Available Benchmarks

### Document AI

| Benchmark | Dataset | Task | Metric | Models |
|-----------|---------|------|--------|--------|
| [checkbox_qa](doc_ai/checkbox_qa/) | CheckboxQA (88 docs, 579 Q) | Form understanding, checkbox detection | ANLS* | VLMs |
| [receipt_extraction](doc_ai/receipt_extraction/) | Custom receipts | OCR extraction | Visual inspection | OCR models, VLMs |

## Quick Start

```bash
cd benchmarks/doc_ai/checkbox_qa
./run_benchmark.sh florence_2 qwen3_vl_2b
cat results/latest/leaderboard.md
```

## Benchmark Philosophy

**Benchmarks vs Examples:**
- **Examples** (`examples/`): Individual model implementations (inference/training)
- **Benchmarks** (`benchmarks/`): Comparative evaluation harnesses

**Design Principles:**
1. **Standardized**: Use public datasets with established metrics
2. **Reproducible**: Version-controlled data and evaluation code
3. **Modular**: Adapters normalize model interfaces
4. **Comparative**: Generate leaderboards across multiple models

## Adding a New Benchmark

1. Create directory: `benchmarks/{task}/{dataset_name}/`
2. Add core files:
   - `README.md` - Dataset info, usage, results
   - `dataset_builder.py` - Load dataset
   - `evaluate.py` - Compute metrics
   - `config.yaml` - Configure models and settings
   - `adapters/` - Model-specific adapters
3. Document in this README

## Adding a Model to Existing Benchmark

1. Create adapter script in `adapters/{model_name}.sh`
2. Add model to `config.yaml`
3. Run benchmark: `./run_benchmark.sh {model_name}`

## Benchmark Results

Results are timestamped and saved in each benchmark's `results/` directory:

```
results/
├── latest -> 20251112_143022/  # Symlink to latest run
└── 20251112_143022/
    ├── leaderboard.md          # Summary table
    ├── scores.csv              # Detailed scores
    └── {model_name}/
        ├── predictions.jsonl   # Model predictions
        └── errors.log          # Error logs
```

## License

Benchmarks use various datasets with different licenses:
- CheckboxQA: CC BY-NC 4.0 (non-commercial research only)
- Custom datasets: See individual benchmark READMEs

Always check dataset licenses before commercial use.

## Contributing

When adding benchmarks:
1. Use established public datasets when possible
2. Implement standard metrics (ANLS, F1, accuracy, etc.)
3. Document evaluation protocol clearly
4. Provide example adapter for at least one model
5. Include requirements.txt for Python dependencies
