# CheckboxQA Benchmark

Benchmark for evaluating vision-language models on checkbox and form element understanding in PDF documents.

## Overview

CheckboxQA is a specialized dataset that tests how well LLMs can interpret checkable content in visually-rich documents (PDFs). It addresses the "checkbox blind spot" - the observation that modern vision-language models often struggle with understanding filled checkboxes, radio buttons, and other form elements.

**Dataset**: [mturski/CheckboxQA](https://huggingface.co/datasets/mturski/CheckboxQA)
**Paper**: [Unchecked and Overlooked: Addressing the Checkbox Blind Spot in Large Language Models with CheckboxQA](https://huggingface.co/papers/2504.10419) (2025)
**License**: CC BY-NC 4.0 (non-commercial research only)

## Dataset Statistics

- **Documents**: 88 (test split only)
- **Questions**: 579 total (~6.6 questions per document)
- **Document Types**: Employment forms, tax forms, police reports, facility inspection forms
- **Answer Types**: Yes/No, categorical selections, lists of checked items

**Pre-created Subsets:**
- `subset_dev.jsonl` - 5 docs, 40 questions (for quick development/testing)
- `subset_test.jsonl` - 20 random docs, 138 questions (for evaluation)
- `gold.jsonl` - Full 88 docs, 579 questions (complete benchmark)

## Quick Start

### 1. Dataset is Pre-downloaded

The PDFs and gold annotations are already in `data/`:
- `data/documents/` - 88 PDF documents
- `data/gold.jsonl` - Full test set annotations (579 questions)
- `data/subset_dev.jsonl` - Dev set (5 docs, 40 questions)
- `data/subset_test.jsonl` - Test set (20 docs, 138 questions)
- `data/page_images/` - Cached PNG pages (auto-created on demand; override via `CHECKBOX_QA_PAGE_CACHE`)

### 2. Run Evaluation

```bash
# Quick test with dev subset (5 docs, ~4 minutes with moondream2)
./run_benchmark.sh moondream2 --subset data/subset_dev.jsonl

# Full evaluation on test subset (20 docs)
./run_benchmark.sh moondream2 --subset data/subset_test.jsonl

# Evaluate on full dataset (88 docs, ~10 hours)
./run_benchmark.sh moondream2

# Multiple models
./run_benchmark.sh florence_2 qwen3_vl_2b --subset data/subset_dev.jsonl
```

> Page rendering: `run_checkbox_qa.py` automatically caches PDF pages to `data/page_images/` before model inference. Set `CHECKBOX_QA_PAGE_CACHE=/custom/path` to change the cache location.

### 3. View Results

```bash
# Results are timestamped and saved to results/YYYYMMDD_HHMMSS/
# View evaluation metrics
cat results/20251112_182416/moondream2/eval_results.json

# View leaderboard
cat results/20251112_182416/leaderboard.json

# View detailed results
cat results/20251112_143022/scores.csv
```

### 4. Optional Docker Workflow

Build the lightweight orchestration image if you prefer running downloads/evaluation inside a container:

```bash
cd benchmarks/doc_ai/checkbox_qa
docker build -t checkbox_qa .
# Download PDFs
./download_data.sh

# Evaluate predictions (example)
./run_eval.sh results/20250101_120000/moondream2/predictions.jsonl
```

All benchmark scripts (e.g., `run_checkbox_qa.py`, `evaluate.py`) can be invoked the same way from the mounted `/workspace`.

### 5. Optional: reuse VLM servers (vLLM)

If you start a persistent vLLM server for a model, the adapters will call it instead of launching a container per question. A helper script automates the workflow:

```bash
# Qwen3-VL multipage run with ephemeral vLLM server
./run_with_vllm.sh qwen3 qwen3_vl_2b_multipage --subset data/subset_dev.jsonl

# Phi-4 multipage run
./run_with_vllm.sh phi4 phi_4_multimodal_multipage --subset data/subset_dev.jsonl
```

Those commands spin up the serve container, run `run_checkbox_qa.py`, then shut the server down automatically. To run servers manually, use the example `serve` presets (`cvl run qwen3_vl serve`, `cvl run phi-4-multimodal-instruct serve`) and export `QWEN3_VL_API_BASE` / `PHI4_API_BASE` before running the benchmark.

## Evaluation Metric

**ANLS* (Average Normalized Levenshtein Similarity)**

- Measures similarity between predicted and ground truth answers
- Accounts for typos and alternative phrasings
- Range: 0.0 (no match) to 1.0 (perfect match)
- Threshold: 0.5 (answers with similarity < 0.5 score 0)

## Directory Structure

```
benchmarks/doc_ai/checkbox_qa/
├── README.md                # This file
├── dataset_builder.py       # Loads CheckboxQA from HuggingFace
├── evaluate.py              # Runs evaluation and computes ANLS*
├── run_benchmark.sh         # Orchestrates multi-model runs
├── config.yaml              # Models and dataset configuration
├── adapters/               # Model-specific adapters
│   ├── florence_2.sh
│   ├── qwen3_vl_2b.sh
│   └── phi_4_multimodal.sh
├── data/                   # Dataset cache (optional)
│   ├── gold.jsonl         # Ground truth Q&A
│   ├── documents/         # Downloaded PDFs
│   ├── page_images/       # Cached PNG pages (auto-generated before inference)
│   └── document_url_map.json
└── results/               # Evaluation results
    ├── latest -> 20251112_143022/
    └── 20251112_143022/
        ├── leaderboard.md
        ├── scores.csv
        └── {model_name}/
            ├── predictions.jsonl
            └── errors.log
```

## Adding New Models

1. Create adapter script in `adapters/{model_name}.sh`
2. Add model to `config.yaml`
3. Run benchmark

Example adapter (see `adapters/florence_2.sh`):
```bash
#!/bin/bash
# Takes: <pdf_path> <question> --output <output_file>
# Returns: Answer text to output file
```

## Results Format

### predictions.jsonl
```json
{
  "name": "document_id",
  "annotations": [
    {
      "id": 1,
      "key": "Is checkbox X marked?",
      "values": [{"value": "Yes"}]
    }
  ]
}
```

### scores.csv
```csv
model,anls_score,num_questions,num_correct
florence_2,0.7234,579,419
qwen3_vl_2b,0.8125,579,471
phi_4_multimodal,0.8456,579,490
```

## Citation

```bibtex
@article{turski2025checkboxqa,
  title={Unchecked and Overlooked: Addressing the Checkbox Blind Spot in Large Language Models with CheckboxQA},
  author={Turski, Michał and Chiliński, Mateusz and Borchmann, Łukasz},
  journal={arXiv preprint arXiv:2504.10419},
  year={2025}
}
```

## License

This benchmark uses the CheckboxQA dataset which is provided under CC BY-NC 4.0 license for non-commercial research purposes only.
