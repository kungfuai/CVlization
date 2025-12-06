# CheckboxQA Benchmark

Evaluate vision-language models on checkbox and form understanding in PDF documents.

## Why CheckboxQA?

Modern VLMs struggle with checkboxes, radio buttons, and form elements - the "checkbox blind spot". This benchmark tests that capability.

**Dataset**: [mturski/CheckboxQA](https://huggingface.co/datasets/mturski/CheckboxQA) | **Paper**: [arXiv:2504.10419](https://huggingface.co/papers/2504.10419)

## Quick Start

```bash
# Build a model (one-time)
cvl run qwen3_vl build

# Run evaluation
cvl run checkbox_qa qwen3-2b --max-pages 2 --max-image-size 1800 --track
```

Results are saved to `results/` and optionally tracked via [trackio](https://github.com/huggingface/trackio).

## Available Models

| Model | Preset | Recommended Command | ANLS* |
|-------|--------|---------------------|-------|
| Phi-4 (14B) | `phi4` | `cvl run checkbox_qa phi4 --max-pages 1 --max-image-size 800` | 0.34 |
| Qwen3-VL-4B | `qwen3-4b` | `cvl run checkbox_qa qwen3-4b --max-pages 3 --max-image-size 1800` | 0.31 |
| Qwen3-VL-2B | `qwen3-2b` | `cvl run checkbox_qa qwen3-2b --max-pages 2 --max-image-size 1800` | 0.28 |
| Moondream2 | `moondream2` | `cvl run checkbox_qa moondream2 --max-pages 2` | 0.27 |

Scores on `subset_test.jsonl` (20 docs, 117 questions). Checkbox understanding remains challenging for current VLMs.

## Configuration Options

| Flag | Description |
|------|-------------|
| `--max-pages <n>` | Pages per document |
| `--max-image-size <px>` | Max image dimension (resize if larger) |
| `--track` | Enable experiment tracking |
| `--project <name>` | Trackio project name |

## Dataset Subsets

| Subset | Docs | Questions | Use case |
|--------|------|-----------|----------|
| `subset_dev.jsonl` | 5 | 24 | Quick iteration |
| `subset_test.jsonl` | 20 | 117 | Evaluation |
| `gold.jsonl` | 88 | 579 | Full benchmark |

Data auto-downloads to `~/.cache/cvlization/data/checkbox_qa/`.

## Metric: ANLS*

Average Normalized Levenshtein Similarity - measures text similarity between predictions and ground truth, accounting for typos. Range: 0.0 to 1.0.

## Viewing Results

```bash
# View trackio dashboard
trackio show --project checkbox-qa

# Check result files
cat results/<run>/eval_results.json
```

## Citation

```bibtex
@article{turski2025checkboxqa,
  title={Unchecked and Overlooked: Addressing the Checkbox Blind Spot in Large Language Models with CheckboxQA},
  author={Turski, Michał and Chiliński, Mateusz and Borchmann, Łukasz},
  year={2025}
}
```

**License**: CC BY-NC 4.0 (non-commercial research only)
