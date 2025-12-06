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

| Model | Preset | Notes |
|-------|--------|-------|
| Qwen3-VL-2B | `qwen3-2b` | Good baseline, fast |
| Qwen3-VL-4B | `qwen3-4b` | Best results so far |
| Moondream2 | `moondream2-batch` | Tiny (1.9B), competitive |
| Phi-4 (14B) | `phi4` | Requires smaller images for L4 GPU |
| Florence-2 | `florence2-batch` | OCR-focused, not VQA |

## Configuration Options

```bash
cvl run checkbox_qa <preset> [options]
```

| Flag | Description |
|------|-------------|
| `--subset <path>` | Subset JSONL file (default: subset_dev.jsonl) |
| `--max-pages <n>` | Pages per document (default: 20) |
| `--max-image-size <px>` | Max image dimension (default: no resize) |
| `--track` | Enable experiment tracking |
| `--project <name>` | Trackio project name |

## Example Results

On `subset_test.jsonl` (20 docs, 117 questions):

| Model | Config | ANLS* |
|-------|--------|-------|
| Qwen3-VL-4B | 3p, 1800px | 0.31 |
| Qwen3-VL-4B | 2p, 1800px | 0.29 |
| Qwen3-VL-2B | 2p, 1800px | 0.28 |
| Moondream2 | 2p, 756px | 0.27 |

These are modest scores - checkbox understanding remains challenging for current VLMs.

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
