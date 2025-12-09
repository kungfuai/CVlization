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
cvl run checkbox_qa qwen3-4b --max-pages 5 --max-image-size 1800 --track
```

Results are saved to `results/` and optionally tracked via [trackio](https://github.com/huggingface/trackio).

## Available Models

| Model | Config | ANLS* (dev) | Notes |
|-------|--------|-------------|-------|
| **Qwen3-VL-4B** | 5-6p, 1800px | **0.44** | Best overall |
| Llama 3.2 Vision 11B | 1p, 1200-1800px | 0.37-0.40 | Strong with short prompts |
| Phi-4 (14B) | 1p, 1200px | 0.38 | Best single-page |
| Qwen3-VL-2B | 2p, 2000px | 0.36 | Good efficiency |
| OlmOCR-2 (7B) | 1p | 0.31 | OCR model with VQA capability |
| InternVL3-8B | 6-8 tiles | 0.27 | Tile count doesn't help much |
| MiniCPM-V | 1600px | 0.26 | Needs larger images |
| Moondream2 | 2p, 756px | 0.27 | Lightweight |
| Pixtral 12B (4-bit) | any | 0.00 | Broken - quantization damage |
| DeepSeek-OCR | any | 0.00 | OCR-only model, no VQA support |

Scores on `subset_dev.jsonl` (5 docs, 24 questions). Checkbox understanding remains challenging.

**Note on OCR-only models**: Models like DeepSeek-OCR use `model.infer()` APIs designed for text extraction, not VQA. They ignore instruction prompts and produce verbose descriptions instead of concise answers. The API lacks `max_new_tokens` or output format control.

## Key Learnings

- **More pages help** up to 5-6 pages, then diminishing returns
- **Image size matters**: 1800px sweet spot for most models
- **4-bit quantization can break VQA**: Pixtral 12B Unsloth 4-bit produces garbage
- **Simple prompts work best**: Just the question, no elaborate instructions
- **Tile/resolution tuning**: Minimal gains for InternVL3, significant for MiniCPM

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
