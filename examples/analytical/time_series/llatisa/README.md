# LLaTiSA: VLM Time-Series Reasoning

VLM-based time series reasoning using dual-view input following the
[LLaTiSA](https://arxiv.org/abs/2604.17295) approach (ACL 2026 Findings).

## What to Expect

- **First-run cost**: ~15 GB model download (Qwen2.5-VL-7B-Instruct),
  cached in `~/.cache/huggingface/` afterward
- **What it does**: generates a line plot and numerical table from a time
  series, then queries a VLM for chain-of-thought reasoning
- **Where output goes**: saves to `artifacts/` in the example directory
  (or host cwd when using `cvl run`)
- **Output format**: `plot.png` (line chart), `numeric_table.png`
  (index-value grid), `result.json` (question + VLM response)
- **Runtime**: ~3 min on an A100 (model loading dominates); images-only
  mode completes in seconds on CPU

## Sample

**Input** -- synthetic series (200 steps, trend + seasonality + anomaly spike):

![Sample plot](https://huggingface.co/datasets/zzsi/cvl/resolve/main/llatisa/plot.png)

**Dual-view numerical table**:

![Numeric table](https://huggingface.co/datasets/zzsi/cvl/resolve/main/llatisa/numeric_table.png)

**Output** -- VLM reasoning (Qwen2.5-VL-7B-Instruct, `l2_trend` preset):

```
The time series exhibits a complex pattern rather than a simple increase,
decrease, or stationarity. It shows an upward trend from index 0 to ~25,
a decline to ~40, a second rise to ~60-70 with a sharp peak (the highest
point in the series), followed by a decline and then fluctuating trends.
Multiple peaks and troughs indicate a complex pattern with a significant
peak at around index 70.
```

## Why LLaTiSA?

Existing time series analysis methods treat numerical data in isolation.
LLaTiSA bridges qualitative visual perception with quantitative numerical
precision by feeding a Vision Language Model **two views** of the same
series: a line plot and a precision-calibrated numerical table rendered as
an image. This dual-view input enables chain-of-thought reasoning over
temporal patterns at multiple cognitive levels.

## What is LLaTiSA?

LLaTiSA (Language-guided Time-Series Analysis) combines:

1. **Line plot** -- captures visual patterns (trend, seasonality, anomalies)
2. **Numerical table image** -- provides exact index-value pairs for
   quantitative precision
3. **VLM backbone** (Qwen2.5-VL) -- performs chain-of-thought reasoning
   over both views

The original paper defines four reasoning levels:

| Level | Task | Description |
|-------|------|-------------|
| L1 | Numerical Read-out | Point-level retrieval, min/max, indexing |
| L2 | Pattern Perception | Trend, seasonality, change-point detection |
| L3 | Semantic Reasoning | Domain-specific interpretation |
| L4 | Predictive Inference | Forecasting |

## Quick Start

```bash
# Build
bash build.sh

# Run with synthetic data (images + VLM reasoning)
bash predict.sh

# Generate images only (no GPU required for this step)
bash predict.sh --images-only

# Ask a specific question
bash predict.sh --question "What is the period of the seasonal component?"

# Use a preset question type
bash predict.sh --question-preset l1_minmax

# Use your own data
bash predict.sh --input /path/to/series.json
bash predict.sh --input /path/to/data.csv --csv-column temperature
```

## Input Formats

**JSON** -- either a plain array or a dict with a `"timeseries"` key:

```json
[1.0, 2.3, 1.5, 3.7, 2.1]
```

```json
{"timeseries": [[1.0, 2.0], [1.5, 2.5], [1.2, 2.8]]}
```

**CSV** -- any CSV file; specify the column with `--csv-column`:

```bash
bash predict.sh --input data.csv --csv-column 1        # by index
bash predict.sh --input data.csv --csv-column temperature  # by name
```

If no input is provided, a synthetic series with trend, seasonality, and
an anomaly spike is generated.

## Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(synthetic)* | Path to JSON or CSV time series file |
| `--csv-column` | `1` | Column index or name for CSV input |
| `--model` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID or local path |
| `--question` | *(preset)* | Custom question to ask about the series |
| `--question-preset` | `l2_trend` | Preset: `l1_minmax`, `l1_read`, `l2_trend`, `l3_anomaly` |
| `--max-new-tokens` | `1024` | Maximum tokens in VLM response |
| `--output-dir` | `./artifacts` | Directory for output images and results |
| `--device` | `auto` | Device map: `auto`, `cuda`, `cpu` |
| `--series-length` | `200` | Length of synthetic series |
| `--seed` | `42` | Random seed for synthetic data |
| `--images-only` | `false` | Generate images without running VLM |

## Output

Results are saved to `artifacts/`:

- `plot.png` -- line plot visualization
- `numeric_table.png` -- index-value table image
- `result.json` -- question, response, and series metadata

## GPU Memory

The default model (Qwen2.5-VL-7B-Instruct) requires approximately 16 GB
of GPU memory with automatic mixed precision. On a 24 GB GPU (A10/A5000),
this runs comfortably. For lower-memory GPUs, consider using a smaller
model variant or adding `--device cpu` (slow but functional).

## Model Options

This example defaults to **Qwen2.5-VL-7B-Instruct** as the VLM backbone.
When LLaTiSA fine-tuned checkpoints become available, point `--model` to
the checkpoint path for improved time-series reasoning performance.

Other compatible models:

```bash
# Smaller variant (less GPU memory)
bash predict.sh --model Qwen/Qwen2.5-VL-2B-Instruct

# Larger variant (better reasoning)
bash predict.sh --model Qwen/Qwen2.5-VL-72B-Instruct
```

## References

- Ding et al., "LLaTiSA: Towards Difficulty-Stratified Time Series
  Reasoning from Visual Perception to Semantics", ACL 2026 Findings.
  [arXiv:2604.17295](https://arxiv.org/abs/2604.17295)
- [LLaTiSA source code](https://github.com/RainingNovember/LLaTiSA)
- [HiTSR dataset](https://huggingface.co/datasets/November-Rain/HiTSR)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

## Related Examples

- `chronos_zero_shot` -- zero-shot time series forecasting
- `moirai_zero_shot` -- foundation model zero-shot forecasting
- `patchtst_supervised` -- supervised patch-based transformer training
