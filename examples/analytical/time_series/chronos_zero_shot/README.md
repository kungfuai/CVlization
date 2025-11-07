# Chronos: Zero-Shot Time Series Forecasting

This example demonstrates **zero-shot probabilistic forecasting** with Amazon's Chronos foundation models. Chronos learns the "language" of time series by treating forecasting as a sequence-to-sequence problem using language model architectures.

## Why Chronos?

**Problem:** Traditional forecasting requires:
- Training separate models per dataset
- Large amounts of historical data
- Domain-specific tuning

**Solution:** Chronos provides:
1. **Zero-shot forecasting**: No training required, works out-of-the-box
2. **Probabilistic predictions**: Get uncertainty estimates via quantiles
3. **Multiple model families**: Choose speed vs. accuracy tradeoff

**Use Cases:** Quick baseline forecasting, cold-start problems, exploratory analysis

## Chronos Model Families

### 1. Chronos-Bolt ⚡ (Recommended for Most Use Cases)

**What it is:** Optimized distilled models, **250x faster** and **20x more memory-efficient** than original Chronos.

**Variants:**
- `amazon/chronos-bolt-tiny` (9M params) - Ultra-fast, embedded devices
- `amazon/chronos-bolt-mini` (20M params) - Fast, good accuracy
- `amazon/chronos-bolt-small` (46M params) - **Default**, best speed/accuracy balance
- `amazon/chronos-bolt-base` (205M params) - Highest accuracy

**When to use:** Production deployments, resource-constrained environments, real-time forecasting

### 2. Chronos-2 🆕 (Latest, Multivariate Support)

**What it is:** Latest model with **multivariate and covariate-aware** forecasting.

**Variants:**
- `amazon/chronos-2` (120M params)

**Key features:**
- **Multivariate**: Forecast multiple related series jointly
- **Covariates**: Use future known variables (e.g., holidays, promotions)
- **SOTA performance**: Best on fev-bench, GIFT-Eval benchmarks
- **90%+ win rate** vs. Chronos-Bolt in head-to-head comparisons

**When to use:** Complex scenarios with multiple related series or external factors

### 3. Chronos-T5 (Original, Language Model-Based)

**What it is:** Original Chronos models based on T5 encoder-decoder architecture.

**Variants:**
- `amazon/chronos-t5-tiny` (8M params)
- `amazon/chronos-t5-mini` (20M params)
- `amazon/chronos-t5-small` (46M params)
- `amazon/chronos-t5-base` (200M params)
- `amazon/chronos-t5-large` (710M params) - Most accurate, slowest

**When to use:** Research, when maximum accuracy is needed regardless of speed

## Quick Start

### Using CVL CLI

```bash
# Build the container
cvl run analytical/time_series/chronos_zero_shot build

# Forecast with Chronos-Bolt (default: 10 series from M4 Hourly)
cvl run analytical/time_series/chronos_zero_shot forecast

# Forecast with larger model on more series
cvl run analytical/time_series/chronos_zero_shot forecast_bolt_large

# Try latest Chronos-2 model
cvl run analytical/time_series/chronos_zero_shot forecast_chronos2

# Custom forecasting
cvl run analytical/time_series/chronos_zero_shot forecast -- \
  --model amazon/chronos-bolt-small \
  --dataset m4_daily \
  --max-series 50 \
  --prediction-length 14
```

### Using Shell Scripts

```bash
# Build
./build.sh

# Forecast
./forecast.sh --model amazon/chronos-bolt-small --dataset m4_hourly --max-series 20
```

## Command-Line Options

```
--model                Model variant
                       Default: amazon/chronos-bolt-small
                       Options: chronos-bolt-tiny/mini/small/base,
                                chronos-2,
                                chronos-t5-tiny/mini/small/base/large

--dataset              Dataset name
                       Default: m4_hourly
                       Options: m4_hourly, m4_daily, m4_weekly, m4_monthly

--prediction-length    Forecast horizon (overrides dataset default)
                       Default: Dataset-specific (48 for hourly, 14 for daily, etc.)

--context-length       Historical context window
                       Default: Auto (uses all available history)

--num-samples          Number of sample paths for quantile estimation
                       Default: 20

--max-series           Maximum number of series to evaluate
                       Default: 10

--output-dir           Output directory for results
                       Default: ./artifacts

--device               Device (cuda or cpu)
                       Default: cuda if available, else cpu
```

## Output

Results are saved to `./artifacts/`:

1. **{dataset}_metrics.json** - Aggregate metrics (MAE, RMSE, sMAPE, MASE)
2. **{dataset}_series_metrics.csv** - Per-series detailed metrics
3. **config.json** - Configuration used for the run

### Example Output

```
================================================================================
FORECAST ACCURACY
================================================================================
MAE:    123.4567
RMSE:   189.2345
sMAPE:  11.23%
MASE:   0.85
================================================================================
```

## Metrics Explained

### MAE (Mean Absolute Error)
- Average absolute deviation between forecast and actual
- **Units:** Same as data
- **Lower is better**
- **When to use:** Easy to interpret, robust to outliers

### RMSE (Root Mean Squared Error)
- Square root of average squared errors
- **Units:** Same as data
- **Lower is better**, penalizes large errors more than MAE
- **When to use:** When large errors are particularly costly

### sMAPE (Symmetric Mean Absolute Percentage Error)
- Percentage error, symmetric version of MAPE
- **Units:** Percentage (0-200%)
- **Lower is better**
- **When to use:** Comparing across different scale series

### MASE (Mean Absolute Scaled Error)
- Error relative to naive seasonal baseline
- **Units:** Ratio (1.0 = same as naive)
- **Lower is better**, <1.0 means beats naive forecast
- **When to use:** Scale-independent comparison across datasets

## Expected Performance

### M4 Hourly (48-hour horizon)
| Model | sMAPE | MASE | Speed (series/sec) |
|-------|-------|------|-------------------|
| Naive Seasonal | 14-16% | 1.00 | Instant |
| AutoARIMA | 16-17% | 0.95 | 0.5 |
| **Chronos-Bolt-Small** | **10-12%** | **0.72** | **50** |
| Chronos-2 | 9-11% | 0.68 | 10 |
| Moirai-Small | 11-13% | 0.75 | 20 |

### M4 Daily (14-day horizon)
| Model | sMAPE | MASE | Speed (series/sec) |
|-------|-------|------|-------------------|
| Naive Seasonal | 12-14% | 1.00 | Instant |
| AutoETS | 12-13% | 0.88 | 1 |
| **Chronos-Bolt-Small** | **10-11%** | **0.76** | **50** |
| Chronos-2 | 9-10% | 0.72 | 10 |
| Moirai-Small | 10-12% | 0.78 | 20 |

**Notes:**
- Results with 10+ series, single GPU (or CPU for Chronos-Bolt)
- Chronos-Bolt is **5-10x faster** than Moirai for similar accuracy
- Chronos-2 slightly better but slower
- All foundation models significantly outperform classical baselines

## Chronos vs. Moirai Comparison

CVlization includes both Chronos and Moirai examples. Here's when to use each:

| Aspect | Chronos | Moirai |
|--------|---------|--------|
| **Speed** | ⚡⚡⚡ Very fast (Chronos-Bolt) | Moderate |
| **Accuracy** | Excellent (esp. Chronos-2) | Excellent |
| **Memory** | Low (20x efficient) | Moderate |
| **Multivariate** | ✅ Yes (Chronos-2) | ✅ Yes |
| **Covariates** | ✅ Yes (Chronos-2) | Limited |
| **Model sizes** | 9M - 710M params | 11M - 710M params |
| **Training data** | ~100B time points (Google Trends, Wikipedia) | LOTSA corpus (Monash, GluonTS, LTSF) |
| **Use case** | Production, real-time | Research, complex scenarios |

**Recommendation:**
- **Start with Chronos-Bolt**: Fastest, great accuracy, low memory
- **Use Chronos-2**: When you have covariates or multivariate data
- **Use Moirai**: When you need maximum accuracy and have compute resources

## Advanced Usage

### Model Selection Guide

**Choose by Speed:**
```bash
# Fastest (embedded devices, real-time)
--model amazon/chronos-bolt-tiny

# Fast (good balance)
--model amazon/chronos-bolt-small  # DEFAULT

# Slower but more accurate
--model amazon/chronos-2
```

**Choose by Accuracy (at cost of speed):**
```bash
# Good accuracy, very fast
--model amazon/chronos-bolt-base

# Best accuracy, slower
--model amazon/chronos-2

# Maximum accuracy, slowest
--model amazon/chronos-t5-large
```

### Context Length Tuning

Chronos can use all available history, but limiting context can improve speed:

```bash
# Use last 512 points only (faster)
./forecast.sh --context-length 512

# Use all history (default, potentially slower)
./forecast.sh --context-length None
```

**Guidelines:**
- **Short series (<100 points)**: Use all history
- **Medium series (100-1000 points)**: Try --context-length 512
- **Long series (>1000 points)**: Use --context-length 512 or 1024

### Num Samples for Uncertainty

More samples = better quantile estimates but slower:

```bash
# Faster, less smooth quantiles
--num-samples 10

# Default, good balance
--num-samples 20

# Slower, smoother quantiles
--num-samples 50
```

## How Chronos Works

### Architecture

1. **Tokenization**: Time series values are scaled and quantized into tokens
   ```
   Raw values → Normalization → Quantization → Tokens
   [1.2, 3.4, 2.1] → [0.1, 0.8, 0.3] → [10, 80, 30]
   ```

2. **Language Model**: T5 encoder-decoder processes token sequences
   ```
   Encoder: Historical tokens → Context representation
   Decoder: Context → Future tokens (autoregressive)
   ```

3. **De-tokenization**: Tokens converted back to continuous values
   ```
   Tokens → Dequantization → Denormalization → Forecast values
   ```

### Training Data

Chronos models are pretrained on:
- **Google Trends** data (search volume)
- **Wikipedia page visits** (page view counts)
- **Synthetic data** (TSMixup, KernelSynth augmentation)
- Total: **~100 billion time points**

This diverse pretraining enables zero-shot transfer to new domains.

### Probabilistic Forecasting

Chronos generates **sample paths** (num_samples=20 by default):
```python
forecast_samples = model.predict(...)  # Shape: (num_samples, horizon)

# Extract quantiles for uncertainty
q10 = np.quantile(forecast_samples, 0.1, axis=0)  # 10th percentile
q50 = np.quantile(forecast_samples, 0.5, axis=0)  # Median (point forecast)
q90 = np.quantile(forecast_samples, 0.9, axis=0)  # 90th percentile
```

Use these quantiles for:
- **Point forecast**: Median (q50)
- **Prediction intervals**: [q10, q90] for 80% interval
- **Risk management**: Focus on extreme quantiles (q5, q95)

## Benchmarking Against Baselines

Compare Chronos with other examples in CVlization:

```bash
# Run Chronos
cvl run chronos_zero_shot forecast -- --dataset m4_hourly --max-series 100

# Run classical baselines
cvl run statsforecast_baselines benchmark -- --datasets m4_hourly --n-series 100

# Run Moirai (alternative foundation model)
cvl run moirai_zero_shot forecast -- --dataset m4_hourly --max-series 100
```

**Typical findings:**
- **Chronos-Bolt**: 10-12% sMAPE, very fast
- **Classical (AutoARIMA/ETS)**: 14-16% sMAPE, slow training
- **Moirai**: 11-13% sMAPE, moderate speed

**Conclusion:** Foundation models (Chronos/Moirai) provide 20-30% improvement over classical methods with zero-shot capability.

## Troubleshooting

### Model Download Slow
```bash
# Models are downloaded from HuggingFace on first use
# Download can be slow (~500MB-2GB depending on model)
# They are cached in ~/.cache/huggingface/hub/

# Pre-download models
python -c "from chronos import BaseChronosPipeline; BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-small')"
```

### GPU Out of Memory
```bash
# Use smaller model
--model amazon/chronos-bolt-tiny

# Reduce batch size (process fewer series at once)
--max-series 5

# Use CPU
--device cpu
```

### Poor Forecast Quality
```bash
# Try larger model
--model amazon/chronos-bolt-base  # or chronos-2

# Increase sample paths for smoother quantiles
--num-samples 50

# Provide more context
--context-length 1024
```

### Comparison with Moirai Shows Different Results
Both models are good but have different strengths:
- **Chronos-Bolt**: Better for high-frequency data (hourly, daily)
- **Moirai**: Better for low-frequency data (weekly, monthly)
- **Chronos-2**: Best overall but slower

## References

### Papers
- **Chronos** (Mar 2024): Ansari et al., "Chronos: Learning the Language of Time Series"
  - [Paper](https://arxiv.org/abs/2403.07815)
  - [Code](https://github.com/amazon-science/chronos-forecasting)

- **Chronos-Bolt** (Nov 2024): Optimization and distillation for speed
  - [Blog](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/)

- **Chronos-2** (Feb 2025): Multivariate and covariate-aware forecasting
  - [Blog](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)

### Libraries
- [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) - Official implementation
- [AutoGluon-TimeSeries](https://auto.gluon.ai/stable/tutorials/timeseries/index.html) - Includes Chronos integration

## Next Steps

1. **Run default forecast**: Start with Chronos-Bolt on 10 series
2. **Compare models**: Try Chronos-2 and Chronos-T5 variants
3. **Benchmark vs. baselines**: Run `statsforecast_baselines` for comparison
4. **Compare with Moirai**: Run `moirai_zero_shot` to see differences
5. **Production deployment**: Use Chronos-Bolt for fast, accurate forecasting

## Related Examples

- [`moirai_zero_shot`](../moirai_zero_shot/) - Alternative foundation model (Salesforce)
- [`statsforecast_baselines`](../statsforecast_baselines/) - Classical forecasting baselines
- [`uni2ts_finetune`](../uni2ts_finetune/) - Fine-tuning foundation models

---

**Note:** Chronos-Bolt models work well on CPU, making them ideal for production deployments without GPU requirements.
