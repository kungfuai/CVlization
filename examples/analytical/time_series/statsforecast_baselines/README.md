# Statistical Forecasting Baselines Benchmark

This example provides a comprehensive benchmark of **classical statistical forecasting methods** using [StatsForecast](https://github.com/Nixtla/statsforecast). It compares AutoARIMA, AutoETS, Theta, and SeasonalNaive across multiple datasets to establish baseline performance.

## Why Baselines Matter

Before deploying deep learning or foundation models, you need to know:
- **Can classical methods solve your problem?** (They're faster and more interpretable)
- **How much improvement do complex models provide?**
- **Which baseline is best for your data characteristics?**

This example helps answer these questions with systematic benchmarking.

## Models Benchmarked

### AutoARIMA
- **What it does**: Automatically selects optimal ARIMA(p,d,q)(P,D,Q)m parameters
- **Best for**: Data with trends and seasonality
- **Speed**: Moderate (parameter search is slow)
- **Interpretable**: ✅ Yes (classic statistical model)

### AutoETS
- **What it does**: Automatically selects exponential smoothing state space model
- **Best for**: Data with smooth trends and seasonal patterns
- **Speed**: Fast
- **Interpretable**: ✅ Yes (weighted averages)

### Theta
- **What it does**: Decomposes series into trend and seasonal components
- **Best for**: M4-competition winner for monthly/quarterly data
- **Speed**: Very fast
- **Interpretable**: ✅ Yes (simple decomposition)

### SeasonalNaive
- **What it does**: Forecasts using value from same season in previous cycle
- **Best for**: Baseline sanity check
- **Speed**: Instant
- **Interpretable**: ✅ Yes (trivial)

## Datasets

### M4 Competition
- **M4 Hourly**: 414 series, 48-hour horizon
- **M4 Daily**: 4,227 series, 14-day horizon
- **Use case**: General forecasting benchmark

### LTSF Suite (Coming Soon)
- **ETT-h1**: Electricity transformer temperature (hourly)
- **Electricity**: 321 clients electricity consumption
- **Use case**: Long-term forecasting benchmark

## Quick Start

### Using CVL CLI

```bash
# Build the container
cvl run analytical/time_series/statsforecast_baselines build

# Run benchmark on M4 Hourly (10 series for quick test)
cvl run analytical/time_series/statsforecast_baselines benchmark

# Compare all models on 50 series
cvl run analytical/time_series/statsforecast_baselines compare

# Custom benchmark
cvl run analytical/time_series/statsforecast_baselines benchmark -- \
  --datasets m4_hourly m4_daily \
  --models AutoARIMA AutoETS Theta \
  --n-series 100
```

### Using Shell Scripts

```bash
# Build
./build.sh

# Benchmark
./benchmark.sh --datasets m4_hourly --n-series 20
```

## Command-Line Options

```
--datasets            Datasets to benchmark (m4_hourly, m4_daily, ett_h1, electricity)
                      Default: m4_hourly

--models              Models to compare (AutoARIMA, AutoETS, Theta, SeasonalNaive)
                      Default: All four

--horizon             Forecast horizon (uses dataset default if not specified)

--n-series            Number of series to sample (for faster testing)
                      Default: 10

--season-length       Seasonal period (uses dataset default if not specified)

--output-dir          Output directory for results
                      Default: ./artifacts

--data-dir            Directory to cache downloaded data
                      Default: /root/.cache/cvlization/statsforecast_data
```

## Output

Results are saved to `./artifacts/`:

1. **{dataset}_results.csv** - Per-dataset detailed results
2. **benchmark_summary.csv** - Combined results across all datasets
3. **config.json** - Configuration used for the run

### Example Output

```
================================================================================
FORECAST ACCURACY (sorted by RMSE)
================================================================================
              RMSE        MAE     sMAPE      MASE
Theta       234.56     145.23     12.34      0.89
AutoETS     245.67     152.34     13.21      0.92
AutoARIMA   267.89     165.43     14.56      1.01
SeasonalNaive 312.45   198.76     18.92      1.21
================================================================================

COMPARATIVE SUMMARY ACROSS ALL DATASETS
Best Model per Dataset (by RMSE):
m4_hourly            → Theta           (RMSE: 234.5634)
m4_daily             → AutoETS         (RMSE: 1245.2341)
```

## Metrics Explained

### RMSE (Root Mean Squared Error)
- **What it measures**: Average magnitude of errors, penalizing large errors
- **Units**: Same as the data
- **Lower is better**
- **When to use**: When large errors are particularly bad

### MAE (Mean Absolute Error)
- **What it measures**: Average absolute deviation
- **Units**: Same as the data
- **Lower is better**
- **When to use**: More interpretable than RMSE

### sMAPE (Symmetric Mean Absolute Percentage Error)
- **What it measures**: Percentage error (symmetric version of MAPE)
- **Units**: Percentage (0-200%)
- **Lower is better**
- **When to use**: Comparing across different scale series

### MASE (Mean Absolute Scaled Error)
- **What it measures**: Error relative to naive baseline
- **Units**: Ratio (1.0 = same as naive)
- **Lower is better**, <1.0 beats naive
- **When to use**: Scale-independent comparison

## Interpreting Results

### Typical Performance Ranges

| Dataset Type | Good sMAPE | Excellent sMAPE |
|--------------|------------|-----------------|
| Hourly data  | <15%       | <10%            |
| Daily data   | <12%       | <8%             |
| Monthly data | <10%       | <5%             |

### When Each Model Wins

- **AutoARIMA wins**: Strong trends, complex seasonality, longer series
- **AutoETS wins**: Smooth patterns, moderate seasonality, reliable data
- **Theta wins**: M4-style competition data, balanced forecasting
- **SeasonalNaive wins**: Very simple seasonal patterns (rarely)

## Comparing with Foundation Models

Use this benchmark to compare against:
- [`moirai_zero_shot`](../moirai_zero_shot/) - Salesforce Moirai foundation model
- [`uni2ts_finetune`](../uni2ts_finetune/) - Fine-tuned Uni2TS models

**Example comparison:**
```bash
# Run baseline benchmark
cvl run statsforecast_baselines benchmark -- --datasets m4_hourly --n-series 100

# Run foundation model
cvl run moirai_zero_shot forecast -- --dataset m4_hourly --max-series 100

# Compare sMAPE results
```

**Typical findings:**
- Classical baselines: sMAPE 10-20% (fast, interpretable)
- Foundation models: sMAPE 8-15% (slower, less interpretable)
- **Conclusion**: Foundation models provide 10-30% improvement at cost of complexity

## Performance Tips

### Faster Benchmarking
- Use `--n-series 10` for quick tests
- Use `--n-series 100+` for reliable comparisons
- Skip AutoARIMA if slow (it searches parameter space)

### Production Deployment
1. Run full benchmark on representative sample
2. Identify best model per dataset type
3. Deploy that model (likely AutoETS or Theta for speed)
4. Monitor performance and re-benchmark periodically

## References

### Papers
- **ARIMA**: Box, G. E., & Jenkins, G. M. (1970). *Time series analysis: Forecasting and control*.
- **ETS**: Hyndman, R. J., et al. (2008). *Forecasting with exponential smoothing*.
- **Theta**: Assimakopoulos, V., & Nikolopoulos, K. (2000). *The theta model*.
- **M4 Competition**: Makridakis, S., et al. (2020). *The M4 Competition*.

### Libraries
- [StatsForecast](https://github.com/Nixtla/statsforecast) - Fast statistical forecasting
- [datasetsforecast](https://github.com/Nixtla/datasetsforecast) - Time series datasets

### Books
- Hyndman, R. J., & Athanasopoulos, G. (2021). [Forecasting: Principles and Practice (3rd ed)](https://otexts.com/fpp3/).

## Next Steps

1. **Run your first benchmark**: Start with `--n-series 10` for speed
2. **Analyze results**: Which model works best for your data characteristics?
3. **Compare with foundation models**: Run `moirai_zero_shot` or `uni2ts_finetune`
4. **Deploy the winner**: Use the best-performing baseline or foundation model

## Related Examples

- [`hierarchical_reconciliation`](../hierarchical_reconciliation/) - Uses AutoETS/AutoARIMA with reconciliation
- [`moirai_zero_shot`](../moirai_zero_shot/) - Foundation model comparison
- [`uni2ts_finetune`](../uni2ts_finetune/) - Fine-tune foundation models

---

**Note**: This example focuses on univariate forecasting. For multivariate or with exogenous variables, consider deep learning approaches or AutoGluon-TimeSeries.
