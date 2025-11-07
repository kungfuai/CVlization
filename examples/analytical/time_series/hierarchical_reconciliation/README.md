# Hierarchical Time Series Forecasting with Reconciliation

This example demonstrates **hierarchical forecasting with reconciliation** using the [Australian Tourism dataset](https://robjhyndman.com/publications/hierarchical-tourism/). It generates base forecasts using classical statistical methods (AutoETS, AutoARIMA) and then applies reconciliation algorithms to ensure **coherence** across the hierarchy levels.

## What is Hierarchical Forecasting?

Hierarchical time series have a natural aggregation structure. For example:

```
Total Tourism
├── State: NSW
│   ├── Purpose: Business
│   └── Purpose: Holiday
├── State: VIC
│   ├── Purpose: Business
│   └── Purpose: Holiday
...
```

**Problem**: Base forecasts (generated independently for each series) typically don't "add up" - the sum of state-level forecasts ≠ total forecast.

**Solution**: Reconciliation methods adjust forecasts to ensure coherence while maintaining accuracy.

## Methods Implemented

### Base Forecasting
- **AutoETS**: Automatic Exponential Smoothing State Space model selection
- **AutoARIMA**: Automatic ARIMA model selection with seasonal components

### Reconciliation Algorithms
- **BottomUp**: Aggregate forecasts from the bottom level upward
- **MinTrace (OLS)**: Minimize forecast variance using ordinary least squares
- **MinTrace (Shrinkage)**: Robust MinTrace with shrinkage covariance estimation
- **MinTrace (WLS)**: Weighted least squares variant

## Dataset

**Australian Tourism** (monthly frequency):
- **TourismSmall**: 366 time series
- **TourismLarge**: 555 time series
- **Hierarchy**: Geographic regions (States) × Travel purposes (Business, Holiday, etc.)
- **Source**: [Monash Time Series Forecasting Repository](https://forecastingdata.org/)

## Quick Start

### Using CVL CLI

```bash
# Build the container
cvl run analytical/time_series/hierarchical_reconciliation build

# Run with default settings (12-month horizon, TourismSmall)
cvl run analytical/time_series/hierarchical_reconciliation forecast

# Custom configuration
cvl run analytical/time_series/hierarchical_reconciliation forecast -- \
  --dataset TourismLarge \
  --horizon 24 \
  --reconcilers BottomUp MinTrace_ols MinTrace_shrink
```

### Using Shell Scripts

```bash
# Build
./build.sh

# Forecast
./forecast.sh --horizon 12 --dataset TourismSmall
```

## Command-Line Options

```
--dataset              Dataset to use (TourismSmall or TourismLarge)
                       Default: TourismSmall

--horizon              Forecast horizon in periods
                       Default: 12

--season-length        Seasonal period (12 for monthly data)
                       Default: 12

--base-models          Base forecasting models (AutoETS, AutoARIMA)
                       Default: AutoETS AutoARIMA

--reconcilers          Reconciliation methods to apply
                       Options: BottomUp, MinTrace_ols, MinTrace_shrink, MinTrace_wls
                       Default: BottomUp MinTrace_ols MinTrace_shrink

--output-dir           Output directory for results
                       Default: ./artifacts

--data-dir             Directory to cache downloaded data
                       Default: ./data
```

## Output

Results are saved to `./artifacts/`:

1. **reconciled_forecasts.csv** - All base and reconciled forecasts
2. **metrics.csv** - Overall accuracy metrics (RMSE, MAE, MAPE) for each method
3. **metrics_by_level.json** - Accuracy broken down by hierarchy level
4. **config.json** - Configuration used for the run

## Example Output

```
================================================================================
OVERALL FORECAST ACCURACY (sorted by RMSE)
================================================================================
                          RMSE         MAE       MAPE
AutoETS/BottomUp       8234.12    4521.33      18.45
AutoETS/MinTrace_ols   8189.45    4498.76      18.21
AutoETS/MinTrace_shrink 8156.89   4476.12      18.03
AutoETS                8567.23    4689.54      19.34
AutoARIMA              8723.45    4798.21      19.87
================================================================================
```

## Key Insights

1. **Reconciliation improves accuracy**: Reconciled forecasts typically outperform base forecasts
2. **MinTrace variants shine**: MinTrace with shrinkage often provides the best balance
3. **BottomUp is simple**: Works well when bottom-level forecasts are most accurate
4. **Hierarchy matters**: Different levels may favor different reconciliation methods

## Why Hierarchical Reconciliation?

### Business Use Cases

1. **Retail Sales Forecasting**
   - Total → Region → Store → Department → Product
   - Ensures department forecasts add up to store totals

2. **Financial Planning**
   - Company → Division → Business Unit → Product Line
   - Coherent P&L forecasts across org structure

3. **Supply Chain**
   - National → Distribution Center → Warehouse → SKU
   - Inventory plans that respect capacity constraints

4. **Energy/Demand Forecasting**
   - Grid → Substation → Customer Segment
   - Load forecasts that balance across infrastructure

### Benefits

- **Coherence**: Forecasts respect natural aggregation constraints
- **Improved accuracy**: Pooling information across levels reduces errors
- **Better decisions**: Consistent forecasts across organizational hierarchy
- **Uncertainty quantification**: Reconciliation can preserve prediction intervals

## References

### Papers
- Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). [Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization](https://robjhyndman.com/publications/mint/). *Journal of the American Statistical Association*, 114(526), 804-819.
- Olivares, K. G., et al. (2024). [HierarchicalForecast: A Reference Framework for Hierarchical Forecasting in Python](https://arxiv.org/abs/2207.03517). arXiv preprint.

### Libraries
- [HierarchicalForecast](https://github.com/Nixtla/hierarchicalforecast) - Reconciliation methods
- [StatsForecast](https://github.com/Nixtla/statsforecast) - Base forecasting models
- [datasetsforecast](https://github.com/Nixtla/datasetsforecast) - Hierarchical datasets

### Books
- Hyndman, R. J., & Athanasopoulos, G. (2021). [Forecasting: Principles and Practice (3rd ed)](https://otexts.com/fpp3/). Chapter 11: Hierarchical forecasting.

## Next Steps

1. **Try different datasets**: Switch to TourismLarge or use custom hierarchical data
2. **Experiment with reconcilers**: Compare performance of different MinTrace variants
3. **Add more base models**: Include seasonal naive, Prophet, or foundation models
4. **Extend to grouped structures**: Adapt for non-strictly hierarchical organizations
5. **Production deployment**: Integrate with your business forecasting pipeline

## Related Examples

- [`uni2ts_finetune`](../uni2ts_finetune/) - Fine-tune foundation models for time series
- [`moirai_zero_shot`](../moirai_zero_shot/) - Zero-shot forecasting with Moirai
- [`merlion_anomaly_dashboard`](../merlion_anomaly_dashboard/) - Anomaly detection

---

**Note**: This example uses CPU-based classical statistical models. For deep learning approaches to hierarchical forecasting, consider combining foundation model base forecasts with reconciliation.
