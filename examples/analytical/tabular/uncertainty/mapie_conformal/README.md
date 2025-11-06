# MAPIE Conformal Prediction (California Housing)

This example demonstrates conformal prediction intervals for tabular regression
using [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) on the California
Housing dataset. We pretrain a `HistGradientBoostingRegressor`, calibrate it
with MAPIE's "plus" method, and export calibrated 90% prediction intervals for
median house values.

## Dataset

- **California Housing (UCI / StatLib)** – accessible through
  [`sklearn.datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).

## Quickstart

```bash
# Build container
cvl run mapie-conformal build

# Train and calibrate
cvl run mapie-conformal train

# Run inference on saved sample inputs
cvl run mapie-conformal predict
```

The training script downloads the dataset via scikit-learn, fits the base
regressor, calibrates MAPIE on a hold-out split, and saves:

- `artifacts/model/` – serialized base estimator (`base_estimator.pkl`), MAPIE
  wrapper (`mapie_regressor.pkl`), and calibration config.
- `artifacts/metrics.json` – MAE, R^2, coverage, and interval width metrics.
- `artifacts/sample_input.csv` & `artifacts/predictions.csv` – sample rows with
  corresponding prediction intervals.

## References

- MAPIE: *Model Agnostic Prediction Interval Estimator* –
  https://github.com/scikit-learn-contrib/MAPIE
- California Housing dataset description –
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
