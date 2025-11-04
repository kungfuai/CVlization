# GBT Housing Price Regression (Ames Housing)

This example uses the [gbt](https://github.com/zzsi/gbt) LightGBM wrapper to
predict housing sale prices on the Ames Housing dataset. We train a gradient
boosting model, apply an isotonic calibration layer for better pricing
accuracy, and export scenario analyses that highlight how upgrades (e.g.,
additional living space) translate into adjusted valuations.

## Dataset

- **Ames Housing (Kaggle)** – accessed via the cleaned CSV published in the
  PyCaret repository:
  https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/house.csv

## Quickstart

```bash
# Build the container image
cvl run analytical-tabular-regression-gbt-housing-prices build

# Train the calibrated LightGBM regression model
cvl run analytical-tabular-regression-gbt-housing-prices train

# Score new listings (defaults to saved sample inputs)
cvl run analytical-tabular-regression-gbt-housing-prices predict
```

The training script will download the dataset, split train/calibration/test
sets, fit the gbt regression pipeline, perform isotonic calibration, and save:

- `artifacts/model/` – serialized gbt model, isotonic calibrator, config JSON.
- `artifacts/metrics.json` – RMSE (raw + calibrated), MAE, and R² on the test
  split.
- `artifacts/feature_importance.csv` – LightGBM gain-based feature importances.
- `artifacts/scenario_analysis.csv` – What-if pricing scenarios (quality
  upgrade, added living space, etc.).
- `artifacts/sample_input.csv` / `artifacts/predictions.csv` – sample listings
  and calibrated predictions for quick inference checks.

All commands run on CPU; no CUDA dependencies are required.
