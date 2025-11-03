# GBT Upsell Propensity (Bank Marketing)

This example trains a gradient-boosted tree model (via the [`gbt`](https://pypi.org/project/gbt/) wrapper on top of LightGBM) to predict whether a customer will subscribe to a term deposit. It uses the **Bank Marketing** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing), specifically the "bank-additional" split.

## Data Sources
- Dataset: [Bank Marketing Data Set — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- `gbt` library: [https://github.com/zzsi/gbt](https://github.com/zzsi/gbt)

## Workflow
1. `build.sh` — builds a CPU-only Docker image with Python 3.11, LightGBM, and gbt 0.3 from PyPI.
2. `train.sh` — downloads the dataset (if needed), trains the model, exports metrics (`artifacts/metrics.json`), feature importances, and sample inference data.
3. `predict.sh` — loads the saved model artifacts and scores inputs, producing `artifacts/predictions.csv` with churn probabilities.

All scripts mount the repository and `~/.cache/cvlization` to comply with CVlization's centralized caching pattern.
