# GBT Credit Default Risk Scoring

This example trains a gradient-boosted tree model (via the [`gbt`](https://pypi.org/project/gbt/) wrapper on top of LightGBM) to predict credit default risk using the German Credit dataset.

## Data Sources
- Dataset: [German Credit Data Set — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) *(mirrored CSV fetched from https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv)*
- `gbt` library: [https://github.com/zzsi/gbt](https://github.com/zzsi/gbt)

## Workflow
1. `build.sh` — builds a CPU Docker image with LightGBM and gbt 0.3 from PyPI.
2. `train.sh` — downloads the dataset, trains the model with class imbalance handling, and writes metrics (`artifacts/metrics.json`), feature importances, and sample inputs.
3. `predict.sh` — loads the saved model and scores borrower records, producing `artifacts/predictions.csv` with default probabilities.

All scripts mount `~/.cache/cvlization` inside the container to align with CVlization's centralized caching strategy.
