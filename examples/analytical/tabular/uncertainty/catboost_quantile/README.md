# CatBoost Quantile Regression (California Housing)

This example trains CatBoost regressors with the quantile (pinball) loss to recover the 10th, 50th, and 90th percentiles of the California Housing target. These quantiles form an asymmetric prediction interval for each record.

## Dataset
- [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) via `sklearn.datasets.fetch_california_housing`.

## Workflow
1. `build.sh` — builds a CPU Docker image with CatBoost and scikit-learn.
2. `train.sh` — downloads the dataset, fits three CatBoost quantile models (0.1/0.5/0.9), evaluates coverage and width, and writes artifacts to `artifacts/`.
3. `predict.sh` — reloads the saved models to score new records, returning the three quantile columns (`pred_10`, `pred_50`, `pred_90`).

All scripts mount the repository and `~/.cache/cvlization` to follow the centralized caching convention used across CVlization.
