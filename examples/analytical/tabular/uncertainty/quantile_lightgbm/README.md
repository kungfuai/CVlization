# Quantile LightGBM (California Housing)

This example trains three LightGBM models with the quantile (pinball) loss to estimate the 10th, 50th, and 90th percentiles of California housing prices. Combining these quantiles provides prediction intervals for each record.

## Dataset
- [California Housing (California Department of Real Estate, 1990)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), accessed via `sklearn.datasets.fetch_california_housing`.

## Workflow
1. `build.sh` — builds a CPU Docker image with LightGBM and scikit-learn.
2. `train.sh` — downloads the dataset, fits three LightGBM quantile models (0.1/0.5/0.9), evaluates coverage/width, and writes artifacts under `artifacts/`.
3. `predict.sh` — loads the saved models and scaler to score new records, emitting `pred_10`, `pred_50`, and `pred_90` columns.

Scripts mount the repository and `~/.cache/cvlization` to follow the centralized caching strategy used throughout CVlization.
