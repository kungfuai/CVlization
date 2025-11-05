# GBT Telco Churn

This example trains a gradient-boosted tree churn model (using the [`gbt`](https://pypi.org/project/gbt/) wrapper over LightGBM) on IBM's Telco Customer Churn dataset.

## Data Sources
- Dataset: [Telco Customer Churn (IBM) — Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) *(mirrored CSV retrieved from IBM's GitHub sample: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)*
- `gbt` library: [https://github.com/zzsi/gbt](https://github.com/zzsi/gbt)

## Workflow
1. `build.sh` — builds a CPU Docker image with LightGBM and gbt 0.3 from PyPI.
2. `train.sh` — downloads the Telco dataset, trains the model, saves metrics to `artifacts/metrics.json`, plus model artifacts and sample inputs.
3. `predict.sh` — loads the saved model and generates churn probabilities for sample records, writing to `artifacts/predictions.csv`.

The scripts follow CVlization's centralized caching pattern by mounting `~/.cache/cvlization` into the container.
