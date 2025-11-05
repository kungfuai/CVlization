# AutoFE Structured Feature Pipeline (Telco Churn)

This example showcases automated feature engineering for tabular datasets. We
work with the Telco Customer Churn data, generate deep features using
Featuretools, and train a Gradient Boosting model with a reproducible feature
transformation pipeline.

## Dataset

- **Telco Customer Churn** – hosted at IBM’s public repo:
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Quickstart

```bash
# Build the container
cvl run analytical-tabular-feature-engineering-autofe-structured build

# Run feature generation + training
cvl run analytical-tabular-feature-engineering-autofe-structured train

# Score new records (defaults to sample_input.csv)
cvl run analytical-tabular-feature-engineering-autofe-structured predict
```

Artifacts produced:

- `artifacts/features/raw_features.parquet` – cleaned raw feature table
- `artifacts/features/engineered_features.parquet` – Featuretools-enriched
  dataset
- `artifacts/model/feature_pipeline.pkl` & `gbm_classifier.pkl` – reusable
  transformation + model stack
- `artifacts/metrics.json` – AUC, accuracy, precision/recall, F1
- `artifacts/sample_input.csv` / `predictions.csv` – sample records and scored
  output

The training command runs entirely on CPU and finishes in under a minute.
