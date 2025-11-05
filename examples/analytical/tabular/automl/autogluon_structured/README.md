# AutoGluon Structured AutoML (Telco Churn)

This example demonstrates an AutoML workflow with [AutoGluon](https://auto.gluon.ai/) on the Telco Customer Churn dataset. AutoGluon automatically explores ensembles, bagging, and multi-layer stacking to deliver strong tabular baselines with minimal code.

## Dataset

- **Telco Customer Churn** – IBM's public dataset: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Quickstart

```bash
# Build container (uses official AutoGluon CPU image)
cvl run autogluon-structured build

# Train AutoML models with 10-minute budget
cvl run autogluon-structured train

# Score new customer records
cvl run autogluon-structured predict
```

Training pipeline steps:
1. Download & clean Telco churn data, convert labels to binary.
2. Split train/test, run `TabularPredictor.fit` with `presets="best_quality"` and a configurable time limit.
3. Export leaderboard, best model summary, metrics, and predictions to `artifacts/`.

Artifacts:
- `artifacts/model/` – AutoGluon predictor (all trained models + leaderboard)
- `artifacts/metrics.json` – ROC-AUC, accuracy, precision/recall/F1, best model
- `artifacts/leaderboard.csv` – full model leaderboard
- `artifacts/sample_input.csv` / `predictions.csv`

The example runs entirely on CPU in under ~10 minutes (adjust `TIME_LIMIT` if desired).
