# PyCaret Structured AutoML (Telco Churn)

This example shows how to use [PyCaret](https://pycaret.org/) to quickly build
and compare tabular churn models. The task is binary churn classification; the
default dataset is Telco churn, but you can plug in any dataset with a churn
label column. PyCaret handles preprocessing, model selection, and ensembling
with minimal code, making it a strong baseline for structured data projects.

## Dataset

- **Telco Customer Churn** – IBM GitHub mirror:
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Quickstart

```bash
# Build the container
cvl run churn-pycaret build

# Run AutoML training (default ~10 min budget)
cvl run churn-pycaret train

# Score new records (defaults to sample_input.csv)
cvl run churn-pycaret predict
```

Training pipeline:
1. Download & clean Telco churn data.
2. Initialize PyCaret classification setup, auto-compare models with
   `compare_models` under a time limit.
3. Save the best model, leaderboard, metrics, and sample predictions to
   `artifacts/`.

Artifacts:
- `artifacts/model/pycaret_best_model.pkl` – saved PyCaret model (alongside
  `config.json`)
- `artifacts/leaderboard.csv` – model comparison summary
- `artifacts/metrics.json` – evaluation metrics (ROC-AUC, accuracy, etc.)
- `artifacts/sample_input.csv` / `predictions.csv`

Runs entirely on CPU; adjust `TIME_LIMIT` in `train.py` if you need faster or
longer searches.
