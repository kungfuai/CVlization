# PyOD Credit Card Fraud Detection

This example trains an anomaly detection pipeline for credit-card transactions
using [PyOD](https://github.com/yzhao062/pyod). We fine-tune an Isolation
Forest detector on a highly imbalanced fraud dataset, evaluate precision/recall
metrics for the fraud class, and export per-transaction anomaly scores for
investigation dashboards.

## Dataset

- **Credit Card Fraud Detection (European cardholders)** – public dataset
  containing 284,807 transactions with anonymised PCA components. Downloaded
  from the TensorFlow public mirror:
  https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

## Quickstart

```bash
# Build the container image
cvl run pyod-fraud build

# Train the PyOD Isolation Forest detector + scaler
cvl run pyod-fraud train

# Score new records (defaults to saved sample inputs)
cvl run pyod-fraud predict
```

The training script downloads the dataset, downsamples the majority class to a
manageable size (configurable via `MAX_NORMAL_SAMPLES`), fits a standardized
Isolation Forest detector, and saves:

- `artifacts/model/iforest_detector.pkl` – trained PyOD model
- `artifacts/model/scaler.pkl` – StandardScaler used for inference
- `artifacts/metrics.json` – ROC-AUC, PR-AUC, precision/recall/F1 on the test
  split
- `artifacts/predictions.csv` – hold-out predictions with anomaly scores and
  labels for downstream analysis
- `artifacts/sample_input.csv` – sample feature rows to drive the prediction
  script
- `artifacts/classification_report.json` / `confusion_matrix.json` – evaluation
  diagnostics for auditability

All commands run on CPU; training completes in seconds with the default sample
cap (80k normal + all fraud examples).
