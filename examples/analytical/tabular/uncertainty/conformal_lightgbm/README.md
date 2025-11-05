# LightGBM Conformal Calibration

This example trains a LightGBM classifier on the German Credit (OpenML id: 31) dataset, applies post-hoc probability calibration (Platt/sigmoid and isotonic), and derives conformal prediction sets with guaranteed coverage. The pipeline demonstrates how to package calibrated LightGBM workflows that expose both well-calibrated probabilities and set-valued predictions.

## Dataset

- **Name**: German Credit (OpenML id: 31)
- **Source**: [https://www.openml.org/d/31](https://www.openml.org/d/31)
- **Caching**: Downloaded once and stored under `~/.cache/cvlization/data/openml_credit_g/credit_g.csv`

## Workflow

```bash
# From repository root
cvl run conformal-lightgbm build
cvl run conformal-lightgbm train
cvl run conformal-lightgbm predict -- \
  --input artifacts/sample_input.csv \
  --output outputs/predictions.csv \
  --include-features
```

- `train.py` downloads/caches the dataset, trains LightGBM with categorical support, fits Platt (sigmoid) and isotonic calibrators, and computes conformal sets targeting 90% coverage (`alpha=0.1`).
- `predict.py` loads the saved artifacts to produce base, sigmoid, and isotonic probabilities. It also returns conformal prediction sets for the requested miscoverage level (`--alpha`, defaults to 0.1).

## Artifacts

- `artifacts/models/lightgbm_classifier.joblib` – base LightGBM model
- `artifacts/models/sigmoid_calibrator.joblib` – Platt-calibrated estimator
- `artifacts/models/isotonic_calibrator.joblib` – Isotonic-calibrated estimator
- `artifacts/models/preprocessor.json` – Column order, categorical schema, class labels
- `artifacts/models/conformal_metadata.json` – Stored calibration scores and threshold for conformal sets
- `artifacts/metrics.json` – Accuracy, ROC-AUC, log-loss, Brier score, and conformal coverage statistics
- `artifacts/sample_input.csv` / `artifacts/sample_predictions.csv` – Example inputs and outputs for quick inspection

## References

- [OpenML German Credit dataset](https://www.openml.org/d/31)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [Conformal prediction tutorial (Angelopoulos & Bates, 2023)](https://arxiv.org/abs/2107.07511)
