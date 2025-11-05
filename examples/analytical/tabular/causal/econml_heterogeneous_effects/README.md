# EconML Linear Doubly Robust Heterogeneous Effects

This example demonstrates how to estimate heterogeneous treatment effects (CATE) using EconML’s `LinearDRLearner` on a synthetic tabular dataset with known counterfactual outcomes. The pipeline quantifies the gap between predicted and true treatment effects, evaluates policy value, and saves reusable artifacts for downstream what-if analysis.

## Dataset

- **Type**: Synthetic uplift dataset with non-linear propensities and heterogeneous treatment effects.
- **Generation**: The training script programmatically simulates features, outcomes, and counterfactual outcomes so model quality can be measured against ground truth.
- **Size**: Configurable via environment variables (`ECONML_SAMPLES`, `ECONML_FEATURES`); defaults to 8,000 rows × 12 features.

## Usage

```bash
# From repository root
cvl run econml-heterogeneous-effects build
cvl run econml-heterogeneous-effects train
cvl run econml-heterogeneous-effects predict -- \
  --input artifacts/sample_input.csv \
  --output outputs/cate_predictions.csv \
  --include-features
```

- `train.py` synthesizes the dataset, fits a doubly-robust learner, logs CATE/ATE metrics, and stores sample predictions plus the trained model.
- `predict.py` reloads the model/metadata to produce CATE estimates, confidence intervals, and policy recommendations for custom tabular inputs.

## Artifacts

- `artifacts/models/econml_linear_dr.joblib` – Trained EconML learner with embedded first-stage models.
- `artifacts/metrics.json` – RMSE between predicted and true treatment effects, ATE error, and policy value comparisons.
- `artifacts/sample_input.csv` / `artifacts/sample_predictions.csv` – Sample rows with true vs. predicted effects and treatment recommendations.
- `artifacts/metadata.json` – Synthetic dataset configuration and feature schema.

## References

- [PyWhy EconML](https://github.com/py-why/EconML)
- [Doubly Robust Learners](https://econml.azurewebsites.net/spec/dml.html)
