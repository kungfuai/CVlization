# PyMC Bayesian Regression (Prediction Intervals)

This example fits a Bayesian linear regression model with PyMC on the California Housing dataset, estimates posterior distributions for the coefficients, and serves calibrated prediction intervals via posterior predictive sampling.

## Dataset

- **Name**: California Housing (Scikit-learn)
- **Source**: [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- **Caching**: Downloaded via scikit-learn and stored in the standard sklearn data cache (typically `~/scikit_learn_data`)

## Workflow

```bash
# From repo root
cvl run pymc-bayesian-regression build
cvl run pymc-bayesian-regression train
cvl run pymc-bayesian-regression predict -- \
  --input artifacts/sample_input.csv \
  --output outputs/predictions.csv \
  --include-features
```

- `train.py` standardizes the features/target, builds a PyMC model with Normal priors on the coefficients, runs NUTS sampling, saves posterior draws, and evaluates coverage against a 90% interval on the test split.
- `predict.py` reloads the scalers and posterior samples to generate posterior predictive summaries (mean, median, std, and 5–95% credible band) for arbitrary CSV inputs.

## Artifacts

- `artifacts/models/posterior_samples.npz` – Posterior draws for coefficients, intercept, and noise scale
- `artifacts/models/feature_scaler.joblib` & `target_scaler.joblib` – Standardization transforms for features and target
- `artifacts/posterior_inference.nc` – Serialized `InferenceData` from PyMC/ArviZ (optional inspection)
- `artifacts/metrics.json` – RMSE, MAE, R², and 90% interval coverage statistics on the test set
- `artifacts/sample_input.csv` / `artifacts/sample_predictions.csv` – Example input rows and posterior summaries

## References

- [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/)
- [ArviZ for Bayesian diagnostics](https://python.arviz.org/en/stable/)
