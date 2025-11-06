# DoWhy Policy Uplift (Lalonde Study)

This example demonstrates causal uplift modelling for policy decisions. We fit
an EconML DR-Learner backed by DoWhy on the Lalonde job-training dataset to
estimate the incremental impact of the treatment program.

## Dataset

- **Lalonde Job Training Program** – observational dataset distributed with the
  `causaldrf` R package. We load it via `statsmodels.datasets.get_rdataset`.

## Quickstart

```bash
# Build container
cvl run dowhy-uplift build

# Train uplift model and estimate treatment effect
cvl run dowhy-uplift train

# Score new cohorts for program rollout
cvl run dowhy-uplift predict
```

Training flow:

1. Load Lalonde data (treatment indicator `treat`, outcome `re78`).
2. Split into train/test, standardise continuous covariates.
3. Fit a double-robust learner (random forest outcomes, logistic propensity).
4. Estimate the average treatment effect with DoWhy for validation.
5. Export uplift scores, policy recommendations, and model artifacts.

Artifacts:

- `artifacts/model/dr_learner.pkl` – EconML DR-Learner for individual uplift
- `artifacts/model/propensity_model.pkl` – logistic propensity estimator
- `artifacts/metrics.json` – ATE (DoWhy & DR), empirical uplift, propensity AUC
- `artifacts/predictions.csv` – hold-out uplift scores with treatment ranks
- `artifacts/sample_input.csv` – sample covariate rows for inference checks

All steps run on CPU in under a minute.
