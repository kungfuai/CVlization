# PyCox Retention Survival Modeling (SUPPORT)

This example demonstrates customer/patient retention analysis with
[PyCox](https://github.com/havakv/pycox). We train a DeepSurv-style CoxPH model
on the SUPPORT survival dataset, export calibrated survival curves, and produce
per-horizon retention probabilities for analytics teams.

## Dataset

- **SUPPORT (Study to Understand Prognoses Preferences Outcomes & Risks of
  Treatments)** – available directly through the PyCox dataset loaders.
  Documentation: https://hypersphere.ai/pycox/

## Quickstart

```bash
# Build the container image
cvl run analytical-tabular-survival-pycox-retention build

# Train DeepSurv-style CoxPH model with PyCox
cvl run analytical-tabular-survival-pycox-retention train

# Score new cohorts (defaults to saved sample inputs)
cvl run analytical-tabular-survival-pycox-retention predict
```

Training loads the SUPPORT dataset, one-hot encodes categorical covariates,
scales numeric features, and fits a PyTorch MLP backbone optimized with the
Cox proportional hazards loss. Outputs include:

- `artifacts/model/` – PyCox weight file, lab-transform state, preprocessing
  config, and scaler
- `artifacts/metrics.json` – concordance index, integrated Brier score, and
  negative log-likelihood on the hold-out split
- `artifacts/sample_survival_curves.csv` – survival curves for sample patients
- `artifacts/survival_horizons.csv` – survival probabilities at 180/365/730-day
  horizons for quick reporting
- `artifacts/predictions.csv` – inference results with survival probabilities
  for supplied inputs

All steps run on CPU in under a couple of minutes.
