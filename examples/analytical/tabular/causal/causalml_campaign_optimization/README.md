# CausalML Campaign Optimization

Simulates a multi-treatment marketing campaign and fits per-treatment uplift models using CausalML meta-learners. The pipeline generates synthetic customer features, assigns control vs. three promotional channels, and learns incremental lift to optimize targeting policies.

## Dataset

- **Type**: Synthetic uplift data with a control arm and three marketing treatments (`email`, `display`, `social`).
- **Generation**: Logistic baseline conversion probabilities plus treatment-specific uplift functions; ground-truth effects are recorded for evaluation.
- **Size**: Configurable via environment variables (`CAMPAIGN_SAMPLES`, `CAMPAIGN_FEATURES`). Defaults to 10k customers × 12 features.

## Workflow

```bash
# From repository root
cvl run causalml-campaign-optimization build
cvl run causalml-campaign-optimization train
cvl run causalml-campaign-optimization predict -- \
  --input artifacts/sample_input.csv \
  --output outputs/uplift_predictions.csv \
  --include-features
```

- `train.py` synthesizes the dataset, fits a one-vs-control `BaseXRegressor` uplift model per treatment, and logs metrics such as RMSE of CATE estimates and policy value vs. random/oracle strategies.
- `predict.py` reloads the trained models to score arbitrary feature tables, returning per-treatment uplift estimates and the recommended channel.

## Artifacts

- `artifacts/models/uplift_models.joblib` – Serialized dictionary of CausalML meta-learners for each treatment arm.
- `artifacts/metrics.json` – CATE RMSE per treatment plus policy evaluation metrics (mean gain vs. random/oracle).
- `artifacts/sample_input.csv` / `artifacts/sample_predictions.csv` – Example customers with observed outcomes, predicted uplift, and true uplift (for inspection).

## References

- [CausalML documentation](https://github.com/uber/causalml)
- Gutierrez & Gérardy (2016), *Causal Inference and Uplift Modeling* – background on uplift methodologies.
