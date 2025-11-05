# DoWhy Berkeley Gender Bias Analysis

Recreates the classic UC Berkeley graduate admissions study (1973) to illustrate Simpson’s paradox and the value of causal adjustment. We encode the published admission counts by gender and department, use DoWhy to estimate the causal effect of gender on admission decisions, and compare the naïve gap with an adjusted estimate that conditions on department choice.

## Dataset

- **Source**: Aggregated admission counts from Bickel, Hammel, & O’Connell (1975), “Sex Bias in Graduate Admissions: Data from Berkeley.”
- **Structure**: Six departments (`A`–`F`), each with male/female admission and rejection counts.
- The training script expands the counts into a row-level dataset (≈4500 records) for causal estimation.

## Workflow

- `cvl run dowhy-berkeley-bias run` – full pipeline (discovery → identification → estimation → refutation).
- `cvl run dowhy-berkeley-bias discovery` – descriptive statistics only.
- `cvl run dowhy-berkeley-bias identify` – construct graph and list estimands.
- `cvl run dowhy-berkeley-bias estimate` – estimate causal effect without refutation.
- `cvl run dowhy-berkeley-bias refute` – estimate and apply a random common-cause refuter.
- `cvl run dowhy-berkeley-bias report -- --input artifacts/sample_input.csv --output outputs/per_department.csv --summary outputs/summary.json` – recompute metrics for the sample counts (replace paths to audit custom data).

- `train.py` materializes the dataset and executes the requested stage of the DoWhy pipeline, emitting metrics and department-level summaries.
- `predict.py` accepts an aggregated dataset (`department, gender, admitted, count`) to recompute fairness metrics and admission rate gaps.

## Artifacts

- `artifacts/metrics.json` – Naïve admission rates, causal effect estimate, and Simpson’s paradox gap.
- `artifacts/metadata.json` – Dataset provenance, graph structure, and modeling notes.
- `artifacts/sample_input.csv` – Berkeley admission counts ready for reuse or modification.
- `artifacts/sample_predictions.csv` – Department-level admission rates and gaps.

## References

- Bickel, P. J., Hammel, E. A., & O’Connell, J. W. (1975). *Sex Bias in Graduate Admissions: Data from Berkeley*. Science, 187(4175), 398–404.
- DoWhy documentation: [https://microsoft.github.io/dowhy/](https://microsoft.github.io/dowhy/)
