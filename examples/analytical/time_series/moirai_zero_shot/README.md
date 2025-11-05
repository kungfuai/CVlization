# Moirai Zero-Shot Forecasting

Zero-shot time series forecasting with Salesforce's Moirai foundation model using the [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts) library. The container downloads the `Salesforce/moirai-1.1-R-small` checkpoint from Hugging Face, samples a handful of series from the M4 hourly dataset, and evaluates the model without any fine-tuning.

## Dataset

- **Source**: [M4 competition (hourly subset)](https://www.m4.unic.ac.cy/). GluonTS downloads the processed splits the first time the example runs.
- **Frequency**: Hourly (`freq="H"`).
- **Horizon**: 48 steps (default competition horizon), configurable via CLI.

## Workflow

```bash
# From repository root
cvl run moirai-zero-shot build
cvl run moirai-zero-shot forecast -- --dataset m4_hourly --model-size small --windows 2
```

- `build.sh` constructs the CPU image with Uni2TS, GluonTS, and PyTorch.
- `forecast.sh` runs `forecast.py`, which:
  - Selects a few hourly series, performs a rolling evaluation, and
  - Loads the requested Moirai checkpoint via `MoiraiModule.from_pretrained`.
  - Writes `artifacts/metrics.json`, `artifacts/sample_forecast.csv`, and a preview plot.

To reuse the downloaded weights between runs, Hugging Face caches are mounted at `${HOME}/.cache/huggingface` by the wrapper script.

## References

- Salesforce Research, *Moirai: Unified Foundation Models for Time Series Forecasting* ([arXiv:2402.02592](https://arxiv.org/abs/2402.02592))
- Uni2TS library: <https://github.com/SalesforceAIResearch/uni2ts>
- Hugging Face model card: <https://huggingface.co/Salesforce/moirai-1.1-R-small>
- M4 competition data: <https://www.m4.unic.ac.cy/>
