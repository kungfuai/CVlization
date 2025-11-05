# Merlion Anomaly Dashboard

Containerized Dash web application for Salesforce's [Merlion](https://github.com/salesforce/Merlion) anomaly and forecasting dashboard. The image bundles the optional Merlion `dashboard` extras, Java runtime support, and a gunicorn entrypoint so you can explore anomaly detectors or forecasters directly from your browser.

## Datasets

The dashboard loads datasets from the container filesystem. Two easy starting points:

- [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) – the data bundle Merlion uses in its tutorials. It is also available through the `ts-datasets` Python package bundled with Merlion.
- Any CSV file with timestamp and value columns; mount a host directory with your files into `/workspace/data` when running the container.

## Workflow

```bash
# From repository root
cvl run merlion-anomaly-dashboard build
cvl run merlion-anomaly-dashboard smoke-test  # smoke-test the dashboard server
cvl run merlion-anomaly-dashboard serve      # launch the UI (Ctrl+C to stop)
```

- `smoke.sh` runs `verify.py`, which hits the underlying Flask test client endpoints to ensure the dashboard boots correctly.
- `serve.sh` starts gunicorn on port 8050. Use `MERLION_DASHBOARD_PORT=9000` to publish a different host port, and mount data via `--mount type=bind,src=/path/to/data,dst=/workspace/data`.

## References

- Salesforce Merlion repository: <https://github.com/salesforce/Merlion>
- Merlion dashboard docs: <https://opensource.salesforce.com/Merlion/merlion.dashboard.html>
- NAB dataset: <https://github.com/numenta/NAB>
