# Uni2TS Time-Series Fine-Tuning

Fine-tune Salesforce Uni2TS (Moirai) foundation models on GluonTS benchmark datasets. The example bundles a reproducible slice of the M4 hourly benchmark, prepares the data in the Uni2TS wide format, and launches the official `cli.train` Hydra configuration inside Docker.

## Prerequisites

- Docker with at least 16 GB RAM available (`--shm-size 16G` is configured in the helper scripts).
- GPU support (enabled by default, set `export CVL_ENABLE_GPU=0` to disable).
- Optional Hugging Face token (`HF_TOKEN`) for gated checkpoints.

## Quickstart

```bash
# From examples/analytical/time_series/uni2ts_finetune
./build.sh                    # Build the container image
./train.sh                    # Fine-tune on the default M4 hourly slice (GPU)
./train.sh --dataset m4_daily --max-series 64
./train.sh --model-config moirai_1.1_R_small --max-series 50 trainer.max_epochs=10

# Or use cvl CLI
cvl run analytical_uni2ts_finetune build
cvl run analytical_uni2ts_finetune train
```

Arguments after `train.sh` are forwarded directly to `train.py`. Available options:

- `--dataset`: GluonTS repository dataset (default `m4_hourly`)
- `--model-config`: Hydra model preset from `uni2ts/cli/conf/finetune/model` (default `moirai_1.0_R_small`)
- `--max-series`: Number of training series to materialize locally (default `100`)
- `--context-steps`: Number of timesteps kept per series (default `0` = full history). Set to positive value to truncate very long series.
- `--val-patch-size`: Patch size used when building validation windows (default `16`)
- `--val-context-length`: Context length for validation windows (default auto)
- Any additional `key=value` pairs (without a leading `--`) are passed straight to Hydra (e.g., `trainer.max_epochs=10` or `train_dataloader.batch_size=128`).

## Outputs

- `data/train_wide.csv`: Wide-format training slice materialized from GluonTS (created on demand).
- `artifacts/dataset_metadata.json`: Summary of the extracted series and prediction horizon.
- `artifacts/metrics.json`: Run metadata including the Hydra output directory and overrides.
- `artifacts/uni2ts_runs`: Hydra-managed fine-tuning outputs (checkpoints, logs).

## Notes

- GPU is enabled by default. Set `CVL_ENABLE_GPU=0` to run on CPU only.
- Set `HF_TOKEN` before invoking `train.sh` to access gated Uni2TS weights; values from `.env` are loaded automatically but never override already-exported variables.
- Default training settings: `max_epochs=5`, `num_batches_per_epoch=20`, `batch_size=32` for quick experimentation on small datasets. Increase batch size (e.g., `train_dataloader.batch_size=128`) for larger datasets and production runs.
- Mount caches (`~/.cache/cvlization`, `~/.cache/huggingface`) persist across runs for faster builds.
- Use `CVL_IMAGE` to override the build/tag name if integrating with CI.
- `--context-steps=0` (default) uses full series history. For very long series (e.g., >10k timesteps), consider truncating with `--context-steps=1000` to reduce memory usage.
