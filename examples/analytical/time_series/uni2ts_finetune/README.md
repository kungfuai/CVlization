# Uni2TS Time-Series Fine-Tuning

Fine-tune Salesforce Uni2TS (Moirai) foundation models on GluonTS benchmark datasets. The example bundles a reproducible slice of the M4 hourly benchmark, prepares the data in the Uni2TS wide format, and launches the official `cli.train` Hydra configuration inside Docker.

## Prerequisites

- Docker with at least 16 GB RAM available (`--shm-size 16G` is configured in the helper scripts).
- Optional GPU support (`export CVL_ENABLE_GPU=1`) and a Hugging Face token (`HF_TOKEN`) for gated checkpoints.

## Quickstart

```bash
# From examples/analytical/time_series/uni2ts_finetune
./build.sh                    # Build the container image
./train.sh                    # Fine-tune on the default M4 hourly slice (CPU)
./train.sh --dataset m4_daily --max-series 64
./train.sh --model-config moirai_1.1_R_small --max-series 8 trainer.max_epochs=2
```

Arguments after `train.sh` are forwarded directly to `train.py`. Available options:

- `--dataset`: GluonTS repository dataset (default `m4_hourly`)
- `--model-config`: Hydra model preset from `uni2ts/cli/conf/finetune/model` (default `moirai_1.0_R_small`)
- `--max-series`: Number of training series to materialize locally (default `100`)
- `--context-steps`: Number of timesteps kept per series before patchification (default `384`, set `<=0` for full history)
- `--val-patch-size`: Patch size used when building validation windows (default `16`)
- `--val-context-length`: Context length for validation windows (default auto)
- Any additional `key=value` pairs (without a leading `--`) are passed straight to Hydra (for example `trainer.max_epochs=3` or `train_dataloader.batch_size=64`).

## Outputs

- `data/train_wide.csv`: Wide-format training slice materialized from GluonTS (created on demand).
- `artifacts/dataset_metadata.json`: Summary of the extracted series and prediction horizon.
- `artifacts/metrics.json`: Run metadata including the Hydra output directory and overrides.
- `artifacts/uni2ts_runs`: Hydra-managed fine-tuning outputs (checkpoints, logs).

## Notes

- Set `HF_TOKEN` before invoking `train.sh` to access gated Uni2TS weights; values from `.env` are loaded automatically but never override already-exported variables.
- The helper scripts cap the run to `trainer.max_epochs=1`, `train_dataloader.num_batches_per_epoch=10`, `train_dataloader.batch_size=32` for a quick CPU-only sanity check. Override them via additional Hydra arguments when you need longer runs.
- Mount caches (`~/.cache/cvlization`, `~/.cache/huggingface`) persist across runs for faster builds.
- Use `CVL_IMAGE` to override the build/tag name if integrating with CI.
- The training preset skips validation by default; Lightning will warn about missing `val/PackedNLLLoss`. Pass a `val_data=...` config if you need evaluation.
