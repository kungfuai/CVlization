# LiteAvatar Docker Example

This example packages the [HumanAIGC LiteAvatar](https://github.com/HumanAIGC/lite-avatar) project for use through the `cvl` runner.
It turns a Mandarin WAV file into a lip-synced 2D avatar video; the first run automatically fetches the upstream demo assets.

## Quickstart

```bash
./build.sh
./predict.sh
```

During the first prediction the helper downloads:
- LiteAvatar sample assets from the upstream GitHub release (cached under `data/`)
- ModelScope checkpoints via `download_model.sh` (cached under `weights/` + `~/.cache/modelscope`)

Once assets exist locally, subsequent runs launch immediately and write the MP4 to `outputs/test_demo.mp4`.

## Options

- Pass extra flags to the Python entrypoint via `./predict.sh -- --audio-file /path/to.wav --result-dir /workspace/outputs`.
- Set `LITE_AVATAR_USE_GPU=1` before running `predict.sh` to attach the host GPU (`--use-gpu` can also be forwarded to the python script).

## Assets

- `data/` is populated lazily with the upstream `sample_data.zip`.
- `weights/` includes the lightweight starter checkpoints; the Python helper downloads heavier files on demand.
- `funasr_local/` and related scripts are copied from the upstream repo under the MIT license.
