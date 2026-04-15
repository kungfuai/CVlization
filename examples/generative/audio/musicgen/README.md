# MusicGen

Text-to-music generation using Meta AudioCraft MusicGen.

The default model is `facebook/musicgen-small` for a smaller smoke-test footprint. You can pass another MusicGen model with `--model`, such as `facebook/musicgen-medium` or `facebook/musicgen-melody`.

## What to Expect

Running `cvl run musicgen predict` or `./predict.sh` generates a WAV file from a text prompt and saves it in your current working directory. The default output is `musicgen.wav`.

The first real model run downloads the selected MusicGen checkpoint from Hugging Face and caches it under `~/.cache/huggingface`. `facebook/musicgen-small` is the smallest default option; larger MusicGen models require more download time and VRAM. The default generation length is 10 seconds so command-line demos produce enough audio to evaluate.

The bundled sample input is a small melody audio file from the upstream AudioCraft repo. It is hosted in `zzsi/cvl` and downloaded lazily only when requested with `--melody sample`.

## Sample

**Input** — text prompt:

```text
upbeat electronic music with a warm bassline
```

**Optional melody input** — upstream AudioCraft Bach sample:

https://huggingface.co/datasets/zzsi/cvl/resolve/main/musicgen/sample_data/bach.mp3

**Output** — a WAV file such as `music.wav` or `musicgen.wav`.

## Usage

Build the Docker image:

```bash
./build.sh
```

Generate music from text:

```bash
./predict.sh \
  --text "bright cinematic electronic music with a steady pulse, warm bass, and crisp percussion" \
  --duration 10 \
  --seed 20260415 \
  --output music.wav
```

Generate from a text file:

```bash
./predict.sh --input prompt.txt --output music.wav
```

Use a melody-conditioned model:

```bash
./predict.sh \
  --model facebook/musicgen-melody \
  --text "cinematic strings and piano" \
  --melody sample \
  --output melody_music.wav
```

Run the smoke test:

```bash
./test.sh
```

The smoke test uses AudioCraft's small `debug` model and verifies sample-data download plus WAV writing. It does not download `facebook/musicgen-small`.

## Options

- `--text`: Prompt text.
- `--input`: Path to a text prompt file.
- `--output`: Output WAV path.
- `--model`: Hugging Face model id. Defaults to `facebook/musicgen-small`.
- `--duration`: Generated duration in seconds. Defaults to `10`.
- `--melody`: Optional melody audio for melody-capable models. Use `sample` to download the bundled Bach sample.
- `--temperature`: Sampling temperature. Use `0` for greedy decoding.
- `--top-k`: Top-k sampling value.
- `--top-p`: Top-p sampling value. `0` disables top-p.
- `--cfg-coef`: Classifier-free guidance coefficient.
- `--seed`: Optional random seed.
- `--device`: `auto`, `cuda`, `cpu`, or `mps`.

## Model Downloads and Caching

Model weights are downloaded lazily by AudioCraft and Hugging Face on first run. The scripts mount:

- `~/.cache/huggingface`
- `~/.cache/torch`

Subsequent runs reuse those caches.

Sample inputs are downloaded lazily from `zzsi/cvl` into `~/.cache/huggingface/cvl_data/musicgen/`.

## Attribution and License

This example uses Meta AudioCraft MusicGen from `facebookresearch/audiocraft`, pinned to commit `896ec7c`.

Attribution:

- AudioCraft and MusicGen: Meta Platforms, Inc. and affiliates, FAIR team of Meta AI.
- Source repository: https://github.com/facebookresearch/audiocraft
- Model: https://huggingface.co/facebook/musicgen-small
- Sample melody asset: `assets/bach.mp3` from the upstream AudioCraft repository, mirrored at `zzsi/cvl` for this CVL example.
- Paper: "Simple and Controllable Music Generation", https://arxiv.org/abs/2306.05284

License:

- AudioCraft source code is MIT licensed.
- Released MusicGen model weights are CC-BY-NC 4.0. Treat this example as research/internal-demo oriented unless model-weight licensing has been reviewed for the intended use.
- Do not imply sponsorship, endorsement, or approval by Meta.

## Notes

AudioCraft's published requirements pin an older Torch stack, so the Dockerfile installs runtime dependencies explicitly and installs AudioCraft with `--no-deps` to keep the PyTorch 2.9.1/CUDA 12.8 base image.

## References

- AudioCraft: https://github.com/facebookresearch/audiocraft
- MusicGen documentation: https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md
- MusicGen small model: https://huggingface.co/facebook/musicgen-small
